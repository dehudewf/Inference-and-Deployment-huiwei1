# Day 3: Continuous Batching + FlashAttention — 学习笔记

> 目标：理解调度策略和底层计算优化
> 今日关键词：Static vs Continuous Batching、三队列调度、Tiling、Online Softmax、SRAM/HBM

---

## 1. Continuous Batching 深入理解

### 1.1 Static Batching 的问题

```
传统Static Batching:

Batch = [请求A(10 tokens), 请求B(50 tokens), 请求C(200 tokens)]

时间线:
  Step 1-10:   A运行, B运行, C运行
  Step 10:     A完成! 但必须等B和C...
  Step 11-50:  A空转(Padding), B运行, C运行
  Step 50:     B完成! 但必须等C...
  Step 51-200: A空转, B空转, C运行

问题:
  1. 短请求等长请求 → GPU空转严重
  2. 新请求无法插入，必须等当前batch全部完成
  3. GPU利用率低（大量Padding浪费算力）
```

### 1.2 Continuous Batching: Token级调度

**核心思想**：不再以"请求"为单位调度，而是以**每个token生成步**为调度粒度。

```
Continuous Batching:

Step 1-10:   [A, B, C] 同时运行
Step 10:     A完成 → 立即移出, 新请求D插入!
Step 11-50:  [D, B, C] 同时运行
Step 50:     B完成 → 立即移出, 新请求E插入!
Step 51:     [D, E, C] 同时运行
...

优势:
  1. 完成即走，新请求即来 → 零等待
  2. GPU始终满载 → 利用率大幅提升
  3. 吞吐量提升 2-3x
```

### 1.3 三个队列协作机制

```
┌──────────────────────────────────────────────────────────┐
│                    调度器 (Scheduler)                      │
│                                                          │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐             │
│   │ Waiting  │ →→ │ Running  │ →→ │ Complete │ → 返回结果 │
│   │  Queue   │    │  Queue   │    │         │             │
│   └────┬────┘    └────┬────┘    └─────────┘             │
│        │              │                                  │
│        │              ↕                                  │
│        │         ┌─────────┐                             │
│        │         │ Swapped  │                             │
│        │         │  Queue   │                             │
│        │         └─────────┘                             │
│        │                                                 │
│  每个调度周期:                                            │
│  1. 检查Waiting队列，是否有新请求可以加入Running          │
│  2. 检查Running队列，移出已完成的请求                     │
│  3. 如果显存不足，将低优先级请求Swap-out到CPU内存          │
│  4. 如果显存充裕，将Swapped队列的请求Swap-in回GPU         │
└──────────────────────────────────────────────────────────┘

队列详解:

Waiting（等待队列）:
  - 新到达但尚未分配显存的请求
  - 等待调度器分配KV Cache Block
  - 进入Running时执行Prefill

Running（运行队列）:
  - 当前在GPU上进行推理的请求
  - 每步生成一个token (Decode)
  - 完成时移出，释放KV Cache Block

Swapped（交换队列）:
  - 显存不足时被换出的请求
  - KV Cache从GPU显存 → CPU内存（异步Swap-out）
  - 显存空闲时从CPU内存 → GPU显存（异步Swap-in）
  - 不丢失计算进度，只是暂停
```

### 1.4 调度策略：FCFS vs Priority

```
FCFS（先来先服务）:
  - 按到达顺序处理
  - 简单公平

Priority（优先级）:
  - Prefill优先 or Decode优先
  - 实际部署中常见：短请求优先（减少延迟）
  - KsanaLLM的实现：ProcessWaitingQueue + ProcessDecodingQueue 分别处理
```

---

## 2. FlashAttention 原理

### 2.1 传统Attention的问题

```
标准Attention计算: Attention(Q, K, V) = softmax(Q × K^T / √d) × V

步骤分解（GPU执行）:
  Step 1: S = Q × K^T          → 需要存储 N×N 矩阵到 HBM     ← O(N²) 显存!
  Step 2: P = softmax(S)       → 从 HBM 读取 S，计算后写回 HBM
  Step 3: O = P × V            → 从 HBM 读取 P，计算后写回 HBM

问题:
  1. 中间矩阵 S 的大小 = N² → 序列越长，显存占用越大
  2. 频繁读写 HBM（慢! 2TB/s）
  3. GPU计算单元等待数据搬运 → IO瓶颈

例: N=2048, d=128, FP16
  S矩阵大小 = 2048 × 2048 × 2 bytes = 8MB (每个head)
  × 32 heads × 40 layers = 10GB+ 的中间结果读写!
```

### 2.2 GPU内存层次（必记关键数字！）

```
┌─────────────────────────────────────────────────────────┐
│                    GPU 内存层次结构                        │
│                                                         │
│   ┌───────────┐                                         │
│   │  寄存器    │  最快，但容量极小                         │
│   └─────┬─────┘                                         │
│         ↓                                               │
│   ┌───────────┐                                         │
│   │  SRAM     │  容量: ~20MB (A100)                      │
│   │ (Shared   │  带宽: ★★★ 19 TB/s ★★★                  │
│   │  Memory)  │  特点: 片上存储，极快但小                  │
│   └─────┬─────┘                                         │
│         ↓    ← FlashAttention的关键: 中间结果留在这里!    │
│   ┌───────────┐                                         │
│   │  HBM      │  容量: 40/80GB (A100)                    │
│   │ (显存)    │  带宽: ★ 2 TB/s ★                        │
│   │           │  特点: 容量大但带宽是SRAM的1/10           │
│   └───────────┘                                         │
│                                                         │
│   SRAM带宽 / HBM带宽 = 19/2 ≈ 10倍差距!                 │
│   → 减少HBM访问 = 巨大性能提升                            │
└─────────────────────────────────────────────────────────┘
```

### 2.3 Tiling分块计算（FlashAttention核心思想）

```
核心思想: 不存储完整的 N×N 矩阵，分块在 SRAM 中计算

传统方式:
  Q(全部) × K^T(全部) → S(N×N, 存HBM) → softmax → P(N×N, 存HBM) → P × V → O

FlashAttention:
  将 Q 分成 Tr 块, K/V 分成 Tc 块
  
  for 每个Q块 Qi:                    ← 外层循环
    for 每个K块 Kj, V块 Vj:          ← 内层循环
      1. 从HBM加载 Qi, Kj, Vj 到 SRAM
      2. 在SRAM中计算: Sij = Qi × Kj^T       ← 只有小块，放得下SRAM
      3. 在SRAM中计算: Pij = softmax(Sij)     ← 使用Online Softmax
      4. 在SRAM中计算: Oi += Pij × Vj         ← 累加到输出
      5. 不把 Sij 和 Pij 写回 HBM!            ← 节省IO的关键
    end
    将最终 Oi 写回 HBM                        ← 只写结果
  end

效果:
  - 显存: O(N²) → O(N)    ← 不需要存完整的S和P矩阵
  - IO: 大幅减少HBM读写    ← 中间结果在SRAM
  - 速度: 加速 2-4x
  - 显存节省: 5-20x
```

### 2.4 Online Softmax（分块计算的关键难题）

```
问题: softmax 需要全局信息!
  softmax(xi) = exp(xi) / Σ exp(xj)

  如果分块计算，每块只看到部分数据，怎么算全局softmax？

解决: Online Softmax —— 边算边修正

处理第1块:
  max₁ = max(S_block1)
  sum₁ = Σ exp(S_block1 - max₁)
  O₁ = (exp(S_block1 - max₁) / sum₁) × V_block1

处理第2块:
  max₂ = max(max₁, max(S_block2))           ← 更新全局max
  
  # 修正第1块的结果（因为max变了）
  correction = exp(max₁ - max₂)
  sum₂ = sum₁ × correction + Σ exp(S_block2 - max₂)  ← 更新全局sum
  
  O₂ = O₁ × (sum₁ × correction / sum₂)     ← 缩放修正旧结果
     + (exp(S_block2 - max₂) / sum₂) × V_block2  ← 加入新块贡献

关键洞察:
  利用 exp(a-b) = exp(a)/exp(b) 的性质
  当全局max更新时，只需乘一个缩放因子修正旧结果
  → 数学上完全等价于一次性计算全局softmax
  → 但不需要存储完整的N×N矩阵!
```

### 2.5 flash-attention-opt项目性能数据（面试可用）

```
项目: 基于CUDA实现的FlashAttention优化实践

测试环境: NVIDIA RTX 4060
测试配置: [bs=32, nh=8, N, M, d]

版本    | [256,256,256] | [1024,1024,256] | 核心优化
--------|---------------|-----------------|------------------
Baseline| 46.8ms        | 92.7ms          | cuBLAS标准实现
Minimal | 741ms         | 11343ms         | 最基础的分块版本
v1      | 104ms         | 1524ms          | 基础Tiling
v2      | 127ms         | 2026ms          | 改进内存访问
v3      | 12.4ms        | 167ms           | Warp优化+消除Bank Conflict

v3 vs Minimal: 性能提升 8-14 倍!
v3 vs Baseline: 短序列更快，长序列略慢（cuBLAS有自适应kernel选择）
但v3的显存优势巨大: 比Baseline少占约50%显存（不存N×N中间矩阵）
```

---

## 3. 今日自检清单

- [x] 能画出Continuous Batching调度流程（三队列 + token级调度）
- [x] 能解释FlashAttention为什么快（一句话版本 + 详细版本）
- [x] 记住带宽数字：SRAM 19TB/s，HBM 2TB/s
- [x] 理解Online Softmax的核心思想（边算边修正，指数缩放）
- [x] 能说出flash-attention-opt的性能数据

---

## 4. 面试标准答案

### Q: Continuous Batching是什么？

> "传统Static Batching要求batch内所有请求同时完成，短请求等长请求，GPU空转严重。Continuous Batching是token级调度——每生成一个token就可以调整batch组成，完成的请求立即移出，新请求可以插入。维护三个队列：Waiting（等待分配显存）、Running（正在GPU推理）、Swapped（显存不足时KV Cache换出到CPU内存）。调度器每个Decode步都会重新调度。好处是GPU利用率大幅提升，吞吐量提高2-3倍。"

### Q: FlashAttention为什么快？

> **一句话版本**：
> "传统Attention需要把N×N的中间矩阵存到HBM，FlashAttention通过Tiling分块计算，中间结果保存在SRAM（带宽19TB/s，是HBM 2TB/s的近10倍），避免了大量HBM读写。"

> **详细版本**：
> "FlashAttention解决了三个问题：一是显存问题——传统Attention需要O(N²)显存存中间矩阵，FlashAttention降到O(N)；二是IO问题——通过Tiling把Q、K、V分成小块在SRAM中计算，不把中间结果写回HBM；三是数值正确性——用Online Softmax算法，在分块计算时维护全局的max和sum，通过指数缩放修正之前的结果，数学上完全等价。最终实现2-4倍加速，显存减少5-20倍。"

### Q: Online Softmax怎么工作？

> "Softmax需要全局信息（全局max和所有元素的指数和），但分块时每块只看到部分数据。Online Softmax的思路是：每处理一个新块，更新全局max；利用exp(a-b)=exp(a)/exp(b)的性质，对之前的结果乘一个缩放因子进行修正。这样不需要存储完整的N×N矩阵，但数学上和一次性算全局softmax完全等价。这是FlashAttention最精妙的部分。"

---

> **明天预告**: Day 4 将深入阅读 KsanaLLM 源码——continuous_batching.h 和 prefix_cache_manager.h
