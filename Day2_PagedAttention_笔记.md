# Day 2: PagedAttention 专题 — 学习笔记

> 目标：深入理解PagedAttention，面试重中之重
> 今日关键词：Block、Block Table、逻辑块/物理块、Copy-on-Write、KV共享

---

## 1. 传统KV Cache的问题（回顾）

现有系统浪费 **60%-80%** 的显存，原因：

```
请求A: 实际需要 300 tokens，但预分配了 2048 tokens → 浪费 85%
请求B: 实际需要 150 tokens，但预分配了 2048 tokens → 浪费 93%
请求C: 实际需要 1800 tokens，预分配 2048 tokens → 浪费 12%

三种浪费:
1. 预分配浪费: 按max_seq_len预分配，实际用不完
2. 内部碎片: 分配的连续块内有空间未被使用
3. 外部碎片: 频繁分配释放导致碎片化，总空闲够但不连续
```

---

## 2. PagedAttention核心设计

### 2.1 操作系统虚拟内存类比

这是理解PagedAttention最好的方式——它**完全借鉴了OS的虚拟内存分页机制**：

| 操作系统概念     | PagedAttention对应概念 | 说明                           |
| ---------------- | ---------------------- | ------------------------------ |
| 进程 (Process)   | 推理请求 (Sequence)    | 每个请求有独立的地址空间       |
| 字节 (Byte)      | Token                  | 最小数据单位                   |
| 内存页 (Page)    | KV Block               | 固定大小的分配单位             |
| 页表 (Page Table) | Block Table            | 逻辑地址→物理地址的映射       |
| 虚拟地址         | 逻辑块号               | 请求视角的连续编号             |
| 物理地址         | 物理块号               | 实际GPU显存中的位置            |
| 按需分配         | 按需分配               | 生成新token时才分配新Block     |
| Copy-on-Write    | Copy-on-Write          | 共享Block被修改时才复制        |

### 2.2 Block的具体设计

```
一个Block = 固定数量的token的KV Cache（通常16个token/block）

Block结构:
┌────────────────────────────────────────────────────┐
│  Block #7 (物理块)                                  │
│  ┌──────┬──────┬──────┬─────┬──────┬──────────────┐ │
│  │ K/V  │ K/V  │ K/V  │ ... │ K/V  │  (空闲slot)  │ │
│  │ tok₁ │ tok₂ │ tok₃ │     │ tok₁₂│  4个空位     │ │
│  └──────┴──────┴──────┴─────┴──────┴──────────────┘ │
│  block_size = 16 tokens                             │
│  已填充: 12/16                                       │
│  浪费: 仅在最后一个Block的未填满部分                  │
└────────────────────────────────────────────────────┘
```

**关键**：
- 一个Block的大小 = `block_size × num_heads × head_dim × 2(K+V) × dtype_size`
- Block之间**不需要连续内存**，可以散布在显存任意位置
- 内存浪费**仅发生在最后一个Block的空闲slot**，不到4%

### 2.3 逻辑块 vs 物理块

```
请求A的视角（逻辑块，连续的）:
  逻辑块0 → 逻辑块1 → 逻辑块2 → 逻辑块3

实际显存分布（物理块，不连续的）:

显存: [  ][B5][  ][B2][  ][B9][B3][  ][  ][B1][  ]
           ↑       ↑       ↑   ↑           ↑
           │       │       │   │           │
请求A:  逻辑块0 逻辑块1     │  逻辑块3     │
请求B:                 逻辑块0         逻辑块1

Block Table（请求A）:
┌──────────┬──────────┐
│ 逻辑块号  │ 物理块号  │
├──────────┼──────────┤
│    0     │    5     │
│    1     │    2     │
│    2     │   (下次分配) │
│    3     │    3     │
└──────────┴──────────┘
```

### 2.4 生成过程中的Block分配

```
Step 1: 用户输入 "Hello world" (2 tokens)
  → 分配物理块 #5 给逻辑块 0
  → 存储 token "Hello" 和 "world" 的 K,V
  → Block #5: [Hello, world, _, _, _, _, _, _, _, _, _, _, _, _, _, _]

Step 2-15: 生成 14 个新token
  → 逐个填入 Block #5 的空闲slot
  → Block #5: [Hello, world, I, am, a, language, model, ...]  (16/16 满)

Step 16: 生成第15个token，Block #5满了
  → 分配新物理块 #2 给逻辑块 1
  → 新token存入 Block #2
  → Block #2: [new_token, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _]

... 按需继续分配
```

---

## 3. KV Cache共享机制

### 3.1 Parallel Sampling（并行采样）

同一个Prompt生成多个不同回复时，Prompt部分的KV Cache**完全相同**，可以共享：

```
Prompt: "写一首关于春天的诗"

回复1: "春风拂面暖..."     ← 共享Prompt的KV Cache
回复2: "万物复苏时..."     ← 共享Prompt的KV Cache
回复3: "桃花朵朵开..."     ← 共享Prompt的KV Cache

Block Table:
            逻辑块0(Prompt)  逻辑块1(各自)  逻辑块2(各自)
回复1:       物理块 #5        物理块 #8       物理块 #12
回复2:       物理块 #5  ←共享  物理块 #3       物理块 #7
回复3:       物理块 #5  ←共享  物理块 #11      物理块 #4

物理块#5 的引用计数 = 3
→ 省去了3份Prompt KV Cache的重复存储！
→ 内存节省高达 55%
```

### 3.2 Beam Search场景

Beam Search中多个beam共享大量前缀：

```
初始Prompt → beam1: A B C
           → beam2: A B D
           → beam3: A B C E

beam1和beam3共享 "A B C" 的KV Cache
所有beam共享 "A B" 的KV Cache
```

### 3.3 Copy-on-Write (CoW)

当共享的物理块需要被其中一个序列修改时：

```
Step 1: beam1 和 beam2 共享物理块 #5
  Block #5 引用计数 = 2

Step 2: beam1 需要在 Block #5 中写入新token
  → 检查引用计数：ref_count > 1，需要CoW
  → 分配新物理块 #9
  → 复制 Block #5 的内容到 Block #9
  → beam1 的Block Table更新：指向 Block #9
  → Block #5 引用计数 = 1（只有beam2引用）
  → Block #9 引用计数 = 1（beam1独占）
  → beam1在Block #9 中写入新token

如果 ref_count == 1:
  → 直接写入，无需复制（和OS的CoW完全一致）
```

---

## 4. vLLM源码结构（面试可提及）

```
vllm/
├── v1/
│   ├── core/
│   │   ├── block_pool.py          ← Block分配器（分配、释放、缓存Block）
│   │   ├── kv_cache_utils.py      ← BlockHash、KVCacheBlock、空闲队列
│   │   ├── kv_cache_manager.py    ← 高层KV Cache管理器
│   │   └── single_type_kv_cache_manager.py
│   ├── worker/
│   │   ├── block_table.py         ← Block Table实现（逻辑块→物理块映射）
│   │   └── gpu/block_table.py     ← GPU专用Block Table
│   └── attention/
│       └── ops/paged_attn.py      ← PagedAttention Python接口
├── csrc/
│   ├── attention/
│   │   ├── paged_attention_v1.cu  ← PagedAttention V1 CUDA内核
│   │   └── paged_attention_v2.cu  ← PagedAttention V2 CUDA内核
│   └── cache_kernels.cu           ← swap_blocks, copy_blocks CUDA操作
```

**面试话术**：
> "我阅读了vLLM的源码，核心的Block管理在 `block_pool.py`，它实现了Block的分配、释放和基于hash的Prefix Cache。Block Table在 `block_table.py` 中管理逻辑块到物理块的映射，底层的PagedAttention计算是用CUDA实现的。"

---

## 5. 为什么浪费从60-80%降到不到4%？

```
传统方式:
  每个请求预分配 max_seq_len 个token的连续空间
  浪费 = (max_seq_len - actual_len) / max_seq_len
  → 如果 max=2048, actual平均=500, 浪费=75%

PagedAttention:
  按需分配Block，只在最后一个Block有空闲slot
  浪费 = (block_size - last_block_used) / total_tokens
  → 如果 block_size=16, 平均浪费 8 tokens
  → 对于500 token的序列: 8/500 = 1.6%
  → 全局平均: < 4%

额外收益:
  1. 消除外部碎片 — 物理Block不需要连续
  2. 更多请求能同时运行 — 省下的显存用于增大batch
  3. GPU利用率提升 → 吞吐量提升 14-24x
```

---

## 6. 今日自检清单

- [x] 能画出PagedAttention的Block映射图（逻辑块→Block Table→物理块）
- [x] 能解释为什么能减少内存碎片（非连续存储，按需分配，仅最后Block有浪费）
- [x] 能说出性能提升数据（vs HuggingFace 14-24x，vs TGI 2.2-3.5x）
- [x] 能解释Copy-on-Write机制（引用计数，ref_count>1时复制再写）
- [x] 能说出Parallel Sampling的内存节省（高达55%）

---

## 7. 面试标准答案

### Q: PagedAttention原理？

> "PagedAttention借鉴了操作系统虚拟内存的分页技术。传统方式需要为每个请求预分配max_seq_len的连续内存，导致60-80%浪费。PagedAttention把KV Cache分成固定大小的Block（如16个token一块），非连续存储，按需分配，通过Block Table维护逻辑块到物理块的映射。浪费只发生在最后一个Block的空闲slot，不到4%。额外支持KV Cache共享——Parallel Sampling和Beam Search时多个序列共享Prompt部分的物理Block，用Copy-on-Write机制保证安全。性能上比HuggingFace提升14-24倍吞吐。"

### Q: Copy-on-Write怎么工作？

> "当多个序列共享同一个物理Block时，每个Block有引用计数。如果某个序列需要修改共享的Block，先检查引用计数：如果大于1，就分配新物理块、复制内容、修改映射，然后在新块上写入；如果等于1，直接写入。和操作系统的CoW机制完全一致。"

---

> **明天预告**: Day 3 将学习 Continuous Batching调度策略 + FlashAttention底层优化
