# Day 1: LLM推理基础概念 — 学习笔记

> 目标：理解LLM推理的本质和KV Cache
> 今日关键词：自回归生成、Prefill、Decode、KV Cache、内存浪费

---

## 1. Transformer自回归推理过程

### 1.1 为什么每次只生成一个token？

这是由Transformer的**自回归（Autoregressive）**性质决定的：

- 模型在生成第 n 个token时，**必须依赖前 n-1 个token作为输入上下文**
- 每一个新生成的token都会被加入到输入序列中，作为下一步计算的依据
- 因此**无法并行生成整个句子**，只能逐个token生成

```
输入: "今天天气"
第1步: "今天天气" → 模型预测 → "很"
第2步: "今天天气很" → 模型预测 → "好"
第3步: "今天天气很好" → 模型预测 → <EOS>
```

### 1.2 Prefill阶段 vs Decode阶段

这是推理框架中被**独立调度**的两个阶段：

| 维度         | Prefill（预填充）             | Decode（解码）                        |
| ------------ | ----------------------------- | ------------------------------------- |
| **任务**     | 处理用户输入的整个Prompt      | 逐个生成后续token                     |
| **输入**     | 整个Prompt（可能几百个token） | 单个新token                           |
| **计算类型** | 矩阵-矩阵乘法 (GEMM)        | 矩阵-向量乘法 (GEMV)                 |
| **瓶颈**     | **计算密集型**（GPU算力）     | **访存密集型**（显存带宽）            |
| **并行度**   | 高（所有token并行计算）       | 低（每步只处理1个token）              |
| **输出**     | 初始KV Cache + 第一个token   | 新token + 更新KV Cache               |

### 1.3 为什么Decode是访存密集型？

核心矛盾：**加载数据的时间远超计算本身的时间**

```
Prefill阶段:
  - 执行大规模 GEMM（矩阵×矩阵）
  - 权重复用率高：每个权重与Prompt中多个token参与计算
  - GPU计算单元满载 → 瓶颈在算力 (TFLOPS)

Decode阶段:
  - 执行 GEMV（矩阵×向量）
  - 为仅1个新token，必须读取：
    · 全部模型参数（几十GB）
    · 该请求所有历史KV Cache
  - 计算单元大量空闲，等待数据搬运 → 瓶颈在显存带宽 (HBM Bandwidth)
```

**面试话术**：
> "Decode阶段是访存密集型，因为每次只计算一个token的注意力，但需要从HBM读取所有历史KV Cache。计算量小而访存量大，计算强度低，GPU算力利用不充分。"

---

## 2. KV Cache加速原理

### 2.1 为什么K、V可以缓存？

在Self-Attention中：
- **Q (Query)**：代表当前token的"查询意图"，每步都是新token，必须实时计算
- **K (Key)**：代表历史token的"被查询信息"，历史token不变，K也不变
- **V (Value)**：代表历史token的"内容信息"，同理不变

```
Attention(Q, K, V) = softmax(Q × K^T / √d) × V

生成过程:
第1步: 计算 K₁, V₁                       → 缓存 [K₁], [V₁]
第2步: 只算 K₂, V₂, 复用缓存的 K₁, V₁    → 缓存 [K₁,K₂], [V₁,V₂]
第3步: 只算 K₃, V₃, 复用缓存的 K₁₂, V₁₂  → 缓存 [K₁,K₂,K₃], [V₁,V₂,V₃]
```

### 2.2 计算量对比

| 场景         | 每生成1个token的计算复杂度 | 说明                             |
| ------------ | -------------------------- | -------------------------------- |
| **无KV Cache** | O(N²)                     | 重新计算所有token的K、V和注意力  |
| **有KV Cache** | O(N)                      | 只算新token的K、V，与历史做注意力 |

### 2.3 KV Cache显存占用公式

```
Memory = 2 × layers × heads × head_dim × seq_len × precision_bytes × batch_size

其中:
  2          → K和V两个矩阵
  layers     → 模型层数（如LLaMA-13B有40层）
  heads      → 注意力头数
  head_dim   → 每个头的维度
  seq_len    → 序列长度
  precision  → 数据精度（FP16=2字节, INT8=1字节）
  batch_size → 并发请求数

例: LLaMA-13B, FP16, 单个序列2048 tokens:
  = 2 × 40 × 40 × 128 × 2048 × 2 bytes
  ≈ 1.7 GB  ← 单个序列就占1.7GB！
```

**面试关键数字**：LLaMA-13B 单个序列的KV Cache占 **~1.7GB**

---

## 3. 传统KV Cache的内存浪费问题

### 3.1 三种浪费

这是vLLM论文的核心洞察：**现有系统浪费60%-80%的显存！**

```
┌──────────────────────────────────────────────────────────────┐
│                    内存浪费的三种形式                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. 预分配浪费 (Reserved)                                     │
│     ├── 必须按最大长度(max_seq_len)预分配连续内存              │
│     ├── 实际生成可能远短于最大长度                             │
│     └── 预留空间大量闲置                                      │
│                                                              │
│  2. 内部碎片 (Internal Fragmentation)                         │
│     ├── 为每个请求分配固定大小的连续内存块                     │
│     ├── 块内未被实际使用的空间无法被其他请求利用               │
│     └── 类似：分配了1024 tokens空间，实际只用了600             │
│                                                              │
│  3. 外部碎片 (External Fragmentation)                         │
│     ├── 频繁的申请和释放导致显存中出现细碎空隙                 │
│     ├── 空隙太小无法分配给新请求                               │
│     └── 总空闲足够，但不连续，无法使用                         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 vLLM的解决方案预告（明天深入学）

**PagedAttention** = 借鉴操作系统虚拟内存的分页技术

| 概念       | 操作系统         | PagedAttention       |
| ---------- | ---------------- | -------------------- |
| 页/块      | 内存页 (Page)    | KV Block             |
| 最小单位   | 字节 (Byte)      | Token                |
| 地址映射   | 页表 (Page Table) | Block Table          |
| 分配方式   | 按需分配         | 按需分配，不连续存储 |

**关键性能数据**：
- 内存浪费：从60-80% → **不到4%**
- vs HuggingFace：**14-24x** 吞吐提升
- vs TGI：**2.2-3.5x** 吞吐提升

---

## 4. vLLM博客关键要点总结

### 4.1 vLLM核心信息

- 开发者：UC Berkeley
- 核心技术：**PagedAttention**
- 已部署：Chatbot Arena（月处理百万请求）
- 部署效果：GPU数量减少50%，同时支持更高流量

### 4.2 PagedAttention核心机制（预习）

1. **分块存储**：KV Cache被分成固定大小的Block（如16 tokens/block）
2. **非连续存储**：Block不需要在内存中连续，按需动态分配
3. **Block Table映射**：逻辑块→物理块的映射表（类似操作系统页表）
4. **内存共享**：
   - Parallel Sampling（并行采样）：多个输出共享Prompt的KV Cache
   - Beam Search：多个候选共享部分KV Cache
   - **Copy-on-Write**：只在需要修改时复制，减少55%内存开销

### 4.3 vLLM使用方式

```python
# 离线推理
from vllm import LLM
llm = LLM(model="lmsys/vicuna-7b-v1.3")
outputs = llm.generate(["Hello, my name is"])

# 在线服务
# python -m vllm.entrypoints.openai.api_server --model lmsys/vicuna-7b-v1.3
```

---

## 5. 今日自检清单

- [x] 能画出KV Cache工作原理图（见2.1节的生成过程图示）
- [x] 能解释Prefill和Decode的区别（见1.2节的对比表格）
- [x] 能说出传统KV Cache的内存浪费问题（60-80%浪费，三种碎片）
- [x] 能说出vLLM的性能数据（14-24x vs HF, 2.2-3.5x vs TGI）

---

## 6. 面试标准答案速记

### Q: KV Cache为什么能加速推理？

> "自回归模型生成时，每次只生成一个token。关键观察是：之前token的K、V值不会改变，只有Q在变。没有KV Cache时每生成一个token都要重新计算所有token的K、V，计算量O(N²)；有了KV Cache只计算新token的K、V然后和缓存的拼接，计算量降到O(N)。"

### Q: 为什么LLM推理是访存密集型？

> "Decode阶段是访存密集型。每次只计算一个token，但要读取全部模型参数和所有历史KV Cache。计算量小（batch_size × hidden_dim），访存量大（读取全部KV Cache），计算强度低，GPU大部分时间在等数据搬运。优化方向：增大batch size提高计算强度、FlashAttention减少内存访问、KV Cache量化减少访存量。"

### Q: 传统KV Cache有什么问题？

> "现有系统浪费60-80%的显存，三个原因：一是预分配浪费——必须按最大长度预分配连续内存；二是内部碎片——分配的固定块内未使用空间浪费；三是外部碎片——频繁分配释放导致碎片化。vLLM的PagedAttention借鉴OS虚拟内存分页解决了这个问题，内存浪费降到不到4%。"

---

> **明天预告**: Day 2 将深入PagedAttention——Block映射、KV共享、Copy-on-Write机制
