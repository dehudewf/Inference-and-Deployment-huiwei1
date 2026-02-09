# Day 4: KsanaLLM 源码阅读 — 学习笔记

> 目标：深入了解腾讯自研项目，面试主动提及
> 今日关键词：ContinuousBatchingStrategy、PrefixCachedBlock、树状缓存、异步Swap

---

## 1. KsanaLLM 项目概览

### 1.1 项目定位

- **腾讯PCG开源的高性能推理引擎**
- GitHub: https://github.com/Tencent/KsanaLLM
- 集成了 vLLM、TensorRT-LLM、FastTransformer、SGLang、LightLLM 的优化 CUDA kernel

### 1.2 核心技术亮点（面试必提3个）

```
1. 多硬件支持:
   - NVIDIA GPU: A10, A100, L20, L40, H20
   - 华为昇腾 NPU: 910B2C
   → 适配国产化需求

2. 高效调度:
   - 改进的Continuous Batching实现
   - 支持Prefill-Decode解耦（PD Separation）
   - Split-fuse: 大Prefill任务分块执行，提高内存效率
   - 分布式协调: 多DP组之间使用barrier和共享计数器同步

3. Prefix Cache设计:
   - 树状结构存储 (PrefixCachedBlock)
   - Token序列hash匹配实现O(1)前缀查找
   - 支持异步swap in/out
   - Block引用计数，多请求共享

4. 多机调度优化（v0.6.0）:
   - PP multi-batch
   - 跨机通讯量降低 98.3%
   - 支持TCP机间通讯

性能: 比vLLM/TensorRT-LLM推理单价低 20%+
```

---

## 2. continuous_batching.h 源码分析

### 2.1 类结构

```cpp
// 文件: src/ksana_llm/batch_scheduler/strategy/continuous_batching.h

class ContinuousBatchingStrategy : public BaseScheduleStrategy {
public:
  // 核心调度函数
  void ProcessDecodingQueue();     // 处理正在Decode的请求
  void ProcessWaitingQueue();      // 处理等待中的请求（新请求入场）
  void ProcessSwappedQueue();      // 处理被换出的请求
  void ProcessTransferQueue();     // PD分离相关的传输队列

private:
  // 请求管理
  void RecomputeRequest(...);      // 重算请求（从头开始）
  void SwapoutRequest(...);        // 将请求换出到CPU内存
  void StopRequest(...);           // 停止/完成请求

  // 资源限制配置
  size_t dp_max_step_token_num_;   // 每步最大token数（DP组）
  size_t dp_max_batch_size_;       // 最大batch大小
  size_t dp_max_decode_batch_size_; // PD分离时decode最大batch
  size_t dp_max_logits_num_;       // 最大logits数量
};
```

### 2.2 ProcessDecodingQueue() — 处理Decode队列

```
调用时机: 每个调度周期开始时
源码位置: continuous_batching.cpp:520-606

执行流程:
┌──────────────────────────────────────────────────┐
│  ProcessDecodingQueue()                           │
│                                                  │
│  Step 1: 合并待SwapIn的请求（如果有）               │
│          MergePendingSwapinRequests()              │
│                                                  │
│  Step 2: 估算所有Decode请求的工作量                 │
│          EstimatePlanningWorkloads()               │
│                                                  │
│  Step 3: 检查是否超出每步最大token数                │
│          CheckRunningQueueStepTokens()             │
│                                                  │
│  Step 4: 为每个请求分配KV Cache Block              │
│          for each request:                        │
│            req_block_num = GetRequestStepBlockNumber() │
│            AllocateRequestBlocks()                │
│            if 分配失败:                            │
│              触发recompute释放内存                  │
│                                                  │
│  Step 5: 确定投机解码的draft token数               │
│                                                  │
│  Step 6: 成功分配的请求加入 running_reqs          │
└──────────────────────────────────────────────────┘

设计要点:
- 只处理 prefill_token_num == 0 的请求（纯Decode阶段）
- Block分配失败时不直接报错，而是通过recompute释放waiting队列的内存
- 支持split-fuse优化
```

### 2.3 ProcessWaitingQueue() — 处理Waiting队列

```
调用时机: ProcessDecodingQueue之后
源码位置: continuous_batching.cpp:607-756

执行流程:
┌──────────────────────────────────────────────────┐
│  ProcessWaitingQueue()                            │
│                                                  │
│  Step 1: 多DP组之间同步计数器                      │
│          scheduler_shared_counter_ barrier        │
│                                                  │
│  Step 2: 检查超时和中止状态                        │
│          CheckTimeoutAndAbort()                   │
│                                                  │
│  Step 3: 对每个等待请求:                           │
│    a. 估算工作量                                  │
│    b. 查找Prefix Cache命中                         │
│       GetRequestPrefixBlockNumber()               │
│       → 返回 shared_block_num, unique_block_num   │
│    c. 计算需要的block数和token数                   │
│    d. Split-fuse: 大Prefill任务分块                │
│    e. 检查资源约束:                                │
│       · logits数 ≤ dp_max_logits_num_             │
│       · batch大小 < dp_max_batch_size_            │
│       · step_token_num ≤ dp_max_step_token_num_   │
│       · 空闲block数 ≥ 需求 + 阈值                 │
│    f. 满足 → 分配Block, 加入Running               │
│       不满足 → 留在Waiting                         │
│                                                  │
│  Step 4: 上报prefix cache命中统计                  │
│          prefix_cache_hit_req_num                 │
│          prefix_cache_hit_token_num               │
└──────────────────────────────────────────────────┘

设计要点:
- Prefix Cache命中检测: 先查树状缓存，找到共享前缀可以跳过重复计算
- Split-fuse: 大Prefill不是一次全做，而是分成小步混在Decode中执行
- 线程安全: 使用Lock()/Unlock()保护资源检查
- 分布式协调: 多DP组之间通过barrier和共享计数器同步
```

---

## 3. prefix_cache_manager.h 源码分析

### 3.1 核心数据结构

```cpp
// 文件: src/ksana_llm/cache_manager/prefix_cache_manager.h

// 树节点 — 每个缓存Block
struct PrefixCachedBlock {
  size_t block_id;                     // 块唯一ID
  bool is_device_location;             // 在GPU(true) 还是CPU(false)
  std::vector<int> memory_block_ids;   // 物理内存块ID（每个设备一个）

  // ========= 树结构 =========
  size_t hash_code;                    // 内容hash值
  bool is_root;                        // 是否根节点
  PrefixCachedBlock* parent;           // 父节点指针
  std::unordered_map<
    std::vector<int>,                  // key: token ID序列
    PrefixCachedBlock*,                // value: 子节点指针
    TokensHash,                        // 自定义hash函数
    TokensEqual                        // 自定义相等比较
  > children;                          // 子节点映射表

  // ========= 共享状态 =========
  bool is_shareable;                   // 是否可被多个请求共享
  std::vector<int> token_ids;          // 本Block包含的token序列

  // ========= 引用追踪 =========
  std::unordered_map<int, std::pair<int, PrefixCachedRequest*>>
    active_requests;                   // 正在运行的引用请求
  std::unordered_map<int, std::pair<int, PrefixCachedRequest*>>
    inactive_requests;                 // 等待/换出的引用请求

  std::mutex swapin_mutex;             // 保护swapin操作
};

// 请求缓存元数据
struct PrefixCachedRequest {
  int64_t req_id;                      // 请求ID
  RequestState req_state;              // kWaiting/kRunning/kSwapped/kFinished
  size_t shared_block_num;             // 共享Block数量
  std::vector<PrefixCachedBlock*> cached_blocks;  // 所有关联Block
};
```

### 3.2 树状结构图示

```
                    root (虚拟根节点)
                   /          \
                  /            \
          Block A              Block D
        tokens: [1,2,3,4]    tokens: [5,6,7,8]
        hash: 0xABC           hash: 0xDEF
           /        \              |
          /          \             |
     Block B       Block C     Block E
   tokens:[5,6,7,8] tokens:[9,10,11,12]  tokens:[1,2,3,4]
   hash: 0x123      hash: 0x456         hash: 0x789

查找过程 (GetRequestPrefixBlockNumber):
  输入tokens: [1,2,3,4,5,6,7,8,...]
  
  Step 1: 在root的children中查找 [1,2,3,4]
          → 命中 Block A! (shared_block_num += 1)
  
  Step 2: 在Block A的children中查找 [5,6,7,8]
          → 命中 Block B! (shared_block_num += 1)
  
  Step 3: 在Block B的children中查找后续token
          → 未命中, 停止匹配
  
  结果: shared_block_num=2, 这2个Block的KV Cache可以直接复用!
```

### 3.3 Hash匹配机制

```cpp
// 通过token序列查找子Block — O(1)查找
PrefixCachedBlock* FindChildCacheBlock(
    PrefixCachedBlock* block,
    const int* start,
    size_t len) {
  std::vector<int> token_ids(start, start + len);
  auto it = block->children.find(token_ids);  // unordered_map O(1)
  return (it == block->children.end()) ? nullptr : it->second;
}

// 验证token匹配
bool CheckSameTokens(const PrefixCachedBlock* block, 
                     const int* start, size_t len) {
  return std::memcmp(block->token_ids.data(), start, 
                     len * sizeof(int)) == 0;
}
```

**设计精妙之处**：用 `std::unordered_map` + 自定义 `TokensHash`，实现按 token 序列的 O(1) 子节点查找。

### 3.4 异步Swap机制

```
Swap-out（GPU → CPU）:
┌──────────────────────────────────────────────────┐
│  SwapoutRequestAsync(req_id)                     │
│                                                  │
│  1. 从尾到头遍历请求的Block（保留前缀Block）       │
│  2. 跳过被多个active请求引用的Block（不能独占换出）│
│  3. 为每个换出Block分配CPU内存                    │
│  4. 异步复制: GPU显存 → CPU内存（线程池）         │
│  5. 更新 is_device_location = false              │
│  6. 请求状态 → kSwapped                          │
└──────────────────────────────────────────────────┘

Swap-in（CPU → GPU）:
┌──────────────────────────────────────────────────┐
│  SwapinRequestAsync(req_id)                      │
│                                                  │
│  1. 找到所有 is_device_location==false 的Block    │
│  2. 检查GPU空闲Block是否足够                      │
│  3. 从 free_cached_blocks_ 队列分配Block         │
│  4. 异步复制: CPU内存 → GPU显存                   │
│  5. 存入 swapin_cached_block_buffer_ 待合并      │
└──────────────────────────────────────────────────┘

Merge（合并回树）:
┌──────────────────────────────────────────────────┐
│  MergeSwapinRequest(req_id)                      │
│                                                  │
│  1. 对每个换入的Block:                            │
│     · 已在树中(parent!=nullptr) → 更新内存ID      │
│     · 可共享(is_shareable) → hash查找是否有重复   │
│       ├── 找到 → 合并到已有Block（去重!）         │
│       └── 未找到 → 加入树，更新parent的children   │
│     · 不可共享 → 仅更新内存ID                     │
│  2. 请求状态 → kRunning                          │
└──────────────────────────────────────────────────┘

设计要点:
  - Swap-out从尾到头: 保留前缀Block（更可能被其他请求复用）
  - Swap-in有去重: 如果hash匹配到树中已有Block，直接合并（避免重复存储）
  - 异步操作: 使用线程池执行设备间数据传输，不阻塞调度
  - 引用计数: 多个active请求引用的Block不能被换出
```

---

## 4. KsanaLLM vs vLLM 对比

| 维度             | vLLM                        | KsanaLLM                       |
| ---------------- | --------------------------- | ------------------------------ |
| **PagedAttention** | 原创设计                    | 采用并优化                     |
| **调度策略**     | Continuous Batching         | CB + PD分离 + Split-fuse       |
| **Prefix Cache** | Block级hash                 | 树状结构 + token序列hash       |
| **Swap机制**     | 同步swap                   | 异步swap + 去重合并            |
| **多硬件**       | 主要NVIDIA GPU              | GPU + 华为NPU                  |
| **多机通信**     | 标准NCCL                   | PP multi-batch, 通信量降98.3%  |
| **PD分离**       | 不支持                      | 支持Prefill-Decode解耦         |

---

## 5. 今日自检清单

- [x] 能说出KsanaLLM的3个技术亮点（多硬件、异步调度+PD分离、树状Prefix Cache）
- [x] 能提到具体的源码文件/函数名（continuous_batching.h/ProcessWaitingQueue、prefix_cache_manager.h/PrefixCachedBlock）
- [x] 能说出多机调度优化（PP multi-batch, 跨机通讯降低98.3%）
- [x] 能对比KsanaLLM和vLLM的设计差异

---

## 6. 面试标准答案

### Q: KsanaLLM有什么技术亮点？

> "KsanaLLM是腾讯PCG开源的推理引擎，我研究过它的源码。主要亮点有四个：
> 
> **第一，多硬件支持**——同时支持NVIDIA GPU和华为昇腾NPU，适配国产化需求。
>
> **第二，高效调度**——在continuous_batching.h中实现了改进的Continuous Batching，ProcessWaitingQueue函数在分配资源时会先查Prefix Cache命中情况，还支持Split-fuse把大Prefill任务分块混在Decode中执行。
>
> **第三，Prefix Cache设计**——在prefix_cache_manager.h中，用PrefixCachedBlock树状结构存储缓存。每个节点用token序列作key，通过自定义hash的unordered_map实现O(1)子节点查找。Swap-in时还会做去重合并——如果hash匹配到树中已有Block就直接复用。
>
> **第四，多机调度优化**——v0.6.0实现了PP multi-batch，把跨机通讯量降低了98.3%。
>
> 性能上比vLLM和TensorRT-LLM推理单价低20%以上。"

### Q: KsanaLLM的Prefix Cache怎么设计的？

> "它用树状结构存储前缀缓存。根节点是虚拟节点，每个子节点代表一个KV Cache Block，包含固定数量token的KV数据。子节点映射用unordered_map，key是token ID序列，通过自定义hash函数实现O(1)查找。当新请求到来时，从根节点开始逐层匹配token序列，命中的Block可以直接复用其KV Cache，不需要重新计算。每个Block还维护引用计数——active_requests和inactive_requests，支持多请求安全共享。Swap操作是异步的，Swap-out从尾到头保留前缀，Swap-in时会和树中已有节点做去重合并。"

---

> **明天预告**: Day 5 将跑通vLLM实际示例，建立操作经验
