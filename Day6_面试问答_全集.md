# Day 6: 面试问答全集 — 所有可能的面试问题

> 目标：准备所有可能的面试问题，流畅回答
> 按优先级分类：P0必问、P1大概率问、P2可能追问、软性问题

---

## P0级：必问，必须秒答（占面试60%+权重）

### Q1: KV Cache为什么能加速推理？

> "自回归模型生成时，每次只生成一个token。关键观察是：之前token的K、V值不会改变，只有Q在变。没有KV Cache时每生成一个token都要重新计算所有token的K、V，计算量O(N²)；有了KV Cache只计算新token的K、V然后和缓存的拼接，计算量降到O(N)。这就是加速的本质。"

**追问：KV Cache有多大？**
> "以LLaMA-13B为例，单个序列2048 tokens的KV Cache大约占1.7GB。计算公式：2(K+V) × 层数 × 头数 × 头维度 × 序列长度 × 数据精度。"

---

### Q2: PagedAttention原理？

> "PagedAttention借鉴了操作系统虚拟内存的分页技术。
>
> **传统问题**：KV Cache需要连续内存预分配，不同请求长度不同，导致60-80%内存浪费。
>
> **解决方案**：将KV Cache分成固定大小的Block（如16个token一块），非连续存储，按需分配。通过Block Table维护逻辑块到物理块的映射——就像OS的页表。
>
> **额外优势**：支持KV Cache共享（Beam Search、并行采样），使用Copy-on-Write机制——引用计数大于1时修改才复制。
>
> **性能**：vs HuggingFace 14-24x吞吐提升，vs TGI 2.2-3.5x提升。内存浪费降到不到4%。"

**追问：为什么浪费只有4%？**
> "因为浪费只发生在每个序列最后一个Block的未填满部分。如果block_size=16，平均浪费8个token，对500 token的序列来说只有1.6%。"

---

### Q3: Continuous Batching是什么？

> "传统Static Batching要求batch内所有请求同时完成，短请求等长请求，GPU空转严重。
>
> Continuous Batching是**token级调度**：每生成一个token就可以调整batch组成。完成的请求立即移出，新请求可以插入。
>
> 维护三个队列：**Waiting**（等待分配显存的新请求）、**Running**（GPU上正在推理的请求）、**Swapped**（显存不足时KV Cache被异步换出到CPU内存的请求）。
>
> 好处：GPU利用率大幅提升，吞吐量提高2-3倍。"

**追问：Swap机制怎么工作？**
> "当Running队列的KV Cache超出显存时，调度器会选择低优先级的请求，将其KV Cache异步从GPU显存复制到CPU内存（Swap-out）。KsanaLLM的实现是从尾到头换出，保留前缀Block（因为前缀更可能被复用）。显存空闲后再异步换回（Swap-in），还会和树中已有Block做去重合并。"

---

### Q4: FlashAttention为什么快？

> **一句话版本**：
> "传统Attention需要把N×N的中间矩阵存到HBM，FlashAttention通过Tiling分块在SRAM中计算，SRAM带宽19TB/s是HBM 2TB/s的近10倍，大幅减少IO。"

> **详细版本**：
> "三个创新：一是**Tiling分块**——Q、K、V分成小块在SRAM中计算，不把中间结果写回HBM；二是**Online Softmax**——分块时维护全局max和sum，用指数缩放修正之前结果，数学完全等价；三是**内存优化**——显存从O(N²)降到O(N)。最终2-4倍加速，显存减少5-20倍。"

**追问：Online Softmax怎么工作？**
> "Softmax需要全局max和指数和，但分块时只看到部分数据。解法是每处理一块就更新全局max，利用exp(a-b)=exp(a)/exp(b)的性质对之前结果乘缩放因子修正。不需要存N×N矩阵但数学等价。"

---

### Q5: KsanaLLM有什么技术亮点？

> "KsanaLLM是腾讯PCG开源的推理引擎，我研究过它的源码。四个亮点：
>
> 1. **多硬件支持**——同时支持NVIDIA GPU和华为昇腾NPU，适配国产化需求。
> 2. **高效调度**——continuous_batching.h中的ProcessWaitingQueue函数在调度时先查Prefix Cache命中，还支持Split-fuse把大Prefill分块混在Decode中执行。
> 3. **Prefix Cache**——prefix_cache_manager.h用PrefixCachedBlock树状结构存储，token序列作key用unordered_map O(1)查找子节点。Swap-in时做去重合并。
> 4. **多机调度**——PP multi-batch把跨机通讯降低98.3%。
>
> 性能比vLLM/TensorRT-LLM推理单价低20%+。"

---

## P1级：大概率会问（占面试25%权重）

### Q6: TP和PP什么时候用哪个？

> "**Tensor Parallel (TP)**适合机内高带宽互联（NVLink，单向带宽50-450GB/s）——把每一层的计算拆分到多GPU，降低延迟。切分方式：列切分MLP，行切分Attention。
>
> **Pipeline Parallel (PP)**适合跨机低带宽场景——按layer切分模型，需要micro-batch减少pipeline bubble。
>
> **混合策略**：机内用TP（利用NVLink），机间用PP（减少跨机通信）。KsanaLLM的PP multi-batch把跨机通讯量降低了98.3%。"

---

### Q7: 长思维链(CoT)场景有什么挑战？

> "CoT意味着输出序列特别长（几千到几万token），三个挑战：
>
> 1. **KV Cache显存爆炸**——序列越长KV Cache越大，可能需要swap到CPU
> 2. **Decode阶段占比大**——生成token多，decode成为主要耗时
> 3. **注意力计算量增长**——序列长时每步注意力计算线性增长
>
> 优化方案：
> - PagedAttention：按需分配，减少碎片
> - Prefix Cache：多轮对话复用前缀KV（KsanaLLM树状结构实现）
> - FlashAttention：显存O(N²)→O(N)
> - Speculative Decoding：小模型draft加速
> - KV Cache量化：FP16→INT8减半显存"

---

### Q8: 大模型推理是计算密集还是访存密集？

> "**分阶段看**：
> - Prefill阶段是**计算密集型**——大量GEMM矩阵乘法，权重复用率高，瓶颈在算力TFLOPS
> - Decode阶段是**访存密集型**——每次只算1个token的GEMV，但要读取所有KV Cache和模型参数，计算强度低，瓶颈在HBM带宽
>
> 优化方向：增大batch_size提高计算强度、FlashAttention减少HBM访问、KV Cache量化减少访存量。"

---

### Q9: Speculative Decoding是什么？

> "用一个小模型（Draft Model）快速生成多个token候选，然后大模型一次性并行验证。如果小模型预测对了，大模型一步就确认多个token，绕过自回归的串行限制。本质是把Decode的访存密集型单步计算，转化为类似Prefill的计算密集型验证。只要小模型准确率够高，就能获得加速。"

---

## P2级：可能追问（占面试10%权重）

### Q10: 多模态推理有什么特点？

> "多模态推理的主要区别在Prefill阶段——图像输入会产生大量token（如一张图可能编码为上千个token），导致Prefill压力远超纯文本。调度优化需要Chunked Prefill——把超长的图像Prefill分块，避免大任务长时间阻塞Decode队列。图像encoder和文本decoder的调度可以分开或统一，这取决于框架设计。"

---

### Q11: KV Cache量化有什么影响？

> "**正面**：FP16→INT8可以直接将KV Cache显存减半，支持更大batch或更长序列。
>
> **负面**：不当量化会导致精度下降，特别是长序列场景——KV Cache的微小误差会随推理步数积累，影响输出质量。需要在吞吐和精度之间权衡。"

---

### Q12: vLLM的Block Table具体是什么结构？

> "Block Table是一个二维数组，维度是 [max_num_reqs, max_num_blocks_per_req]，每个元素存储物理块ID。类似OS页表，通过逻辑块号索引到物理块号。在GPU上用Tensor存储，支持高效的Kernel访问。"

---

## 软性问题

### Q: 自我介绍（1分钟）

> "您好，我是XXX，目前本科四年级，即将去香港城市大学读硕士。我对大模型推理优化方向很感兴趣。最近系统学习了vLLM和KsanaLLM两个框架，特别深入阅读了KsanaLLM的源码，对Continuous Batching调度和Prefix Cache设计有了理解。我了解到腾讯混元在长思维链和多模态场景有大量优化需求，KsanaLLM的多机调度优化也给我留下了深刻印象。我希望能加入团队，将我的算法背景和对系统优化的兴趣结合起来。"

---

### Q: 为什么选这个方向？

> "三个原因：
> 1. **商业价值**——训练是一次性的，推理是持续成本。优化推理效率直接影响产品体验和商业可行性。
> 2. **算法+系统结合**——我有算法背景，同时对底层系统很感兴趣。FlashAttention就是利用GPU硬件特性重新设计算法的典范。
> 3. **腾讯技术**——我研究过KsanaLLM，对其多硬件支持和异步调度设计印象深刻，特别是针对长思维链的优化。"

---

### Q: 你做的最有挑战的事是什么？

> "最有挑战的是理解Online Softmax算法——如何在分块计算时正确维护全局的max和sum。我花了不少时间画图推导，最终理解了它通过指数的scale来实现数值稳定的分块计算。这让我意识到很多系统优化的本质是重新设计算法来适配硬件特性。"

---

### Q: 你有什么问题想问我们？

> 1. "KsanaLLM在长思维链场景下，KV Cache的swap策略是怎么设计的？有没有考虑预测性的预加载？"
> 2. "团队目前在多模态推理调度上有什么挑战？图像encoder和文本decoder的调度是分开还是统一的？"
> 3. "对于实习生，您建议我重点学习哪些技术栈或工具？"

---

### 诚实应对不会的问题

**被问到不会的深入细节**：
> "这部分我主要是阅读源码理解原理，没有实际修改过代码。但我理解它的设计思路是... [说你知道的部分]"

**被问到没有实践经验**：
> "我时间有限，主要做了两件事：一是跑通vLLM示例，理解serving流程；二是阅读KsanaLLM源码，特别是continuous_batching和prefix_cache_manager模块。我理解原理但还没有深入优化的经验，这也是我希望通过实习来积累的。"

**被问到完全不懂的概念**：
> "这个概念我还没有深入学习，但根据名字我猜测可能是... 您能简单介绍一下吗？我很想了解。"

---

## 必须记住的数字速查

| 数字                              | 用途                           |
| --------------------------------- | ------------------------------ |
| PagedAttention vs HF: **14-24x**  | 吞吐提升                      |
| PagedAttention vs TGI: **2.2-3.5x** | 吞吐提升                    |
| A100 SRAM: **19 TB/s**            | FlashAttention关键            |
| A100 HBM: **2 TB/s**              | FlashAttention关键            |
| 传统KV Cache浪费: **60-80%**      | PagedAttention动机            |
| PagedAttention浪费: **<4%**       | PagedAttention效果            |
| FlashAttention加速: **2-4x**      | 速度提升                      |
| FlashAttention显存: **5-20x减少** | 显存优化                      |
| KsanaLLM跨机通信: **降低98.3%**   | 多机调度优化                  |
| KsanaLLM推理单价: **低20%+**      | 成本优势                      |
| LLaMA-13B单序列KV: **~1.7GB**     | KV Cache规模                  |
| NVLink带宽: **50-450 GB/s**       | TP适用场景                    |
| Block size: **16 tokens**         | PagedAttention默认            |
| Continuous Batching: **2-3x吞吐** | 调度收益                      |

---

## 今日自检清单

- [x] 所有P0问题能流畅回答（不看稿子）
- [x] P1问题能回答出要点
- [x] 项目讲解控制在1-3分钟
- [x] 准备好诚实应对不会问题的话术
- [x] 关键数字全部记住

---

> **明天预告**: Day 7 模拟面试 + 最终Checklist检查
