# Day 7: 模拟面试 + 最终检查

> 目标：查漏补缺，调整状态
> 今天的任务：模拟面试、回顾笔记、最终Checklist

---

## 1. 自我模拟面试脚本

### 使用方法

1. 对着镜子或录像，按下面顺序回答每个问题
2. 每个回答计时，控制在1-2分钟内
3. 回答完自己打分：流畅/基本/卡顿/不会
4. 卡顿和不会的回去看对应笔记

### 模拟面试题序（按真实面试顺序）

```
第一阶段: 自我介绍 (2分钟)
────────────────────────
Q: 请做一下自我介绍

  你的回答应包含:
  □ 基本信息（学校、方向）
  □ 为什么对推理优化感兴趣
  □ 做了什么准备（vLLM实践 + KsanaLLM源码）
  □ 对腾讯混元的了解和期待

  计时: _____ 分钟
  评价: □流畅  □基本  □卡顿  □需要重练

第二阶段: 技术深挖 (15-20分钟)
────────────────────────
Q1: KV Cache为什么能加速推理？
  计时: _____   评价: □流畅 □基本 □卡顿

Q2: PagedAttention原理？
  计时: _____   评价: □流畅 □基本 □卡顿

Q3: Continuous Batching是什么？三个队列怎么协作？
  计时: _____   评价: □流畅 □基本 □卡顿

Q4: FlashAttention为什么快？Online Softmax怎么工作？
  计时: _____   评价: □流畅 □基本 □卡顿

Q5: KsanaLLM有什么技术亮点？
  计时: _____   评价: □流畅 □基本 □卡顿

Q6: TP和PP什么时候用哪个？
  计时: _____   评价: □流畅 □基本 □卡顿

Q7: 大模型推理是计算密集还是访存密集？
  计时: _____   评价: □流畅 □基本 □卡顿

第三阶段: 项目讲解 (3-5分钟)
────────────────────────
Q: 讲一下你做过的最相关的项目

  STAR法则检查:
  □ Situation: 说明了背景和动机
  □ Task: 明确了目标
  □ Action: 具体说了做了什么（源码阅读+vLLM实践）
  □ Result: 说了学到了什么

  计时: _____ 分钟
  评价: □流畅  □基本  □卡顿

第四阶段: 追问和软性问题 (5分钟)
────────────────────────
Q: 你有什么问题想问我们？
  □ 准备好了2-3个有深度的问题

Q: 为什么选这个方向？
  计时: _____   评价: □流畅 □基本 □卡顿
```

---

## 2. 知识点快速复习

### 2.1 一句话复习（快速过一遍）

| 知识点                | 一句话总结                                                |
| --------------------- | --------------------------------------------------------- |
| KV Cache              | 缓存不变的K、V，每步只算新token的K、V，O(N²)→O(N)         |
| PagedAttention        | 类比OS分页，Block非连续存储，按需分配，浪费<4%            |
| Continuous Batching   | Token级调度，三队列(W/R/S)，完成即走新来即入              |
| FlashAttention        | Tiling+SRAM计算，不存N×N矩阵，Online Softmax             |
| KsanaLLM调度          | CB+PD分离+Split-fuse，ProcessWaitingQueue先查PrefixCache  |
| KsanaLLM PrefixCache  | 树状PrefixCachedBlock，token hash O(1)查找，异步swap+去重 |
| TP vs PP              | TP：机内NVLink高带宽；PP：跨机，需micro-batch             |
| Decode访存密集        | 每步只算1 token但读全部KV Cache，计算强度低               |
| Speculative Decoding  | 小模型draft多token，大模型并行验证                        |
| CoT挑战               | KV Cache爆炸，Decode占比大，需要Prefix Cache+量化         |

### 2.2 关键数字快速记忆

```
14-24x    → PagedAttention vs HuggingFace 吞吐提升
2.2-3.5x  → PagedAttention vs TGI 吞吐提升
19 TB/s   → A100 SRAM 带宽
2 TB/s    → A100 HBM 带宽
60-80%    → 传统KV Cache 内存浪费
<4%       → PagedAttention 内存浪费
2-4x      → FlashAttention 加速
5-20x     → FlashAttention 显存减少
98.3%     → KsanaLLM 跨机通信降低
20%+      → KsanaLLM 推理单价降低
~1.7GB    → LLaMA-13B 单序列KV Cache
16        → PagedAttention 默认 block size
```

---

## 3. 最终Checklist

### 3.1 必须能画的图

```
□ KV Cache工作原理图
  第1步: 计算K₁,V₁ → 缓存
  第2步: 复用K₁,V₁，只算K₂,V₂ → 缓存
  第3步: 复用K₁₂,V₁₂，只算K₃,V₃ → 缓存

□ PagedAttention Block映射图
  逻辑块0→物理块5，逻辑块1→物理块2（不连续!）
  Block Table: [逻辑块号 → 物理块号]

□ Continuous Batching调度流程图
  Waiting → Running → Complete
                ↕
            Swapped

□ FlashAttention Tiling示意图
  Q分块 × K分块 → 小S → 在SRAM计算softmax → × V分块 → 累加到O
  中间结果不写回HBM!
```

### 3.2 必须能说的话术

```
□ "我研究过腾讯的KsanaLLM源码..."
□ "continuous_batching.h中的ProcessWaitingQueue..."
□ "prefix_cache_manager用树状结构PrefixCachedBlock实现..."
□ "token序列作key，unordered_map O(1)查找子节点..."
□ "Swap-in时做去重合并..."
□ "PP multi-batch把跨机通讯降低了98.3%..."
□ "我理解原理但还没有深入修改代码的经验..."
```

### 3.3 技术Checklist

```
□ 能回答全部P0问题（KV Cache、PagedAttention、CB、FA、KsanaLLM）
□ 能回答P1问题（TP/PP、CoT、访存密集）
□ 能画4张关键图
□ 记住10+个关键数字
□ 项目STAR讲解控制在3分钟内
□ 准备好诚实应对不会问题的话术
□ 准备好2-3个反问问题
```

### 3.4 面试当天Checklist

```
面试前一天:
□ 早睡（保证7-8小时睡眠）
□ 准备好简历（电子版+打印版）
□ 检查面试设备（如果是视频面试）
□ 最后过一遍关键数字和话术

面试当天:
□ 提前15分钟到达/上线
□ 准备纸笔（随时画图解释）
□ 手机静音
□ 深呼吸，调整心态
□ 记住：诚实 > 完美，展示学习能力 > 假装什么都会
```

---

## 4. 心态调整

```
你的优势:
  ✓ 系统学习了推理优化的核心技术
  ✓ 阅读了腾讯自研项目的源码（99%的候选人没做过）
  ✓ 能说出具体的文件名、函数名、数据结构
  ✓ 跑通了vLLM实际示例
  ✓ 对技术方向有真诚的兴趣

你需要坦诚的:
  ✓ 没有实际修改框架代码的经验
  ✓ CUDA/C++基础还在学习中
  ✓ 这些都是希望通过实习来积累的

面试官看重的:
  ✓ 学习能力和速度（1周能理解到这个程度很好）
  ✓ 技术好奇心（主动研究开源项目源码）
  ✓ 诚实和自知（知道什么会什么不会）
  ✓ 成长潜力（有基础，缺的只是工程经验）
```

---

## 5. 笔记文件索引

| 文件                                           | 内容                          |
| ---------------------------------------------- | ----------------------------- |
| `Day1_基础概念_笔记.md`                         | Prefill/Decode、KV Cache原理 |
| `Day2_PagedAttention_笔记.md`                   | Block映射、CoW、KV共享       |
| `Day3_ContinuousBatching_FlashAttention_笔记.md`| CB三队列、FA Tiling/Softmax  |
| `Day4_KsanaLLM源码_笔记.md`                     | 调度源码、PrefixCache源码    |
| `Day5_实践体验_笔记.md`                         | vLLM安装运行命令、参数说明   |
| `Day6_面试问答_全集.md`                          | 所有面试QA标准答案           |
| `Day7_模拟面试_最终检查.md`                      | 模拟面试脚本、Checklist      |

---

**加油！你已经准备得很充分了。面试时记住：展示你的学习能力和技术热情。**
