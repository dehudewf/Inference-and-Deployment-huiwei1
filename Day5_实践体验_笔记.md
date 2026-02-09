# Day 5: vLLM 实践体验 — 操作手册 + 笔记

> 目标：跑通vLLM，建立实际操作经验
> 今日关键词：pip install、离线推理、API Server、关键参数

---

## 1. 环境准备

### 1.1 安装vLLM

```bash
# 创建虚拟环境（推荐）
conda create -n vllm python=3.10 -y
conda activate vllm

# 安装vLLM（需要CUDA环境）
pip install vllm

# 如果没有GPU，可以用CPU模式测试基本功能
# 但性能不具有参考价值
```

### 1.2 确认环境

```bash
# 检查CUDA
nvidia-smi

# 检查vLLM
python -c "import vllm; print(vllm.__version__)"
```

---

## 2. 离线推理示例

### 2.1 最简示例（小模型，CPU也能跑）

```python
# test_vllm_offline.py
from vllm import LLM, SamplingParams

# 小模型，资源要求低
llm = LLM(model="facebook/opt-125m")

# 采样参数
sampling_params = SamplingParams(
    temperature=0.8,     # 温度: 越高越随机
    top_p=0.95,          # nucleus sampling
    max_tokens=100,      # 最大生成token数
)

# 输入prompts
prompts = [
    "Hello, my name is",
    "The capital of France is",
    "Artificial intelligence is",
]

# 生成
outputs = llm.generate(prompts, sampling_params)

# 打印结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated: {generated_text!r}")
    print("-" * 50)
```

### 2.2 带GPU的推理示例（如果有A100/3090等）

```python
# test_vllm_gpu.py
from vllm import LLM, SamplingParams

# 使用更大的模型
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",  # 或其他7B模型
    tensor_parallel_size=1,              # GPU数量
    gpu_memory_utilization=0.9,          # GPU显存利用率
    max_model_len=2048,                  # 最大序列长度
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256,
    repetition_penalty=1.1,
)

prompts = [
    "请解释什么是PagedAttention？",
    "大模型推理优化的主要方向有哪些？",
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
    print("=" * 60)
```

### 2.3 运行命令

```bash
# 运行
python test_vllm_offline.py

# 记录下面这些信息（面试素材）:
# 1. 启动时的模型加载信息
# 2. 推理速度（tokens/s）
# 3. 显存占用（nvidia-smi）
```

---

## 3. API Server 示例

### 3.1 启动服务

```bash
# 启动OpenAI兼容的API Server
python -m vllm.entrypoints.openai.api_server \
    --model facebook/opt-125m \
    --host 0.0.0.0 \
    --port 8000

# 如果有GPU和大模型:
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 2048 \
    --host 0.0.0.0 \
    --port 8000
```

### 3.2 发送请求

```bash
# Completion API
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "facebook/opt-125m",
        "prompt": "San Francisco is a",
        "max_tokens": 50,
        "temperature": 0.7
    }'

# Chat API
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "facebook/opt-125m",
        "messages": [
            {"role": "user", "content": "What is PagedAttention?"}
        ],
        "max_tokens": 100
    }'
```

### 3.3 Python客户端

```python
# test_vllm_api.py
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # vLLM不需要真实API key
)

response = client.chat.completions.create(
    model="facebook/opt-125m",
    messages=[
        {"role": "user", "content": "Hello!"}
    ],
    max_tokens=50,
)

print(response.choices[0].message.content)
```

---

## 4. 关键参数说明（面试可能问到）

| 参数                       | 含义                                    | 默认值  |
| -------------------------- | --------------------------------------- | ------- |
| `--model`                  | 模型路径或HuggingFace ID               | 必填    |
| `--tensor-parallel-size`   | TP并行GPU数量                           | 1       |
| `--pipeline-parallel-size` | PP并行阶段数                            | 1       |
| `--gpu-memory-utilization` | GPU显存使用比例（0-1）                  | 0.9     |
| `--max-model-len`          | 最大序列长度（影响KV Cache预分配）      | 模型默认 |
| `--block-size`             | PagedAttention的Block大小               | 16      |
| `--swap-space`             | CPU swap空间大小（GB）                  | 4       |
| `--max-num-batched-tokens` | 每步最大batch token数                   | 自动    |
| `--max-num-seqs`           | 最大并发序列数                          | 256     |
| `--enable-prefix-caching`  | 启用Prefix Cache                        | False   |
| `--dtype`                  | 数据类型（auto/half/float16/bfloat16）  | auto    |

**面试话术**：
> "我跑vLLM时用的命令是 `python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 1 --gpu-memory-utilization 0.9`。关键参数包括 tensor-parallel-size 控制TP并行度，gpu-memory-utilization 控制显存使用比例，block-size 默认16个token一个Block。"

---

## 5. 观察和记录

### 5.1 需要截图/记录的内容

```
□ nvidia-smi 显示的显存占用
□ vLLM启动时的日志（模型加载、Block分配信息）
□ 推理请求的输出结果
□ 如果有多个请求，观察batch处理的效果
```

### 5.2 启动日志关键信息

```
vLLM启动时会打印:
  INFO: Initializing an LLM engine with config:
    model='...'
    tensor_parallel_size=1
    dtype=float16
    max_seq_len=2048
    
  INFO: # GPU blocks: 7200, # CPU blocks: 512
  ↑ 这个数字反映了PagedAttention分配的Block数量
  ↑ GPU blocks × block_size = 最大可缓存的token数
  
  INFO: Model loaded in XX.XX seconds
```

### 5.3 简单benchmark（可选）

```python
# benchmark_vllm.py
import time
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")
params = SamplingParams(max_tokens=100, temperature=0.8)

# 不同batch size测试
for batch_size in [1, 4, 8, 16]:
    prompts = ["Hello, my name is"] * batch_size
    
    start = time.time()
    outputs = llm.generate(prompts, params)
    elapsed = time.time() - start
    
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_tokens / elapsed
    
    print(f"Batch={batch_size}, Time={elapsed:.2f}s, "
          f"Tokens={total_tokens}, Throughput={throughput:.1f} tok/s")
```

---

## 6. 今日自检清单

- [ ] 成功运行vLLM离线推理示例
- [ ] 记录启动命令和关键参数
- [ ] 截图保存作为面试素材（显存占用、Block分配信息）
- [ ] 理解vLLM的关键启动参数含义

---

## 7. 面试话术

### Q: 你有没有实际用过推理框架？

> "有的，我安装并运行了vLLM。用 `pip install vllm` 安装后，先跑了离线推理示例，然后启动了OpenAI兼容的API Server。启动时vLLM会打印分配了多少GPU blocks和CPU blocks——这就是PagedAttention实际分配的Block数量。我还做了简单的benchmark，测试不同batch_size下的吞吐量，观察到batch_size增大时吞吐量确实提升了，印证了Continuous Batching的效果。"

---

> **明天预告**: Day 6 将系统准备所有面试问答
