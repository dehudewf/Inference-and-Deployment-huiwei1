# Inference-and-Deployment-huiwei

LLM 推理优化学习与实践资源库

## 📁 项目结构

```
.
├── flash-attention-opt/     # Flash Attention CUDA 算子优化实现
├── vllm/                    # vLLM 高吞吐量推理引擎源码
├── KsanaLLM/                # 腾讯 KsanaLLM 推理框架源码
└── docs/                    # 学习笔记与面试准备
```

## 🚀 Flash Attention 优化实现

手写 Flash Attention CUDA Kernel，从基础版本逐步优化到高性能版本。

### 性能对比 (NVIDIA RTX 4060)

| `[bs, nh, N, M, d]` | Baseline | Minimal | v1 | v2 | v3 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| [32, 8, 256, 256, 256] | 46.79ms | 741.07ms | 104.73ms | 127.37ms | **12.42ms** |
| [32, 8, 256, 256, 1024] | 56.87ms | 9544.46ms | 618.70ms | 542.99ms | **48.84ms** |
| [32, 8, 1024, 1024, 256] | 92.70ms | 11343.90ms | 1524.32ms | 2026.06ms | **167.24ms** |
| [32, 8, 1024, 1024, 1024] | 232.27ms | 153121ms | 10134.30ms | 8636.17ms | **707.79ms** |

### 优化点

- **v1**: 基础 tiling 实现
- **v2**: Shared memory 优化
- **v3**: Warp-level 优化 + Bank conflict 消除

详见 [flash-attention-opt/README.md](flash-attention-opt/README.md)

## 📚 推理框架源码分析

### vLLM
- PagedAttention 内存管理
- Continuous Batching 调度
- Tensor Parallelism 实现

### KsanaLLM (腾讯)
- Prefix Cache 树状结构设计
- 异步 Swap in/out 机制
- Prefill/Decode 队列调度策略

核心源码路径：
```
KsanaLLM/src/ksana_llm/batch_scheduler/strategy/continuous_batching.h
KsanaLLM/src/ksana_llm/cache_manager/prefix_cache_manager.h
```

## 📖 学习资料

| 文件 | 内容 |
|------|------|
| `简历项目_推理优化方向.md` | 项目经历 STAR 法则描述 |
| `腾讯混元推理优化面试准备指南.md` | 面试知识点梳理 |
| `AI硕士求职入门培训手册.md` | 求职准备指南 |

## 🛠️ 环境要求

- CUDA >= 11.0
- CMake >= 3.18
- Python >= 3.8

### 编译 Flash Attention

```bash
cd flash-attention-opt
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## 📌 核心知识点

### 推理优化技术栈
- **算子优化**: Flash Attention, Fused Kernel, Quantization (INT8/FP8)
- **内存管理**: PagedAttention, KV Cache 优化, Prefix Caching
- **调度策略**: Continuous Batching, Chunked Prefill
- **并行策略**: Tensor Parallelism, Pipeline Parallelism

### 常见面试问题
1. Flash Attention 为什么能减少显存？（tiling + recomputation）
2. PagedAttention 如何管理 KV Cache？（虚拟内存思想）
3. Continuous Batching vs Static Batching 的区别？
4. Prefill 和 Decode 阶段的计算特点？

## 📝 License

本仓库仅供学习交流使用。

- vLLM: Apache 2.0 License
- KsanaLLM: Apache 2.0 License
