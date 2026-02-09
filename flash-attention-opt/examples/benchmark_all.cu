#include "utils.h"
#include "attention.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct BenchConfig {
    int batch_size;
    int num_head;
    int N;
    int M;
    int d;
};

float runKernel(const char* name,
    void (*launcher)(const float*, const float*, const float*, float*, float*, float*,
                     int, int, int, int, int, cudaStream_t),
    const float* d_Q, const float* d_K, const float* d_V,
    float* d_O, float* d_l, float* d_m,
    int batch_size, int num_head, int N, int M, int d,
    int warmup, int repeat)
{
    size_t O_size = (size_t)batch_size * num_head * N * d;
    size_t lm_size = (size_t)batch_size * num_head * N;

    for (int i = 0; i < warmup; ++i) {
        launcher(d_Q, d_K, d_V, d_O, d_l, d_m, batch_size, num_head, N, M, d, 0);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < repeat; ++i) {
        launcher(d_Q, d_K, d_V, d_O, d_l, d_m, batch_size, num_head, N, M, d, 0);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / repeat;
}

float runBaseline(
    const float* d_Q, const float* d_K, const float* d_V,
    float* d_QK, float* d_QK_softmax, float* d_O,
    int batch_size, int num_head, int N, int M, int d,
    int warmup, int repeat)
{
    for (int i = 0; i < warmup; ++i) {
        attention::launchAttentionBaseline(d_Q, d_K, d_V, d_QK, d_QK_softmax, d_O,
            batch_size, num_head, N, M, d, 0);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < repeat; ++i) {
        attention::launchAttentionBaseline(d_Q, d_K, d_V, d_QK, d_QK_softmax, d_O,
            batch_size, num_head, N, M, d, 0);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / repeat;
}

int main(int argc, char* argv[])
{
    printf("=============================================================\n");
    printf("  FlashAttention Kernel Benchmark - All Versions\n");
    printf("=============================================================\n\n");

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("Global Memory: %.1f GB\n\n", prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);

    std::vector<BenchConfig> configs = {
        {32, 8, 256,  256,  128},
        {32, 8, 512,  512,  128},
        {32, 8, 1024, 1024, 128},
        {16, 8, 2048, 2048, 128},
        {8,  8, 4096, 4096, 128},
    };

    constexpr int WARMUP = 3;
    constexpr int REPEAT = 5;

    printf("%-30s | %10s | %10s | %10s | %10s | %10s\n",
           "[bs,nh,N,M,d]", "Baseline", "v1", "v2", "v3", "v3/Base");
    printf("-------------------------------|------------|------------|------------|------------|------------\n");

    for (auto& cfg : configs) {
        int bs = cfg.batch_size, nh = cfg.num_head;
        int N = cfg.N, M = cfg.M, d = cfg.d;

        size_t QO_size = (size_t)bs * nh * N * d;
        size_t KV_size = (size_t)bs * nh * M * d;
        size_t lm_size = (size_t)bs * nh * N;
        size_t QK_size = (size_t)bs * nh * N * M;

        float* h_Q = new float[QO_size];
        float* h_K = new float[KV_size];
        float* h_V = new float[KV_size];

        for (size_t i = 0; i < QO_size; ++i)
            h_Q[i] = static_cast<float>(static_cast<int>(i * 41 % 2001) * 0.01f - 10.0f);
        for (size_t i = 0; i < KV_size; ++i) {
            h_K[i] = static_cast<float>((static_cast<int>(i % 211) - 105) * 0.095f);
            h_V[i] = static_cast<float>(static_cast<int>(i * 53 % 1999) * 0.01f - 10.0f);
        }

        float *d_Q, *d_K, *d_V, *d_O, *d_l, *d_m, *d_QK, *d_QK_softmax;
        cudaMalloc(&d_Q, sizeof(float) * QO_size);
        cudaMalloc(&d_K, sizeof(float) * KV_size);
        cudaMalloc(&d_V, sizeof(float) * KV_size);
        cudaMalloc(&d_O, sizeof(float) * QO_size);
        cudaMalloc(&d_l, sizeof(float) * lm_size);
        cudaMalloc(&d_m, sizeof(float) * lm_size);

        cudaMemcpy(d_Q, h_Q, sizeof(float) * QO_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_K, h_K, sizeof(float) * KV_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, h_V, sizeof(float) * KV_size, cudaMemcpyHostToDevice);

        // Baseline needs extra buffers for N*M intermediate matrices
        bool run_baseline = true;
        size_t qk_bytes = sizeof(float) * QK_size;
        if (qk_bytes > (size_t)2 * 1024 * 1024 * 1024) {
            run_baseline = false;  // skip if >2GB intermediate
        }

        float t_baseline = -1, t_v1 = -1, t_v2 = -1, t_v3 = -1;

        if (run_baseline) {
            cudaMalloc(&d_QK, sizeof(float) * QK_size);
            cudaMalloc(&d_QK_softmax, sizeof(float) * QK_size);
            t_baseline = runBaseline(d_Q, d_K, d_V, d_QK, d_QK_softmax, d_O,
                                     bs, nh, N, M, d, WARMUP, REPEAT);
            cudaFree(d_QK);
            cudaFree(d_QK_softmax);
        }

        // Reset l, m for each run
        auto resetLM = [&]() {
            std::vector<float> h_l(lm_size, 0.0f);
            std::vector<float> h_m(lm_size, -1e20f);
            cudaMemcpy(d_l, h_l.data(), sizeof(float) * lm_size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_m, h_m.data(), sizeof(float) * lm_size, cudaMemcpyHostToDevice);
            cudaMemset(d_O, 0, sizeof(float) * QO_size);
        };

        resetLM();
        t_v1 = runKernel("v1", attention::launchFlashAttentionKernel_v1,
                          d_Q, d_K, d_V, d_O, d_l, d_m, bs, nh, N, M, d, WARMUP, REPEAT);

        resetLM();
        t_v2 = runKernel("v2", attention::launchFlashAttentionKernel_v2,
                          d_Q, d_K, d_V, d_O, d_l, d_m, bs, nh, N, M, d, WARMUP, REPEAT);

        resetLM();
        t_v3 = runKernel("v3", attention::launchFlashAttentionKernel_v3,
                          d_Q, d_K, d_V, d_O, d_l, d_m, bs, nh, N, M, d, WARMUP, REPEAT);

        char config_str[64];
        snprintf(config_str, sizeof(config_str), "[%d,%d,%d,%d,%d]", bs, nh, N, M, d);

        printf("%-30s | ", config_str);
        if (t_baseline > 0) printf("%8.2f ms", t_baseline); else printf("      OOM ");
        printf(" | %8.2f ms | %8.2f ms | %8.2f ms | ", t_v1, t_v2, t_v3);
        if (t_baseline > 0) printf("%8.2fx", t_baseline / t_v3); else printf("      N/A ");
        printf("\n");

        // Print memory savings
        size_t baseline_mem = sizeof(float) * (QO_size + KV_size * 2 + QK_size * 2);
        size_t flash_mem = sizeof(float) * (QO_size + KV_size * 2 + lm_size * 2);
        // (printed at end)

        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
        cudaFree(d_O); cudaFree(d_l); cudaFree(d_m);
        delete[] h_Q; delete[] h_K; delete[] h_V;
    }

    printf("\n=============================================================\n");
    printf("  Memory Usage Comparison (Baseline vs FlashAttention)\n");
    printf("=============================================================\n\n");

    printf("%-30s | %12s | %12s | %10s\n",
           "[bs,nh,N,M,d]", "Baseline", "FlashAttn", "Savings");
    printf("-------------------------------|--------------|--------------|------------\n");

    for (auto& cfg : configs) {
        int bs = cfg.batch_size, nh = cfg.num_head;
        int N = cfg.N, M = cfg.M, d = cfg.d;
        size_t QO_size = (size_t)bs * nh * N * d;
        size_t KV_size = (size_t)bs * nh * M * d;
        size_t lm_size = (size_t)bs * nh * N;
        size_t QK_size = (size_t)bs * nh * N * M;

        size_t baseline_mem = sizeof(float) * (QO_size * 2 + KV_size * 2 + QK_size * 2);
        size_t flash_mem = sizeof(float) * (QO_size * 2 + KV_size * 2 + lm_size * 2);

        char config_str[64];
        snprintf(config_str, sizeof(config_str), "[%d,%d,%d,%d,%d]", bs, nh, N, M, d);

        printf("%-30s | %8.1f MB | %8.1f MB | %8.1fx\n",
               config_str,
               baseline_mem / 1024.0 / 1024.0,
               flash_mem / 1024.0 / 1024.0,
               (float)baseline_mem / flash_mem);
    }

    printf("\nDone.\n");
    return 0;
}
