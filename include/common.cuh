#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>

// Attention tensor layout: [B, H, S, D] (batch, heads, seq, head_dim)
struct AttentionParams {
    int batch_size;
    int num_heads;
    int seq_len_q;
    int seq_len_k;
    int head_dim;
    float scale;  // 1 / sqrt(head_dim)
};

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

// Warp-level reduction utilities
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int mask = 16; mask > 0; mask >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, mask);
    return val;
}
