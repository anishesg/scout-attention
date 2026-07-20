#include "naive_attention.cuh"

// One thread computes one output element O[b,h,q,d]
// Uses global memory for the S x S attention matrix; correct but slow.
__global__ void naive_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int B, int H, int S_q, int S_k, int D, float scale)
{
    int b = blockIdx.z;
    int h = blockIdx.y;
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= S_q) return;

    long long qkv_stride = (long long)S_q * D;  // per (b,h) block
    const float* q_ptr = Q + ((long long)b * H + h) * S_q * D + q * D;
    const float* k_base = K + ((long long)b * H + h) * S_k * D;
    const float* v_base = V + ((long long)b * H + h) * S_k * D;
    float*       o_ptr  = O + ((long long)b * H + h) * S_q * D + q * D;

    // Allocate logits in registers (S_k must be small for naive kernel)
    // For correctness testing we limit S_k <= 4096 and use local stack.
    // Production usage should use shared memory or a temp buffer.
    float row_max = -1e38f;

    // Two-pass softmax for numerical stability
    // Pass 1: compute all dot products, find max
    float* logits = new float[S_k];
    for (int k = 0; k < S_k; ++k) {
        const float* k_ptr = k_base + k * D;
        float dot = 0.0f;
        for (int d = 0; d < D; ++d)
            dot += q_ptr[d] * k_ptr[d];
        logits[k] = dot * scale;
        row_max = fmaxf(row_max, logits[k]);
    }

    // Pass 2: softmax denominator
    float denom = 0.0f;
    for (int k = 0; k < S_k; ++k) {
        logits[k] = expf(logits[k] - row_max);
        denom += logits[k];
    }
    float inv_denom = 1.0f / denom;

    // Accumulate V
    for (int d = 0; d < D; ++d) {
        float acc = 0.0f;
        for (int k = 0; k < S_k; ++k)
            acc += logits[k] * v_base[k * D + d];
        o_ptr[d] = acc * inv_denom;
    }

    delete[] logits;
}

void naive_attention(
    const float* Q,
    const float* K,
    const float* V,
    float*       O,
    const AttentionParams& params)
{
    dim3 grid(CEIL_DIV(params.seq_len_q, 32), params.num_heads, params.batch_size);
    dim3 block(32);

    naive_attention_kernel<<<grid, block>>>(
        Q, K, V, O,
        params.batch_size,
        params.num_heads,
        params.seq_len_q,
        params.seq_len_k,
        params.head_dim,
        params.scale);

    CUDA_CHECK(cudaGetLastError());
}
