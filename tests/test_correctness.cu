#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include "../src/naive_attention.cuh"
#include "../include/common.cuh"

static void fill_random(float* buf, int n) {
    for (int i = 0; i < n; ++i)
        buf[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
}

// CPU reference: row-wise softmax attention
static void cpu_attention(
    const float* Q, const float* K, const float* V,
    float* O, int B, int H, int S, int D, float scale)
{
    for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h)
    for (int q = 0; q < S; ++q) {
        const float* qv = Q + ((b * H + h) * S + q) * D;
        float* ov = O + ((b * H + h) * S + q) * D;
        float row_max = -1e38f;
        float* logits = (float*)malloc(S * sizeof(float));
        for (int k = 0; k < S; ++k) {
            const float* kv = K + ((b * H + h) * S + k) * D;
            float dot = 0;
            for (int d = 0; d < D; ++d) dot += qv[d] * kv[d];
            logits[k] = dot * scale;
            row_max = fmaxf(row_max, logits[k]);
        }
        float denom = 0;
        for (int k = 0; k < S; ++k) { logits[k] = expf(logits[k] - row_max); denom += logits[k]; }
        for (int d = 0; d < D; ++d) {
            float acc = 0;
            for (int k = 0; k < S; ++k) acc += logits[k] * V[((b*H+h)*S+k)*D+d];
            ov[d] = acc / denom;
        }
        free(logits);
    }
}

int main() {
    srand(42);
    const int B = 1, H = 2, S = 256, D = 64;
    const float scale = 1.0f / sqrtf((float)D);
    const int N = B * H * S * D;

    float* h_Q = (float*)malloc(N * sizeof(float));
    float* h_K = (float*)malloc(N * sizeof(float));
    float* h_V = (float*)malloc(N * sizeof(float));
    float* h_O = (float*)malloc(N * sizeof(float));
    float* h_ref = (float*)malloc(N * sizeof(float));

    fill_random(h_Q, N);
    fill_random(h_K, N);
    fill_random(h_V, N);

    // CPU reference
    cpu_attention(h_Q, h_K, h_V, h_ref, B, H, S, D, scale);

    // GPU naive
    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, N*sizeof(float), cudaMemcpyHostToDevice));

    AttentionParams params{B, H, S, S, D, scale};
    naive_attention(d_Q, d_K, d_V, d_O, params);
    CUDA_CHECK(cudaMemcpy(h_O, d_O, N*sizeof(float), cudaMemcpyDeviceToHost));

    float max_err = 0;
    for (int i = 0; i < N; ++i)
        max_err = fmaxf(max_err, fabsf(h_O[i] - h_ref[i]));

    printf("Naive attention correctness test\n");
    printf("  B=%d H=%d S=%d D=%d\n", B, H, S, D);
    printf("  Max absolute error vs CPU: %.2e\n", max_err);
    printf("  %s\n", max_err < 1e-4f ? "PASS" : "FAIL");

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    free(h_Q); free(h_K); free(h_V); free(h_O); free(h_ref);
    return max_err < 1e-4f ? 0 : 1;
}
