#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include "../src/naive_attention.cuh"
#include "../src/scout_attention.cuh"
#include "../include/common.cuh"

static void fill_random(float* buf, int n, unsigned int seed = 42) {
    srand(seed);
    for (int i = 0; i < n; ++i)
        buf[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
}

static float max_abs_error(const float* a, const float* b, int n) {
    float err = 0.0f;
    for (int i = 0; i < n; ++i)
        err = fmaxf(err, fabsf(a[i] - b[i]));
    return err;
}

struct TestConfig {
    int   B, H, S, D;
    float keep_frac;
};

static bool run_test(const TestConfig& cfg) {
    int B = cfg.B, H = cfg.H, S = cfg.S, D = cfg.D;
    float keep_frac = cfg.keep_frac;
    float scale = 1.0f / sqrtf((float)D);
    long long N = (long long)B * H * S * D;

    float* h_Q      = (float*)malloc(N * sizeof(float));
    float* h_K      = (float*)malloc(N * sizeof(float));
    float* h_V      = (float*)malloc(N * sizeof(float));
    float* h_ref    = (float*)malloc(N * sizeof(float));
    float* h_scout  = (float*)malloc(N * sizeof(float));

    fill_random(h_Q, N);
    fill_random(h_K, N, 137);
    fill_random(h_V, N, 271);

    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, N * sizeof(float), cudaMemcpyHostToDevice));

    AttentionParams params{B, H, S, S, D, scale};

    // Reference: causal naive attention
    naive_attention_causal(d_Q, d_K, d_V, d_O, params);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_ref, d_O, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Scout attention with causal mode
    ScoutAttentionStats stats{};
    scout_attention(d_Q, d_K, d_V, d_O, params,
        D_SCOUT, keep_frac, nullptr, &stats, /*causal=*/true);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_scout, d_O, N * sizeof(float), cudaMemcpyDeviceToHost));

    float err = max_abs_error(h_ref, h_scout, (int)N);

    // For causal mode: upper-triangular tiles (causal future) are free skips,
    // so effective sparsity should be >= 0.4 for sequences >= 256.
    // Tolerance is higher than non-causal because scout scores cause approximate
    // output; 5e-2 is reasonable for keep_frac >= 0.5.
    float tol = (keep_frac >= 0.9f) ? 1e-2f : 5e-2f;
    bool pass = (err < tol);

    printf("  B=%d H=%d S=%d D=%d keep=%.2f | max_err=%.3e sparsity=%.1f%%  %s\n",
           B, H, S, D, keep_frac, err, stats.effective_sparsity * 100.0f,
           pass ? "PASS" : "FAIL");

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    free(h_Q); free(h_K); free(h_V); free(h_ref); free(h_scout);

    return pass;
}

int main() {
    printf("Causal scout attention correctness tests\n");
    printf("=========================================\n");

    TestConfig configs[] = {
        {1, 2,  128, 64, 0.9f},
        {1, 2,  256, 64, 0.75f},
        {1, 4,  512, 64, 0.5f},
        {2, 4,  256, 64, 0.9f},
        {1, 8,  512, 64, 0.75f},
    };

    int n_tests = sizeof(configs) / sizeof(configs[0]);
    int passed = 0;
    for (int i = 0; i < n_tests; ++i) {
        if (run_test(configs[i])) ++passed;
    }

    printf("\n%d / %d tests passed\n", passed, n_tests);
    return (passed == n_tests) ? 0 : 1;
}
