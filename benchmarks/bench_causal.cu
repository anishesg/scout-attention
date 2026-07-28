#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <string>
#include <cuda_runtime.h>
#include "../src/scout_attention.cuh"
#include "../include/common.cuh"

// ============================================================================
// Causal sparsity composition benchmark
//
// Measures how causal masking and scout-based skipping compose multiplicatively
// for long sequences.  For a sequence of length S with tile size TK:
//
//   num_total_tiles  = (S/TQ) * (S/TK)
//   causal_tiles     = upper-triangular = ~0.5 * total (free skips)
//   scout_tiles      = keep_frac * (S/TQ) * (S/TK)    (remaining tiles)
//
// Effective tiles = causal_tiles_actually_computed + scout_tiles
//   causal effective = keep_frac * 0.5 * total
//
// For causal mode:
//   effective_sparsity = 1 - keep_frac * 0.5  (approximately)
//
// For non-causal mode:
//   effective_sparsity = 1 - keep_frac
//
// As S grows, causal masking's contribution becomes more significant because
// the ratio of upper-triangular tiles stays constant (~50%) and compounds
// with scout sparsity.
//
// Output: CSV with seq_len, keep_frac, causal_sparsity, scout_sparsity,
//         combined_sparsity, causal_ms, scout_ms, combined_ms
// ============================================================================

struct GpuTimer {
    cudaEvent_t start, stop;
    GpuTimer()  { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~GpuTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void Start() { cudaEventRecord(start); }
    float Stop() {
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop); return ms;
    }
};

static void fill_random(float* buf, int n) {
    for (int i = 0; i < n; ++i)
        buf[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
}

struct BenchRow {
    int   seq_len;
    float keep_frac;
    float sparsity_scout_only;    // non-causal scout
    float sparsity_causal_only;   // causal + no scout (keep_frac=1.0)
    float sparsity_combined;      // causal + scout
    float ms_scout_only;
    float ms_causal_only;
    float ms_combined;
    float speedup_over_scout;     // combined vs scout-only
};

static BenchRow run_row(int B, int H, int S, int D, float keep_frac,
                        int warmup, int iters)
{
    BenchRow r{};
    r.seq_len   = S;
    r.keep_frac = keep_frac;

    long long N = (long long)B * H * S * D;
    float scale = 1.0f / sqrtf((float)D);

    float *h_Q = (float*)malloc(N * sizeof(float));
    float *h_K = (float*)malloc(N * sizeof(float));
    float *h_V = (float*)malloc(N * sizeof(float));
    fill_random(h_Q, N); fill_random(h_K, N); fill_random(h_V, N);

    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, N * sizeof(float), cudaMemcpyHostToDevice));

    AttentionParams params{B, H, S, S, D, scale};
    ScoutAttentionStats stats{};
    GpuTimer timer;

    // (1) Scout-only (non-causal), keep_frac
    for (int i = 0; i < warmup; ++i)
        scout_attention(d_Q, d_K, d_V, d_O, params, D_SCOUT, keep_frac,
            nullptr, nullptr, /*causal=*/false);
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.Start();
    for (int i = 0; i < iters; ++i)
        scout_attention(d_Q, d_K, d_V, d_O, params, D_SCOUT, keep_frac,
            nullptr, nullptr, /*causal=*/false);
    r.ms_scout_only = timer.Stop() / (float)iters;
    scout_attention(d_Q, d_K, d_V, d_O, params, D_SCOUT, keep_frac,
        nullptr, &stats, /*causal=*/false);
    r.sparsity_scout_only = stats.effective_sparsity;

    // (2) Causal-only (keep_frac=1.0 so no scout skipping, only causal free skips)
    for (int i = 0; i < warmup; ++i)
        scout_attention(d_Q, d_K, d_V, d_O, params, D_SCOUT, 1.0f,
            nullptr, nullptr, /*causal=*/true);
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.Start();
    for (int i = 0; i < iters; ++i)
        scout_attention(d_Q, d_K, d_V, d_O, params, D_SCOUT, 1.0f,
            nullptr, nullptr, /*causal=*/true);
    r.ms_causal_only = timer.Stop() / (float)iters;
    scout_attention(d_Q, d_K, d_V, d_O, params, D_SCOUT, 1.0f,
        nullptr, &stats, /*causal=*/true);
    r.sparsity_causal_only = stats.effective_sparsity;

    // (3) Combined: causal + scout
    for (int i = 0; i < warmup; ++i)
        scout_attention(d_Q, d_K, d_V, d_O, params, D_SCOUT, keep_frac,
            nullptr, nullptr, /*causal=*/true);
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.Start();
    for (int i = 0; i < iters; ++i)
        scout_attention(d_Q, d_K, d_V, d_O, params, D_SCOUT, keep_frac,
            nullptr, nullptr, /*causal=*/true);
    r.ms_combined = timer.Stop() / (float)iters;
    scout_attention(d_Q, d_K, d_V, d_O, params, D_SCOUT, keep_frac,
        nullptr, &stats, /*causal=*/true);
    r.sparsity_combined = stats.effective_sparsity;

    r.speedup_over_scout = r.ms_scout_only / r.ms_combined;

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    free(h_Q); free(h_K); free(h_V);
    return r;
}

int main(int argc, char** argv) {
    srand(42);
    const char* out_file = (argc > 1) ? argv[1] : "results/bench_causal.csv";

    int dev; cudaGetDevice(&dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    printf("Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    printf("\nCausal sparsity composition benchmark\n");
    printf("Shows multiplicative composition of causal + scout sparsity\n");
    printf("Prediction: combined_sparsity >= causal_sparsity + (1-causal_sparsity)*(1-keep_frac)\n\n");

    const int B = 1, H = 16, D = 64;
    int   seq_lens[]   = {512, 1024, 2048, 4096, 8192};
    float keep_fracs[] = {0.25f, 0.50f, 0.75f};

    system("mkdir -p results");
    FILE* fp = fopen(out_file, "w");
    fprintf(fp, "seq_len,keep_frac,sparsity_scout_only,sparsity_causal_only,"
                "sparsity_combined,predicted_combined,"
                "ms_scout_only,ms_causal_only,ms_combined,speedup_vs_scout\n");

    printf("%-8s %-9s %-14s %-14s %-16s %-16s %-8s\n",
           "seq_len", "keep_frac", "scout_sparsity", "causal_sparsity",
           "combined_sparsity", "predicted_combined", "speedup");
    printf("%s\n", std::string(95, '-').c_str());

    for (int s : seq_lens)
    for (float kf : keep_fracs) {
        BenchRow r = run_row(B, H, s, D, kf, 3, 10);

        // Theoretical prediction: causal gives ~50% free, scout operates on remainder
        float predicted = 1.0f - (1.0f - r.sparsity_causal_only) * kf;

        fprintf(fp, "%d,%.2f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                r.seq_len, r.keep_frac,
                r.sparsity_scout_only, r.sparsity_causal_only,
                r.sparsity_combined, predicted,
                r.ms_scout_only, r.ms_causal_only, r.ms_combined,
                r.speedup_over_scout);

        printf("%-8d %-9.2f %-14.1f%% %-14.1f%% %-16.1f%% %-16.1f%% %-8.2fx\n",
               r.seq_len, r.keep_frac,
               r.sparsity_scout_only * 100.0f,
               r.sparsity_causal_only * 100.0f,
               r.sparsity_combined * 100.0f,
               predicted * 100.0f,
               r.speedup_over_scout);
    }

    fclose(fp);
    printf("\nResults written to %s\n", out_file);
    return 0;
}
