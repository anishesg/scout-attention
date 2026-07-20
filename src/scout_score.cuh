#pragma once

#include "../include/common.cuh"
#include "../include/scout_config.cuh"

// Scout score: cheap proxy for tile importance using first D_SCOUT dimensions.
//
// Given a query vector q[D] and a tile of keys K_tile[TK][D], computes:
//   scout_score = (1/D_SCOUT) * sum_{d=0}^{D_SCOUT-1} q[d] * mean(K_tile[:, d])
//
// The tile mean is precomputed and stored as k_scout_mean[D_SCOUT].
//
// This device function is called from within the KV tile loop and executes
// entirely in registers; no shared memory access beyond what the caller already
// has loaded.
__device__ __forceinline__ float compute_scout_score(
    const float* __restrict__ q_smem,     // q row in smem, length D
    const float* __restrict__ k_tile_smem, // K tile in smem, [TK][D]
    int TK, int D_SCOUT_ARG)
{
    float score = 0.0f;
    for (int d = 0; d < D_SCOUT_ARG; ++d) {
        // Compute mean of K_tile[:, d] on the fly
        float k_mean = 0.0f;
        for (int ki = 0; ki < TK; ++ki)
            k_mean += k_tile_smem[ki * (D_SCOUT_ARG) + d];
        k_mean /= (float)TK;
        score += q_smem[d] * k_mean;
    }
    return score / (float)D_SCOUT_ARG;
}

// Precompute per-tile K scout means: k_scout_means[tile][d] for d < D_SCOUT.
// Called as a preprocessing kernel before the main attention loop, or
// lazily within the loop when K tiles are loaded.
__device__ __forceinline__ void compute_k_tile_scout_mean(
    const float* __restrict__ k_tile_smem,  // [TK][D]
    float* __restrict__ k_scout_mean,       // [D_SCOUT]
    int TK, int D, int D_SCOUT_ARG)
{
    for (int d = 0; d < D_SCOUT_ARG; ++d) {
        float sum = 0.0f;
        for (int ki = 0; ki < TK; ++ki)
            sum += k_tile_smem[ki * D + d];
        k_scout_mean[d] = sum / (float)TK;
    }
}

// Compute scout score using precomputed tile means.
__device__ __forceinline__ float scout_score_from_mean(
    const float* __restrict__ q_smem,      // [D]
    const float* __restrict__ k_scout_mean, // [D_SCOUT]
    int D_SCOUT_ARG)
{
    float score = 0.0f;
    for (int d = 0; d < D_SCOUT_ARG; ++d)
        score += q_smem[d] * k_scout_mean[d];
    return score / (float)D_SCOUT_ARG;
}

// Standalone kernel: precompute scout means for all KV tiles.
// Output shape: [B, H, num_tiles, D_SCOUT]
void precompute_scout_means(
    const float* K,          // [B, H, S_k, D]
    float*       scout_means, // [B, H, num_tiles, D_SCOUT]
    int B, int H, int S_k, int D, int tile_size, int d_scout
);

// Compute per-query-tile adaptive threshold based on a sample of scout scores.
// Uses the D_SCOUT-dimensional partial dot products of Q against K scout means.
// threshold is updated via EMA across tiles to target SCOUT_KEEP_FRAC.
void compute_adaptive_threshold(
    const float* Q,           // [B, H, S_q, D]
    const float* scout_means, // [B, H, num_k_tiles, D_SCOUT]
    float*       thresholds,  // [B, H, num_q_tiles]  output
    int B, int H, int S_q, int S_k, int D,
    int tile_q, int tile_k, int d_scout,
    float keep_frac
);
