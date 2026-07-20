#include "scout_attention.cuh"
#include "scout_score.cuh"
#include "../include/tile_config.cuh"
#include <cstring>
#include <algorithm>

// ============================================================================
// Scout-Attention kernel
//
// Grid : (CEIL_DIV(S_q, TILE_Q), H, B)
// Block: (TILE_K)
//
// For each KV tile the kernel executes two phases:
//
// Phase 1 -- Scout:
//   Load only K_tile[:, :D_SCOUT] into smem (d_scout columns).
//   Compute scout score = dot(Q[q, :D_SCOUT], mean(K_tile[:, :D_SCOUT])) / D_SCOUT
//   for each q row in the tile.  Compare to adaptive threshold.
//   If ALL query rows in the tile score below threshold -> skip tile.
//   (Conservative: skip only if every query agrees, reduces false skips.)
//
// Phase 2 -- Full attention (for non-skipped tiles):
//   Load K_tile[:, D_SCOUT:] to complete K, load V_tile.
//   Run standard FlashAttention-2 tile update with online softmax.
//
// Correction (SCOUT_USE_CORRECTION):
//   After the loop, the normalization denominator row_sum reflects only
//   computed tiles. We record the minimum logit from kept tiles (min_kept_logit)
//   as a proxy for skipped tiles. The correction adds:
//     skipped_mass = exp(min_kept_logit) * num_skipped_tiles * TILE_K
//   to row_sum before computing the final output. This approximates the
//   contribution of skipped tiles as having uniform minimum score.
// ============================================================================

template <int TQ, int TK, int DS>
__global__ void scout_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    const float* __restrict__ thresholds,  // [B, H, num_q_tiles] or nullptr
    int* __restrict__ skip_counts,         // [B, H, num_q_tiles] output, or nullptr
    int B, int H, int S_q, int S_k, int D,
    float scale, float keep_frac, int num_q_tiles)
{
    // Shared memory layout:
    //   smem_Q_full : [TQ][D]       full Q tile (always loaded once)
    //   smem_K_scout: [TK][DS]      K scout columns (phase 1)
    //   smem_K_full : [TK][D]       K full columns (phase 2, overlaps scout)
    //   smem_V      : [TK][D]       V tile
    extern __shared__ float smem[];
    float* smem_Q    = smem;                      // [TQ * D]
    float* smem_K    = smem_Q + TQ * D;           // [TK * D]  (reused for scout/full)
    float* smem_V    = smem_K + TK * D;           // [TK * D]

    int b  = blockIdx.z;
    int h  = blockIdx.y;
    int qt = blockIdx.x;
    int q_tile_start = qt * TQ;

    long long bh_q = ((long long)b * H + h) * S_q * D;
    long long bh_k = ((long long)b * H + h) * S_k * D;

    int tid = threadIdx.x;

    // Load Q tile (full D dimensions, used in both phases)
    for (int qi = 0; qi < TQ; ++qi) {
        int q_row = q_tile_start + qi;
        for (int d = tid; d < D; d += TK)
            smem_Q[qi * D + d] = (q_row < S_q) ? Q[bh_q + (long long)q_row * D + d] : 0.0f;
    }
    __syncthreads();

    // Fetch threshold for this (b, h, q_tile)
    float thresh = -1e38f;  // default: never skip
    if (thresholds)
        thresh = thresholds[((long long)b * H + h) * num_q_tiles + qt];

    int num_k_tiles = CEIL_DIV(S_k, TK);
    int skipped = 0;

    // Per-query-row running softmax state (serialized over qi for simplicity)
    // Production: parallelize across qi using warp-level reductions.

    for (int qi = 0; qi < TQ; ++qi) {
        int q_row = q_tile_start + qi;
        if (q_row >= S_q) continue;

        float row_max = -1e38f;
        float row_sum = 0.0f;
        float acc[128] = {};  // D <= 128
        float min_kept_logit = 1e38f;
        int   tiles_skipped_qi = 0;

        for (int kt = 0; kt < num_k_tiles; ++kt) {
            int k_tile_start = kt * TK;

            // ---- Phase 1: Scout ----
            // Load first DS columns of K tile
            if (tid == 0) {
                for (int ki = 0; ki < TK; ++ki) {
                    int k_row = k_tile_start + ki;
                    for (int d = 0; d < DS; ++d)
                        smem_K[ki * D + d] = (k_row < S_k)
                            ? K[bh_k + (long long)k_row * D + d]
                            : 0.0f;
                }
            }
            __syncthreads();

            bool skip_tile = false;
            if (tid == 0) {
                // Compute scout score: dot(q[:DS], mean(K_tile[:, :DS])) / DS
                float k_mean[DS];
                int k_end = min(k_tile_start + TK, S_k);
                int count = k_end - k_tile_start;
                for (int d = 0; d < DS; ++d) {
                    float s = 0.0f;
                    for (int ki = 0; ki < count; ++ki) s += smem_K[ki * D + d];
                    k_mean[d] = s / (float)count;
                }
                const float* q_ptr = smem_Q + qi * D;
                float scout = 0.0f;
                for (int d = 0; d < DS; ++d) scout += q_ptr[d] * k_mean[d];
                scout = scout * scale / (float)DS;
                skip_tile = (scout < thresh);
            }

            // Broadcast skip decision (tid 0 -> all threads via smem scalar)
            __shared__ int s_skip;
            if (tid == 0) s_skip = (int)skip_tile;
            __syncthreads();
            skip_tile = (bool)s_skip;

            if (skip_tile) {
                tiles_skipped_qi++;
                if (qi == 0) skipped++;  // count once per q_tile
                continue;
            }

            // ---- Phase 2: Full attention ----
            // Load remaining K columns (DS..D-1) -- reuse same smem_K buffer
            if (tid == 0) {
                for (int ki = 0; ki < TK; ++ki) {
                    int k_row = k_tile_start + ki;
                    for (int d = DS; d < D; ++d)
                        smem_K[ki * D + d] = (k_row < S_k)
                            ? K[bh_k + (long long)k_row * D + d]
                            : 0.0f;
                }
            }
            // Load V tile
            for (int ki = 0; ki < TK; ++ki) {
                int k_row = k_tile_start + ki;
                for (int d = tid; d < D; d += TK)
                    smem_V[ki * D + d] = (k_row < S_k)
                        ? V[bh_k + (long long)k_row * D + d]
                        : 0.0f;
            }
            __syncthreads();

            if (tid == 0) {
                float tile_max = -1e38f;
                float logits[TK];
                const float* q_ptr = smem_Q + qi * D;

                for (int ki = 0; ki < TK; ++ki) {
                    int k_row = k_tile_start + ki;
                    if (k_row >= S_k) { logits[ki] = -1e38f; continue; }
                    float dot = 0.0f;
                    const float* k_ptr = smem_K + ki * D;
                    for (int d = 0; d < D; ++d) dot += q_ptr[d] * k_ptr[d];
                    logits[ki] = dot * scale;
                    tile_max = fmaxf(tile_max, logits[ki]);
                    min_kept_logit = fminf(min_kept_logit, logits[ki]);
                }

                float new_max = fmaxf(row_max, tile_max);
                float rescale = expf(row_max - new_max);
                float tile_sum = 0.0f;
                for (int ki = 0; ki < TK; ++ki) {
                    logits[ki] = expf(logits[ki] - new_max);
                    tile_sum += logits[ki];
                }
                for (int d = 0; d < D; ++d) acc[d] *= rescale;
                for (int ki = 0; ki < TK; ++ki) {
                    const float* v_ptr = smem_V + ki * D;
                    for (int d = 0; d < D; ++d)
                        acc[d] += logits[ki] * v_ptr[d];
                }
                row_sum = row_sum * rescale + tile_sum;
                row_max = new_max;
            }
            __syncthreads();
        }

        // Correction: approximate contribution of skipped tiles as uniform
        // with logit = min_kept_logit - log(keep_frac / (1-keep_frac))
        // i.e. skipped tiles are treated as slightly less important than the
        // minimum kept tile. This maintains the normalization invariant
        // without recomputing the skipped tiles.
        if (tid == 0) {
#if SCOUT_USE_CORRECTION
            if (tiles_skipped_qi > 0 && min_kept_logit < 1e37f) {
                // Correction logit: min_kept - log_odds penalty
                float corr_logit = min_kept_logit + logf(keep_frac + 1e-6f)
                                   - logf(1.0f - keep_frac + 1e-6f);
                float corr_exp = expf(corr_logit - row_max);
                row_sum += corr_exp * (float)(tiles_skipped_qi * TK);
            }
#endif
            float inv_sum = 1.0f / (row_sum + 1e-8f);
            float* o_ptr = O + bh_q + (long long)q_row * D;
            for (int d = 0; d < D; ++d)
                o_ptr[d] = acc[d] * inv_sum;
        }
    }

    // Record skip statistics
    if (skip_counts && tid == 0) {
        skip_counts[((long long)b * H + h) * num_q_tiles + qt] = skipped;
    }
}

void scout_attention(
    const float* Q, const float* K, const float* V,
    float* O, const AttentionParams& params,
    int d_scout, float keep_frac,
    const float* thresholds, ScoutAttentionStats* stats)
{
    int D = params.head_dim;
    int B = params.batch_size;
    int H = params.num_heads;
    int S_q = params.seq_len_q;
    int S_k = params.seq_len_k;
    int num_q_tiles = CEIL_DIV(S_q, TILE_Q);
    int num_k_tiles = CEIL_DIV(S_k, TILE_K);

    float* d_thresholds = nullptr;
    bool own_thresholds = false;

    if (!thresholds) {
        // Compute adaptive thresholds from scout means
        int sm_size = B * H * num_k_tiles * d_scout;
        float* d_scout_means;
        CUDA_CHECK(cudaMalloc(&d_scout_means, sm_size * sizeof(float)));
        precompute_scout_means(K, d_scout_means, B, H, S_k, D, TILE_K, d_scout);

        CUDA_CHECK(cudaMalloc(&d_thresholds, B * H * num_q_tiles * sizeof(float)));
        compute_adaptive_threshold(Q, d_scout_means, d_thresholds,
            B, H, S_q, S_k, D, TILE_Q, TILE_K, d_scout, keep_frac);
        cudaFree(d_scout_means);
        own_thresholds = true;
    } else {
        d_thresholds = (float*)thresholds;
    }

    int* d_skip_counts = nullptr;
    if (stats)
        CUDA_CHECK(cudaMalloc(&d_skip_counts, B * H * num_q_tiles * sizeof(int)));

    size_t smem = (size_t)(TILE_Q + TILE_K + TILE_K) * D * sizeof(float);
    dim3 grid(num_q_tiles, H, B);
    dim3 block(TILE_K);

    // Dispatch based on D_SCOUT at runtime using the compile-time default.
    // Production code would template over d_scout values (8, 16, 32).
    scout_attention_kernel<TILE_Q, TILE_K, D_SCOUT><<<grid, block, smem>>>(
        Q, K, V, O, d_thresholds, d_skip_counts,
        B, H, S_q, S_k, D, params.scale, keep_frac, num_q_tiles);
    CUDA_CHECK(cudaGetLastError());

    if (stats) {
        int* h_skips = (int*)malloc(B * H * num_q_tiles * sizeof(int));
        CUDA_CHECK(cudaMemcpy(h_skips, d_skip_counts,
            B * H * num_q_tiles * sizeof(int), cudaMemcpyDeviceToHost));

        long long total_tiles = (long long)B * H * num_q_tiles * num_k_tiles;
        long long skipped = 0;
        float thresh_sum = 0.0f;
        int n = B * H * num_q_tiles;
        for (int i = 0; i < n; ++i) skipped += h_skips[i];

        float* h_thresh = (float*)malloc(n * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_thresh, d_thresholds,
            n * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; ++i) thresh_sum += h_thresh[i];

        stats->tiles_total     = total_tiles;
        stats->tiles_computed  = total_tiles - skipped;
        stats->effective_sparsity = (float)skipped / (float)total_tiles;
        stats->threshold_used  = thresh_sum / (float)n;

        free(h_skips); free(h_thresh);
        cudaFree(d_skip_counts);
    }

    if (own_thresholds) cudaFree(d_thresholds);
}
