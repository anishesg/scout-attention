#include "scout_score.cuh"
#include "../include/tile_config.cuh"
#include <cstring>
#include <algorithm>

// Kernel: for each (b, h, k_tile), compute mean of K[:, :d_scout] within tile.
// Grid: (num_k_tiles, H, B)
// Block: (d_scout)
__global__ void precompute_scout_means_kernel(
    const float* __restrict__ K,
    float* __restrict__ scout_means,
    int S_k, int D, int tile_size, int d_scout, int num_tiles)
{
    int b  = blockIdx.z;
    int h  = blockIdx.y;
    int ti = blockIdx.x;   // tile index
    int d  = threadIdx.x;  // dimension index within d_scout

    if (d >= d_scout || ti >= num_tiles) return;

    int k_start = ti * tile_size;
    int k_end   = min(k_start + tile_size, S_k);
    int count   = k_end - k_start;

    const float* K_bh = K + ((long long)b * gridDim.y + h) * S_k * D;
    float sum = 0.0f;
    for (int ki = k_start; ki < k_end; ++ki)
        sum += K_bh[(long long)ki * D + d];

    int num_heads = gridDim.y;
    scout_means[((long long)b * num_heads + h) * num_tiles * d_scout
                + ti * d_scout + d] = sum / (float)count;
}

void precompute_scout_means(
    const float* K,
    float* scout_means,
    int B, int H, int S_k, int D, int tile_size, int d_scout)
{
    int num_tiles = CEIL_DIV(S_k, tile_size);
    dim3 grid(num_tiles, H, B);
    dim3 block(d_scout);
    precompute_scout_means_kernel<<<grid, block>>>(
        K, scout_means, S_k, D, tile_size, d_scout, num_tiles);
    CUDA_CHECK(cudaGetLastError());
}

// Compute adaptive skip threshold for each (b, h, q_tile).
//
// Algorithm:
//   1. For each q row in the tile, compute all scout scores against k tiles.
//   2. Sort scores (on CPU for clarity; production would use a GPU quantile).
//   3. Threshold = score at (1 - keep_frac) quantile.
//   4. Average across q rows in the tile.
//
// This runs on CPU over pre-fetched scout means. For very long sequences
// the scout means tensor is small (S_k/tile_k * d_scout floats per head).

void compute_adaptive_threshold(
    const float* Q,
    const float* scout_means,
    float* thresholds,
    int B, int H, int S_q, int S_k, int D,
    int tile_q, int tile_k, int d_scout,
    float keep_frac)
{
    int num_q_tiles = CEIL_DIV(S_q, tile_q);
    int num_k_tiles = CEIL_DIV(S_k, tile_k);

    // Copy Q and scout_means to host for threshold computation
    int q_size = B * H * S_q * D;
    int sm_size = B * H * num_k_tiles * d_scout;

    float* h_Q  = (float*)malloc(q_size * sizeof(float));
    float* h_sm = (float*)malloc(sm_size * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_Q,  Q,           q_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sm, scout_means, sm_size * sizeof(float), cudaMemcpyDeviceToHost));

    float* h_thresh = (float*)malloc(B * H * num_q_tiles * sizeof(float));
    float* scores = (float*)malloc(num_k_tiles * sizeof(float));

    for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h)
    for (int qt = 0; qt < num_q_tiles; ++qt) {
        int q_start = qt * tile_q;
        int q_end   = std::min(q_start + tile_q, S_q);
        float tile_thresh = 0.0f;

        for (int qi = q_start; qi < q_end; ++qi) {
            const float* q_row = h_Q + ((b * H + h) * S_q + qi) * D;

            for (int kt = 0; kt < num_k_tiles; ++kt) {
                const float* sm_row = h_sm + ((b * H + h) * num_k_tiles + kt) * d_scout;
                float s = 0.0f;
                for (int d = 0; d < d_scout; ++d) s += q_row[d] * sm_row[d];
                scores[kt] = s / (float)d_scout;
            }

            // Partial sort to find (1-keep_frac) quantile
            int keep_count = std::max(1, (int)(keep_frac * num_k_tiles));
            std::nth_element(scores, scores + (num_k_tiles - keep_count), scores + num_k_tiles);
            tile_thresh += scores[num_k_tiles - keep_count];
        }

        h_thresh[(b * H + h) * num_q_tiles + qt] = tile_thresh / (float)(q_end - q_start);
    }

    CUDA_CHECK(cudaMemcpy(thresholds, h_thresh,
        B * H * num_q_tiles * sizeof(float), cudaMemcpyHostToDevice));

    free(h_Q); free(h_sm); free(h_thresh); free(scores);
}
