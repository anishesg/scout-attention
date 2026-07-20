#include "tiled_attention.cuh"
#include "../include/tile_config.cuh"

// Tiled FlashAttention-2 kernel.
//
// Grid  : (CEIL_DIV(S_q, TILE_Q), H, B)
// Block : (TILE_K, 1, 1)  -- TILE_K threads cooperate on the KV dimension
//
// Each block handles one (b, h, q_tile) triple. It iterates over all KV
// tiles, accumulating O[q, :] with online softmax (Algorithm 2 from the
// FlashAttention-2 paper).
//
// Shared memory layout:
//   smem_Q  : [TILE_Q][HEAD_DIM]
//   smem_K  : [TILE_K][HEAD_DIM]
//   smem_V  : [TILE_K][HEAD_DIM]
//   smem_S  : [TILE_Q][TILE_K]   (attention logits scratch)

template <int TQ, int TK>
__global__ void tiled_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int B, int H, int S_q, int S_k, int D, float scale)
{
    extern __shared__ float smem[];
    float* smem_Q = smem;                       // [TQ][D]
    float* smem_K = smem_Q + TQ * D;            // [TK][D]
    float* smem_V = smem_K + TK * D;            // [TK][D]
    float* smem_S = smem_V + TK * D;            // [TQ][TK]

    int b = blockIdx.z;
    int h = blockIdx.y;
    int q_tile_start = blockIdx.x * TQ;

    long long bh_offset_q = ((long long)b * H + h) * S_q * D;
    long long bh_offset_k = ((long long)b * H + h) * S_k * D;

    int tid = threadIdx.x;  // 0 .. TK-1

    // Load Q tile into shared memory: threads cooperate in strided fashion
    // Each thread loads D/TK floats per query row (assumes D >= TK).
    for (int qi = 0; qi < TQ; ++qi) {
        int q_row = q_tile_start + qi;
        for (int d = tid; d < D; d += TK) {
            smem_Q[qi * D + d] = (q_row < S_q)
                ? Q[bh_offset_q + (long long)q_row * D + d]
                : 0.0f;
        }
    }
    __syncthreads();

    // Per-query running state: row_max, row_sum, accumulator O[q, d]
    // We can't keep per-query accumulators in registers when TQ > warp size,
    // so we serialize over query rows within the block (TQ iterations).
    // This is cache-friendly because K/V tiles stay in SMEM.
    //
    // For each query row q in the tile, we do a full KV sweep.
    // This serialization is intentional: the focus of this kernel is
    // correctness and clarity as a baseline, not maximum throughput.

    for (int qi = 0; qi < TQ; ++qi) {
        int q_row = q_tile_start + qi;
        if (q_row >= S_q) continue;

        // Running softmax state
        float row_max = -1e38f;
        float row_sum = 0.0f;

        // Output accumulator in local array (D may be large; use shared)
        // We reuse smem_S rows for the output accumulator.
        float* o_acc = smem_S + qi * TK;  // borrow TK floats per qi
        // Reset accumulator (first TK elements suffice for bookkeeping;
        // full D accumulation uses a register loop below).

        // We keep a float[D] accumulator on the stack (D <= 128 typically).
        // Stack usage: 128 * 4 = 512 bytes per thread -- acceptable.
        float acc[128] = {};  // zero-initialized; compile-time constant

        for (int k_tile_start = 0; k_tile_start < S_k; k_tile_start += TK) {
            // Load K tile
            for (int ki = 0; ki < TK; ++ki) {
                int k_row = k_tile_start + ki;
                for (int d = tid; d < D; d += TK) {
                    smem_K[ki * D + d] = (k_row < S_k)
                        ? K[bh_offset_k + (long long)k_row * D + d]
                        : 0.0f;
                }
            }
            // Load V tile
            for (int ki = 0; ki < TK; ++ki) {
                int k_row = k_tile_start + ki;
                for (int d = tid; d < D; d += TK) {
                    smem_V[ki * D + d] = (k_row < S_k)
                        ? V[bh_offset_k + (long long)k_row * D + d]
                        : 0.0f;
                }
            }
            __syncthreads();

            // Thread 0 does the per-query serial work for simplicity.
            // (For production we'd parallelize over D here.)
            if (tid == 0) {
                // Compute logits S[q, k_tile_start : k_tile_start+TK]
                float tile_max = -1e38f;
                float logits[TK];  // TK is a compile-time constant
                for (int ki = 0; ki < TK; ++ki) {
                    int k_row = k_tile_start + ki;
                    if (k_row >= S_k) { logits[ki] = -1e38f; continue; }
                    float dot = 0.0f;
                    const float* q_ptr = smem_Q + qi * D;
                    const float* k_ptr = smem_K + ki * D;
                    for (int d = 0; d < D; ++d) dot += q_ptr[d] * k_ptr[d];
                    logits[ki] = dot * scale;
                    tile_max = fmaxf(tile_max, logits[ki]);
                }

                // Online softmax rescale
                float new_max = fmaxf(row_max, tile_max);
                float rescale = expf(row_max - new_max);
                float tile_sum = 0.0f;
                for (int ki = 0; ki < TK; ++ki) {
                    logits[ki] = expf(logits[ki] - new_max);
                    tile_sum += logits[ki];
                }

                // Rescale existing accumulator and add new contribution
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

        // Write output
        if (tid == 0) {
            float inv_sum = 1.0f / row_sum;
            float* o_ptr = O + bh_offset_q + (long long)q_row * D;
            for (int d = 0; d < D; ++d)
                o_ptr[d] = acc[d] * inv_sum;
        }
    }
}

void tiled_attention(
    const float* Q, const float* K, const float* V,
    float* O, const AttentionParams& params)
{
    int D = params.head_dim;
    size_t smem = (size_t)(TILE_Q + TILE_K + TILE_K) * D * sizeof(float)
                + (size_t)TILE_Q * TILE_K * sizeof(float);

    dim3 grid(CEIL_DIV(params.seq_len_q, TILE_Q), params.num_heads, params.batch_size);
    dim3 block(TILE_K);

    tiled_attention_kernel<TILE_Q, TILE_K><<<grid, block, smem>>>(
        Q, K, V, O,
        params.batch_size, params.num_heads,
        params.seq_len_q, params.seq_len_k,
        D, params.scale);

    CUDA_CHECK(cudaGetLastError());
}
