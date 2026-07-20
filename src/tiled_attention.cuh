#pragma once

#include "../include/common.cuh"

// FlashAttention-2 style tiled attention kernel.
// Tiles the KV sequence dimension; maintains running (max, sum) per query row
// in registers (online softmax), never materializing the full S x S matrix.
void tiled_attention(
    const float* Q,   // [B, H, S_q, D]
    const float* K,   // [B, H, S_k, D]
    const float* V,   // [B, H, S_k, D]
    float*       O,   // [B, H, S_q, D]
    const AttentionParams& params
);
