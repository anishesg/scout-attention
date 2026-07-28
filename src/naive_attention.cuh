#pragma once

#include "../include/common.cuh"

// Full O(S^2) attention: Q @ K^T, softmax, @ V
// Output shape: [B, H, S_q, D]
void naive_attention(
    const float* Q,   // [B, H, S_q, D]
    const float* K,   // [B, H, S_k, D]
    const float* V,   // [B, H, S_k, D]
    float*       O,   // [B, H, S_q, D]
    const AttentionParams& params
);

// Causal variant: applies lower-triangular mask (k_pos <= q_pos) before softmax.
// Positions in the causal future receive logit = -inf so their softmax weight is 0.
void naive_attention_causal(
    const float* Q,   // [B, H, S_q, D]
    const float* K,   // [B, H, S_k, D]
    const float* V,   // [B, H, S_k, D]
    float*       O,   // [B, H, S_q, D]
    const AttentionParams& params
);
