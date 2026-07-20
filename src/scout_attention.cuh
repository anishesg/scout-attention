#pragma once

#include "../include/common.cuh"
#include "../include/scout_config.cuh"

struct ScoutAttentionStats {
    long long tiles_total;
    long long tiles_computed;   // tiles that passed the scout threshold
    float     effective_sparsity;  // 1 - tiles_computed/tiles_total
    float     threshold_used;   // average adaptive threshold
};

// Scout-attention forward pass.
//
// Two-phase tile evaluation per KV tile:
//   Phase 1 (scout): load K[:, :d_scout], compute proxy score, compare to threshold
//   Phase 2 (full) : load remaining K[:, d_scout:], load V, compute full attention
//
// Tiles scoring below threshold are skipped; their missing probability mass is
// redistributed to kept tiles via a correction factor (SCOUT_USE_CORRECTION=1).
//
// Parameters:
//   d_scout    -- number of dimensions for the proxy score (default: D_SCOUT)
//   keep_frac  -- fraction of tiles to keep (default: SCOUT_KEEP_FRAC)
//   thresholds -- optional precomputed per-query-tile thresholds [B,H,num_q_tiles]
//                 pass nullptr to use adaptive per-row EMA threshold
void scout_attention(
    const float* Q,           // [B, H, S_q, D]
    const float* K,           // [B, H, S_k, D]
    const float* V,           // [B, H, S_k, D]
    float*       O,           // [B, H, S_q, D]
    const AttentionParams& params,
    int          d_scout    = D_SCOUT,
    float        keep_frac  = SCOUT_KEEP_FRAC,
    const float* thresholds = nullptr,
    ScoutAttentionStats* stats = nullptr
);
