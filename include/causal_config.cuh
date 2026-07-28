#pragma once

#include "tile_config.cuh"

// Causal tile predicates for block-sparse autoregressive attention.
//
// A KV tile spans rows [k_tile_start, k_tile_start + TK).
// A Q tile spans rows [q_tile_start, q_tile_start + TQ).
//
// Three distinct cases:
//   TILE_FULLY_MASKED  - all k positions are strictly after ALL q positions
//                        (entire tile is in the causal future; skip for free)
//   TILE_FULLY_VISIBLE - all k positions are <= all q positions
//                        (no masking needed; proceed as non-causal)
//   TILE_PARTIAL       - tile straddles the causal boundary
//                        (must apply per-element mask inside the kernel)

enum CausalTileType {
    TILE_FULLY_VISIBLE = 0,
    TILE_PARTIAL       = 1,
    TILE_FULLY_MASKED  = 2,
};

// Classify a (q_tile, k_tile) pair for causal masking.
//
// The minimum q index in the Q tile is q_tile_start.
// The maximum k index in the KV tile is k_tile_start + TK - 1.
// The maximum q index in the Q tile is q_tile_start + TQ - 1.
// The minimum k index in the KV tile is k_tile_start.
//
// Fully masked:  k_tile_start > (q_tile_start + TQ - 1)
//   -> cheapest k position is already beyond most advanced q position.
// Fully visible: (k_tile_start + TK - 1) <= q_tile_start
//   -> most expensive k position is still behind least advanced q position.
// Partial: otherwise.
__host__ __device__ __forceinline__ CausalTileType classify_causal_tile(
    int q_tile_start, int k_tile_start, int TQ_arg, int TK_arg)
{
    int q_max = q_tile_start + TQ_arg - 1;
    int k_max = k_tile_start + TK_arg - 1;

    if (k_tile_start > q_max)   return TILE_FULLY_MASKED;
    if (k_max <= q_tile_start)  return TILE_FULLY_VISIBLE;
    return TILE_PARTIAL;
}

// Per-element causal mask: returns true if position (q_pos, k_pos) is valid
// (k_pos <= q_pos in standard autoregressive attention).
__device__ __forceinline__ bool causal_valid(int q_pos, int k_pos) {
    return k_pos <= q_pos;
}

// Fraction of a causal KV tile that is actually visible to a given Q tile.
// Used to weight the scout score for partial tiles: we scale the scout score
// by the visible fraction so that a tile with only 1 valid key is not
// mistakenly treated as important as a fully visible tile.
__host__ __device__ __forceinline__ float causal_visible_fraction(
    int q_tile_start, int k_tile_start, int TQ_arg, int TK_arg)
{
    int valid = 0;
    int q_end = q_tile_start + TQ_arg;
    int k_end = k_tile_start + TK_arg;
    for (int q = q_tile_start; q < q_end; ++q)
        for (int k = k_tile_start; k < k_end; ++k)
            if (k <= q) ++valid;
    return (float)valid / (float)(TQ_arg * TK_arg);
}
