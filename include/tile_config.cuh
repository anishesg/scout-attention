#pragma once

// Tile dimensions for the KV loop in tiled/scout attention.
// TILE_Q: number of query rows handled by one thread block.
// TILE_K: number of key/value rows loaded per KV tile iteration.
// These are tuned for 64-dim heads on an A100 (80 GB SMEM per SM).
//
// SMEM per block:
//   K tile: TILE_K * HEAD_DIM * 4 bytes
//   V tile: TILE_K * HEAD_DIM * 4 bytes
//   Q tile: TILE_Q * HEAD_DIM * 4 bytes
//   logits: TILE_Q * TILE_K * 4 bytes
// With TILE_Q=64, TILE_K=64, HEAD_DIM=128: ~256 KB (two tiles per SM).

#ifndef TILE_Q
#define TILE_Q 64
#endif

#ifndef TILE_K
#define TILE_K 64
#endif

// Warp size
#define WARP_SIZE 32
