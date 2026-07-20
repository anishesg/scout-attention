#pragma once

// d_scout: number of head-dim dimensions used for the cheap proxy score.
// Correlation with full-dim score: r = sqrt(D_SCOUT / HEAD_DIM).
// Tradeoff: higher D_SCOUT = better skip decisions, less bandwidth saved.
//
//   D_SCOUT = HEAD_DIM/16  ->  r ~ 0.25  (aggressive skip, ~75% tiles skipped)
//   D_SCOUT = HEAD_DIM/4   ->  r ~ 0.50  (moderate skip, ~50% tiles skipped)
//   D_SCOUT = HEAD_DIM/2   ->  r ~ 0.71  (conservative, ~25% tiles skipped)
//
// Override at compile time: -DD_SCOUT=8

#ifndef D_SCOUT
#define D_SCOUT 16
#endif

// Fraction of tiles to KEEP based on scout scores.
// This is the target keep-rate; actual rate depends on score distribution.
// Set adaptively at runtime via exponential moving average of past thresholds.
#ifndef SCOUT_KEEP_FRAC
#define SCOUT_KEEP_FRAC 0.5f
#endif

// EMA decay for adaptive threshold update (per KV tile processed).
#define SCOUT_EMA_ALPHA 0.05f

// Correction: when tiles are skipped, their probability mass is distributed
// uniformly to kept tiles as an approximation. This is equivalent to
// treating skipped tiles as having the minimum kept-tile attention score.
// Set to 1 to enable this correction, 0 to hard-skip (faster, less accurate).
#ifndef SCOUT_USE_CORRECTION
#define SCOUT_USE_CORRECTION 1
#endif
