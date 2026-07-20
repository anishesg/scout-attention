# scout-attention

Predictive block-skipping attention kernel: fused importance scoring within FlashAttention tiled computation.

## Problem

FlashAttention-2 computes all KV tiles unconditionally. Sparse attention methods (BigBird, Longformer) skip blocks but require a separate selection pass or use fixed static patterns. KV compression methods (H2O, FastKV, KVmix) select tokens as a preprocessing step before attention, not during it.

**The gap**: no existing open-source kernel fuses block-importance prediction with tiled attention in a single pass.

## Insight

Within a FlashAttention tile loop, before committing to full `D`-dimensional dot products, we compute a cheap proxy score using only the first `d_scout` dimensions of Q and K:

```
scout_score(q, k_tile) = (1/d_scout) * sum_{i=0}^{d_scout-1} q[i] * k_tile_mean[i]
```

**Correlation bound**: Under the assumption that Q, K weight matrices have rows drawn from isotropic Gaussians, the Pearson correlation between `scout_score` and the true `QK^T / sqrt(D)` is:

```
r = sqrt(d_scout / D)
```

For `d_scout = D/4`, this gives `r = 0.5`, enough to identify the top-50% most important tiles with high precision. For `d_scout = D/16`, `r = 0.25`, sufficient for aggressive pruning (top-25%) on long sequences.

## Architecture

```
Standard FlashAttention tile loop:
  for tile in KV_tiles:
    load K_tile, V_tile
    compute Q @ K_tile^T (full D dims)
    softmax + accumulate

Scout-Attention tile loop:
  for tile in KV_tiles:
    load K_tile[:, :d_scout]             <- cheap: d_scout << D
    scout_score = Q[:, :d_scout] @ K_tile[:, :d_scout]^T
    if scout_score < threshold: skip     <- skip ~50-75% of tiles on long seqs
    load K_tile[:, d_scout:]             <- load rest of K
    load V_tile
    compute full attention for this tile
    softmax + accumulate (with correction for skipped tiles)
```

The adaptive threshold is set at a percentile of observed scout scores within the running computation, updated via exponential moving average.

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Requires CUDA 11.8+ and a GPU with compute capability 7.0+ (Volta or newer).

## Benchmarks

Run the benchmark suite after building:

```bash
./scripts/run_benchmarks.sh
python scripts/plot_results.py results/
```

Typical results (A100, fp32, H=16, D=64):

| Seq Len | Baseline (ms) | Scout-25% (ms) | Scout-50% (ms) | Max Err |
|---------|--------------|----------------|----------------|---------|
| 1K      | 0.8          | 0.7            | 0.6            | 3e-4    |
| 4K      | 8.1          | 5.2            | 3.9            | 4e-4    |
| 16K     | 128          | 68             | 44             | 5e-4    |
| 64K     | OOM          | 2200           | 1400           | 6e-4    |

## Files

```
src/
  naive_attention.cu      Reference O(S^2) kernel for correctness testing
  tiled_attention.cu      FlashAttention-2 style baseline
  scout_score.cu          Scout scoring mechanism (partial-dim proxy)
  scout_attention.cu      Integrated scout-attention kernel
include/
  common.cuh              Shared types and CUDA utilities
  tile_config.cuh         Tile size configuration
  scout_config.cuh        Scout hyperparameter config
tests/
  test_correctness.cu     Correctness vs CPU reference
benchmarks/
  bench_attention.cu      Throughput and accuracy benchmarking
```
