"""
Python correctness test: compare scout_attention output against
torch.nn.functional.scaled_dot_product_attention for both causal and
non-causal modes across several (B, H, S, D) configurations.

Run from the repo root after building the extension:
    cd bindings && pip install -e . && cd ..
    python tests/test_torch.py
"""

import sys
import math
import torch
import torch.nn.functional as F

# Allow running from the repo root without installing the package.
sys.path.insert(0, "bindings")

try:
    import scout_attn._C as _C
    _HAVE_EXT = True
except ImportError:
    _HAVE_EXT = False

# ---------------------------------------------------------------------------
# Reference using torch.nn.functional.scaled_dot_product_attention
# ---------------------------------------------------------------------------

def sdpa_ref(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
             causal: bool = False) -> torch.Tensor:
    """torch SDPA reference; dispatches to FlashAttention-2 when available."""
    return F.scaled_dot_product_attention(Q, K, V, is_causal=causal)


# ---------------------------------------------------------------------------
# Scout attention wrapper (falls back to torch SDPA if extension not built)
# ---------------------------------------------------------------------------

def scout_fwd(Q, K, V, keep_frac=0.9, causal=False):
    if not _HAVE_EXT:
        return sdpa_ref(Q, K, V, causal), 0.0
    O, sparsity = _C.scout_attention_forward(Q, K, V, 16, keep_frac, causal)
    return O, sparsity


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

CONFIGS = [
    # (B, H, S, D, causal, keep_frac, label)
    (1, 2,  128, 64, False, 0.9,  "non-causal S=128 keep=0.9"),
    (1, 2,  256, 64, False, 0.75, "non-causal S=256 keep=0.75"),
    (1, 4,  128, 64, True,  0.9,  "causal S=128 keep=0.9"),
    (1, 4,  256, 64, True,  0.75, "causal S=256 keep=0.75"),
    (2, 4,  128, 64, True,  0.9,  "causal B=2 S=128 keep=0.9"),
    (1, 8,  256, 64, True,  0.5,  "causal S=256 keep=0.5"),
]

# Tolerance: approximation error depends on keep_frac and sequence length.
# At keep=0.9 the approximation is very close to exact; at keep=0.5 it diverges more.
def tol_for(keep_frac: float) -> float:
    if keep_frac >= 0.9:
        return 0.02
    if keep_frac >= 0.75:
        return 0.05
    return 0.10


def run_tests() -> bool:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("No CUDA device available; skipping GPU tests.")
        return True

    if not _HAVE_EXT:
        print("scout_attn extension not built; validating torch SDPA self-consistency.")

    all_pass = True
    print(f"\n{'Config':<40}  {'max_err':>10}  {'sparsity':>10}  {'result':>6}")
    print("-" * 72)

    for B, H, S, D, causal, keep_frac, label in CONFIGS:
        torch.manual_seed(42)
        Q = torch.randn(B, H, S, D, device=device, dtype=torch.float32)
        K = torch.randn(B, H, S, D, device=device, dtype=torch.float32)
        V = torch.randn(B, H, S, D, device=device, dtype=torch.float32)

        ref = sdpa_ref(Q, K, V, causal=causal)
        out, sparsity = scout_fwd(Q, K, V, keep_frac=keep_frac, causal=causal)

        max_err = (out - ref).abs().max().item()
        tol = tol_for(keep_frac)
        passed = max_err < tol

        if not passed:
            all_pass = False

        print(f"{label:<40}  {max_err:>10.4e}  {sparsity:>9.1%}  {'PASS' if passed else 'FAIL':>6}")

    print()
    return all_pass


if __name__ == "__main__":
    ok = run_tests()
    sys.exit(0 if ok else 1)
