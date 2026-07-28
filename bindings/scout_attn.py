"""
scout_attn: Python functional API for the scout-attention CUDA kernel.

Usage:
    from scout_attn import scout_attention, scout_attention_causal

    # Q, K, V: [B, H, S, D] float32 CUDA tensors
    out, sparsity = scout_attention(Q, K, V, keep_frac=0.5)
    out_c, sparsity_c = scout_attention_causal(Q, K, V, keep_frac=0.5)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function
from typing import Optional, Tuple

# Lazy import of the compiled C extension.
# This allows importing the Python module before the extension is built
# (useful for type checking and documentation tools).
_C = None

def _load_extension() -> None:
    global _C
    if _C is not None:
        return
    try:
        import scout_attn._C as _ext
        _C = _ext
    except ImportError as e:
        raise ImportError(
            "scout_attn C extension not found. "
            "Build it with: cd bindings && pip install -e ."
        ) from e


def scout_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    d_scout: int = 16,
    keep_frac: float = 0.5,
) -> Tuple[Tensor, float]:
    """
    Non-causal scout attention forward pass.

    Args:
        Q, K, V:    [B, H, S, D] float32 CUDA tensors in BHSD layout.
        d_scout:    Number of head-dim dimensions used for the scout score proxy.
                    Larger values give better skip decisions at higher bandwidth cost.
        keep_frac:  Target fraction of KV tiles to keep (0 < keep_frac <= 1).
                    The actual kept fraction depends on the score distribution.

    Returns:
        (O, sparsity):
            O         : [B, H, S, D] attention output tensor.
            sparsity  : Float in [0, 1), fraction of KV tiles skipped.
    """
    _load_extension()
    O, sparsity = _C.scout_attention_forward(Q, K, V, d_scout, keep_frac, False)
    return O, sparsity


def scout_attention_causal(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    d_scout: int = 16,
    keep_frac: float = 0.5,
) -> Tuple[Tensor, float]:
    """
    Causal scout attention forward pass (autoregressive, lower-triangular mask).

    Tiles entirely in the causal future (k_start > q_max) are skipped for free
    before scout scoring. Partial tiles get a visibility-weighted scout score.
    Non-skipped partial tiles have per-element masking applied in the full
    attention computation, ensuring exact causal semantics.

    Args:
        Q, K, V:    [B, H, S, D] float32 CUDA tensors; S_q == S_k required.
        d_scout:    Scout dimension count.
        keep_frac:  Target fraction of non-causal tiles to keep.

    Returns:
        (O, sparsity):
            O         : [B, H, S, D] causally masked attention output.
            sparsity  : Total fraction of tiles skipped (causal + scout combined).
    """
    _load_extension()
    O, sparsity = _C.scout_attention_forward(Q, K, V, d_scout, keep_frac, True)
    return O, sparsity


def naive_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    causal: bool = False,
) -> Tensor:
    """
    Reference O(S^2) attention using the CUDA naive kernel. For validation only.
    """
    _load_extension()
    return _C.naive_attention_forward(Q, K, V, causal)


class ScoutAttentionFunction(Function):
    """
    torch.autograd.Function wrapper for scout attention.

    Forward computes the approximate sparse attention output.
    Backward is a placeholder: it falls back to dense SDPA gradients,
    which is correct for training with approximate forward but defeats
    the purpose of sparsity for gradient computation.

    For inference-only use, this gradient is never called.
    """

    @staticmethod
    def forward(
        ctx,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        d_scout: int,
        keep_frac: float,
        causal: bool,
    ) -> Tensor:
        _load_extension()
        O, sparsity = _C.scout_attention_forward(Q, K, V, d_scout, keep_frac, causal)
        ctx.save_for_backward(Q, K, V, O)
        ctx.causal = causal
        return O

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        Q, K, V, O = ctx.saved_tensors
        causal = ctx.causal

        # Recompute softmax weights via dense SDPA for gradient.
        # attn_mask shape for causal: handled via is_causal flag.
        scale = Q.size(-1) ** -0.5
        # Expand Q/K/V back to [B*H, S, D] for sdpa, or use 4-D directly.
        grad_Q, grad_K, grad_V = torch.ops.aten._scaled_dot_product_flash_attention_backward(
            grad_output, Q, K, V, O,
            None,  # logsumexp placeholder
            None,  # cum_seq_q
            None,  # cum_seq_k
            Q.size(2), K.size(2),
            0.0,   # dropout
            causal,
            scale,
        )
        return grad_Q, grad_K, grad_V, None, None, None


def scout_attention_autograd(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    d_scout: int = 16,
    keep_frac: float = 0.5,
    causal: bool = False,
) -> Tensor:
    """
    Scout attention wrapped in a differentiable autograd Function.

    Forward uses the sparse scout kernel; backward uses dense SDPA gradients.
    Intended for inference profiling of the forward pass with autograd support.
    """
    return ScoutAttentionFunction.apply(Q, K, V, d_scout, keep_frac, causal)
