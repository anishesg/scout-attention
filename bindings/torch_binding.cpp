#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "../src/scout_attention.cuh"
#include "../src/naive_attention.cuh"
#include "../include/common.cuh"

// ---------------------------------------------------------------------------
// Tensor validation helpers
// ---------------------------------------------------------------------------

static void check_attention_inputs(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V)
{
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(),
        "Q, K, V must be CUDA tensors");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(),
        "Q, K, V must be contiguous; call .contiguous() if needed");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat32 &&
                K.scalar_type() == torch::kFloat32 &&
                V.scalar_type() == torch::kFloat32,
        "Only float32 is supported (FP16/BF16 planned)");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4 && V.dim() == 4,
        "Expected 4-D tensors with shape [B, H, S, D]");
    TORCH_CHECK(Q.size(0) == K.size(0) && Q.size(0) == V.size(0),
        "Batch size mismatch");
    TORCH_CHECK(Q.size(1) == K.size(1) && Q.size(1) == V.size(1),
        "Head count mismatch");
    TORCH_CHECK(K.size(2) == V.size(2),
        "K and V must have the same sequence length");
    TORCH_CHECK(Q.size(3) == K.size(3) && Q.size(3) == V.size(3),
        "Head dimension mismatch");
    TORCH_CHECK(Q.size(3) <= 128,
        "Head dimension must be <= 128 (current kernel limitation)");
}

static AttentionParams make_params(const torch::Tensor& Q, const torch::Tensor& K) {
    int B  = (int)Q.size(0);
    int H  = (int)Q.size(1);
    int Sq = (int)Q.size(2);
    int Sk = (int)K.size(2);
    int D  = (int)Q.size(3);
    float scale = 1.0f / sqrtf((float)D);
    return {B, H, Sq, Sk, D, scale};
}

// ---------------------------------------------------------------------------
// scout_attention_forward
//
// Args:
//   Q, K, V  : [B, H, S_q, D] float32 CUDA tensors
//   d_scout  : int, number of scout dimensions (default D_SCOUT)
//   keep_frac: float, fraction of KV tiles to keep (default 0.5)
//   causal   : bool, apply causal mask (default false)
//
// Returns:
//   O        : [B, H, S_q, D] float32 CUDA tensor
//   sparsity : Python float, fraction of tiles skipped
// ---------------------------------------------------------------------------

std::tuple<torch::Tensor, double> scout_attention_forward(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    int   d_scout_arg,
    float keep_frac,
    bool  causal)
{
    check_attention_inputs(Q, K, V);

    at::cuda::CUDAGuard guard(Q.device());

    AttentionParams params = make_params(Q, K);
    auto O = torch::empty_like(Q);

    ScoutAttentionStats stats{};
    scout_attention(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        params,
        d_scout_arg,
        keep_frac,
        /*thresholds=*/nullptr,
        &stats,
        causal);

    return std::make_tuple(O, (double)stats.effective_sparsity);
}

// ---------------------------------------------------------------------------
// naive_attention_forward (for testing/reference)
// ---------------------------------------------------------------------------

torch::Tensor naive_attention_forward(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    bool causal)
{
    check_attention_inputs(Q, K, V);

    at::cuda::CUDAGuard guard(Q.device());

    AttentionParams params = make_params(Q, K);
    auto O = torch::empty_like(Q);

    if (causal) {
        naive_attention_causal(
            Q.data_ptr<float>(),
            K.data_ptr<float>(),
            V.data_ptr<float>(),
            O.data_ptr<float>(),
            params);
    } else {
        naive_attention(
            Q.data_ptr<float>(),
            K.data_ptr<float>(),
            V.data_ptr<float>(),
            O.data_ptr<float>(),
            params);
    }

    return O;
}

// ---------------------------------------------------------------------------
// Pybind11 module registration
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Scout-attention CUDA kernels: predictive block-skipping attention";

    m.def("scout_attention_forward",
        &scout_attention_forward,
        "Scout attention forward pass with optional causal masking",
        py::arg("Q"),
        py::arg("K"),
        py::arg("V"),
        py::arg("d_scout")   = D_SCOUT,
        py::arg("keep_frac") = SCOUT_KEEP_FRAC,
        py::arg("causal")    = false);

    m.def("naive_attention_forward",
        &naive_attention_forward,
        "Naive O(S^2) attention reference (for validation)",
        py::arg("Q"),
        py::arg("K"),
        py::arg("V"),
        py::arg("causal") = false);
}
