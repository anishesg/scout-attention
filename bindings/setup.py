"""
PyTorch C++ extension build for scout-attention.

Build:
    pip install -e .       (editable install, rebuilds on source change)
    python setup.py build_ext --inplace  (in-place .so)

Requires:
    - CUDA 11.0+ with nvcc
    - PyTorch >= 1.12 with CUDA support
    - sm_80+ GPU (A100) for best performance; sm_70 (V100) also supported
"""

import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir   = os.path.join(repo_root, "src")
inc_dir   = os.path.join(repo_root, "include")

# Compute capability targets: sm_70 (V100), sm_75 (T4), sm_80 (A100), sm_86 (A10)
cuda_archs = os.environ.get("TORCH_CUDA_ARCH_LIST", "7.0;7.5;8.0;8.6")

nvcc_flags = [
    "-O3",
    "--use_fast_math",
    "-lineinfo",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-std=c++17",
    # Suppress warnings from torch headers in device code
    "-Xcudafe", "--diag_suppress=esa_on_defaulted_function_ignored",
]

cxx_flags = [
    "-O3",
    "-std=c++17",
    "-fvisibility=hidden",
]

setup(
    name="scout_attn",
    version="0.1.0",
    description="Predictive block-skipping attention kernel with fused scout scoring",
    ext_modules=[
        CUDAExtension(
            name="scout_attn._C",
            sources=[
                "torch_binding.cpp",
                os.path.join(src_dir, "naive_attention.cu"),
                os.path.join(src_dir, "tiled_attention.cu"),
                os.path.join(src_dir, "scout_score.cu"),
                os.path.join(src_dir, "scout_attention.cu"),
            ],
            include_dirs=[inc_dir, src_dir],
            extra_compile_args={
                "cxx":  cxx_flags,
                "nvcc": nvcc_flags,
            },
            define_macros=[
                ("SCOUT_USE_CORRECTION", "1"),
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    packages=["scout_attn"],
    package_dir={"scout_attn": "."},
    py_modules=["scout_attn"],
    python_requires=">=3.8",
)
