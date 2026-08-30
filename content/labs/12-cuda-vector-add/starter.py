"""Lab 12 — your first CUDA kernel.

Return CUDA C++ source as strings. The harness compiles them with
torch.utils.cpp_extension.load_inline, checks the results against PyTorch, and
times them.

Every kernel takes float32 tensors and returns a new tensor.
"""

CPP_DECLARATIONS = """
torch::Tensor vector_add(torch::Tensor a, torch::Tensor b);
torch::Tensor strided_copy(torch::Tensor a, int64_t stride);
torch::Tensor rms_norm(torch::Tensor x, torch::Tensor w, double eps);
"""


def cuda_source() -> str:
    """Return the CUDA source defining all three functions.

    vector_add(a, b)
        Elementwise sum. One thread per element, consecutive threads on
        consecutive addresses.

    strided_copy(a, stride)
        out[i] = a[(i * stride) % n]. Deliberately uncoalesced, so the harness
        can measure what scattered access costs.

    rms_norm(x, w, eps)
        x is (rows, cols). One block per row. Each block reduces the sum of
        squares in float32, then scales the row by w. Assume cols is at most
        1024 and use a shared-memory reduction.
    """
    return r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// TODO: write add_kernel and vector_add.

// TODO: write strided_kernel and strided_copy.

// TODO: write rms_norm_kernel and rms_norm.
"""
