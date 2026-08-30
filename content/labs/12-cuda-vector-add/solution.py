CPP_DECLARATIONS = """
torch::Tensor vector_add(torch::Tensor a, torch::Tensor b);
torch::Tensor strided_copy(torch::Tensor a, int64_t stride);
torch::Tensor rms_norm(torch::Tensor x, torch::Tensor w, double eps);
"""


def cuda_source() -> str:
    return r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Coalesced: thread i touches element i, so a warp covers 128 contiguous bytes
// and the memory system serves it in one transaction.
__global__ void add_kernel(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
    auto out = torch::empty_like(a);
    int n = a.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}

// Uncoalesced: consecutive threads land `stride` floats apart, so one warp
// needs up to 32 separate transactions instead of one.
__global__ void strided_kernel(const float* a, float* out, int n, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[((long long)i * stride) % n];
    }
}

torch::Tensor strided_copy(torch::Tensor a, int64_t stride) {
    auto out = torch::empty_like(a);
    int n = a.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    strided_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(), out.data_ptr<float>(), n, (int)stride);
    return out;
}

// One block per row. The row stays close to the arithmetic units between the
// reduction and the scaling.
__global__ void rms_norm_kernel(const float* x, const float* w, float* out,
                                int cols, float eps) {
    extern __shared__ float partial[];
    int row = blockIdx.x;
    const float* x_row = x + (long long)row * cols;
    float* out_row = out + (long long)row * cols;

    // Accumulate in float32, striding so every load stays coalesced.
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = x_row[i];
        sum += v * v;
    }
    partial[threadIdx.x] = sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            partial[threadIdx.x] += partial[threadIdx.x + offset];
        }
        __syncthreads();
    }

    float scale = rsqrtf(partial[0] / cols + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        out_row[i] = x_row[i] * scale * w[i];
    }
}

torch::Tensor rms_norm(torch::Tensor x, torch::Tensor w, double eps) {
    auto out = torch::empty_like(x);
    int rows = x.size(0);
    int cols = x.size(1);
    int threads = 256;
    rms_norm_kernel<<<rows, threads, threads * sizeof(float)>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), out.data_ptr<float>(),
        cols, (float)eps);
    return out;
}
"""
