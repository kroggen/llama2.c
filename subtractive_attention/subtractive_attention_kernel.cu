// subtractive_attention/subtractive_attention_kernel.cu

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for subtractive_attention
__global__ void subtractive_attention_kernel(
    const float* __restrict__ inputs,
    const float* __restrict__ keys,
    float* __restrict__ similarities,
    int batch_size,
    int seq_len,
    int num_tokens,
    int channels
) {
    int b = blockIdx.x;
    int s = blockIdx.y;
    int t = blockIdx.z * blockDim.x + threadIdx.x;

    if (b < batch_size && s < seq_len && t < num_tokens) {
        float sum = 0.0f;
        for (int c = 0; c < channels; ++c) {
            float diff = inputs[b * seq_len * channels + s * channels + c] - keys[t * channels + c];
            sum += 1.0f - fabsf(diff);
        }
        similarities[b * seq_len * num_tokens + s * num_tokens + t] = sum;
    }
}

// Launcher function
void subtractive_attention_cuda(
    torch::Tensor inputs,
    torch::Tensor keys,
    torch::Tensor similarities
) {
    const int batch_size = inputs.size(0);
    const int seq_len = inputs.size(1);
    const int num_tokens = keys.size(0);
    const int channels = inputs.size(2);

    // Maximum threads per block (1024 for modern GPUs)
    const int max_threads = 1024;

    // Calculate threads and blocks for token dimension
    const int threads = min(num_tokens, max_threads);
    const int token_blocks = (num_tokens + threads - 1) / threads;

    dim3 blocks(batch_size, seq_len, token_blocks);

    subtractive_attention_kernel<<<blocks, threads>>>(
        inputs.data_ptr<float>(),
        keys.data_ptr<float>(),
        similarities.data_ptr<float>(),
        batch_size,
        seq_len,
        num_tokens,
        channels
    );

    // Check for any CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in subtractive_attention_kernel: %s\n", cudaGetErrorString(err));
    }
}
