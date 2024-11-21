#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA forward kernel for subtractive_attention
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

// CUDA backward kernel for subtractive_attention
__global__ void subtractive_attention_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ inputs,
    const float* __restrict__ keys,
    float* __restrict__ grad_inputs,
    float* __restrict__ grad_keys,
    int batch_size,
    int seq_len,
    int num_tokens,
    int channels
) {
    int b = blockIdx.x;
    int s = blockIdx.y;
    int c = threadIdx.x;

    // Use shared memory to accumulate gradients
    extern __shared__ float shared_grad_keys[];

    // Initialize shared memory
    if (c < channels) {
        shared_grad_keys[c] = 0.0f;
    }
    __syncthreads();

    // Each thread handles one channel
    if (b < batch_size && s < seq_len && c < channels) {
        float grad_input_acc = 0.0f;

        // Loop over tokens
        for (int t = 0; t < num_tokens; ++t) {
            float diff = inputs[b * seq_len * channels + s * channels + c] - keys[t * channels + c];
            float grad_scale = grad_output[b * seq_len * num_tokens + s * num_tokens + t];
            float grad_diff = (diff > 0) ? -1.0f : 1.0f;

            // Accumulate gradients
            grad_input_acc += grad_scale * grad_diff;
            atomicAdd(&shared_grad_keys[c], -grad_scale * grad_diff);
        }

        // Write accumulated grad_inputs
        grad_inputs[b * seq_len * channels + s * channels + c] = grad_input_acc;
    }

    __syncthreads();

    // Write accumulated grad_keys to global memory
    if (b == 0 && s == 0 && c < channels) {
        atomicAdd(&grad_keys[c], shared_grad_keys[c]);
    }
}

// Launcher function for forward pass
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
    const int WARP_SIZE = 32;

    // Round up num_threads to nearest multiple of WARP_SIZE
    const int rounded_threads = ((num_tokens + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

    // Calculate threads and blocks for token dimension
    const int threads = min(rounded_threads, max_threads);
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

// Launcher function for backward pass
void subtractive_attention_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor inputs,
    torch::Tensor keys,
    torch::Tensor grad_inputs,
    torch::Tensor grad_keys
) {
    const int batch_size = inputs.size(0);
    const int seq_len = inputs.size(1);
    const int num_tokens = keys.size(0);
    const int channels = inputs.size(2);

    // Use channels as thread dimension
    const int max_threads = 1024;
    const int WARP_SIZE = 32;

    // Round up num_threads to nearest multiple of WARP_SIZE
    const int rounded_threads = ((channels + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

    // Calculate threads and blocks for channel dimension
    const int threads = min(rounded_threads, max_threads);
    const int channel_blocks = (channels + threads - 1) / threads;

    dim3 blocks(batch_size, seq_len, channel_blocks);

    size_t shared_mem_size = channels * sizeof(float);

    subtractive_attention_backward_kernel<<<blocks, threads, shared_mem_size>>>(
        grad_output.data_ptr<float>(),
        inputs.data_ptr<float>(),
        keys.data_ptr<float>(),
        grad_inputs.data_ptr<float>(),
        grad_keys.data_ptr<float>(),
        batch_size,
        seq_len,
        num_tokens,
        channels
    );

    // Check for any CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in subtractive_attention_backward_kernel: %s\n", cudaGetErrorString(err));
    }
}
