// subtractive_attention/subtractive_attention.cpp

#include <torch/extension.h>

// Declaration of the CUDA function
void subtractive_attention_cuda(
    torch::Tensor inputs,
    torch::Tensor keys,
    torch::Tensor similarities
);

// C++ interface
torch::Tensor subtractive_attention(
    torch::Tensor inputs,
    torch::Tensor keys
) {
    // Ensure inputs are CUDA tensors
    if (!inputs.is_cuda()) {
        throw std::invalid_argument("inputs must be a CUDA tensor");
    }
    if (!keys.is_cuda()) {
        throw std::invalid_argument("keys must be a CUDA tensor");
    }

    const int batch_size = inputs.size(0);
    const int seq_len = inputs.size(1);
    const int num_tokens = keys.size(0);

    // Allocate output tensor
    auto similarities = torch::empty({batch_size, seq_len, num_tokens}, inputs.options());

    // Launch CUDA kernel
    subtractive_attention_cuda(inputs, keys, similarities);

    return similarities;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("subtractive_attention", &subtractive_attention, "Subtractive Attention CUDA");
}
