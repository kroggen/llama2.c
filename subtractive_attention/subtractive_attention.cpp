#include <torch/extension.h>

// Declaration of the CUDA forward function
void subtractive_attention_cuda(
    torch::Tensor inputs,
    torch::Tensor keys,
    torch::Tensor similarities
);

// Declaration of the CUDA backward function
void subtractive_attention_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor inputs,
    torch::Tensor keys,
    torch::Tensor grad_inputs,
    torch::Tensor grad_keys
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
    // Ensure inputs and keys have the correct dimensions
    TORCH_CHECK(inputs.dim() == 3, "inputs must be 3-dimensional");
    TORCH_CHECK(keys.dim() == 2, "keys must be 2-dimensional");
    TORCH_CHECK(inputs.size(2) == keys.size(1),
        "inputs and keys must have matching feature dimensions");

    const int batch_size = inputs.size(0);
    const int seq_len = inputs.size(1);
    const int num_tokens = keys.size(0);

    // Allocate output tensor
    auto similarities = torch::empty({batch_size, seq_len, num_tokens}, inputs.options());

    // Launch CUDA kernel
    subtractive_attention_cuda(inputs, keys, similarities);

    return similarities;
}

std::tuple<torch::Tensor, torch::Tensor> subtractive_attention_backward(
    torch::Tensor grad_output,
    torch::Tensor inputs,
    torch::Tensor keys
) {
    // Ensure inputs are CUDA tensors
    if (!grad_output.is_cuda() || !inputs.is_cuda() || !keys.is_cuda()) {
        throw std::invalid_argument("all tensors must be CUDA tensors");
    }

    auto grad_inputs = torch::zeros_like(inputs);
    auto grad_keys = torch::zeros_like(keys);

    // Launch CUDA kernel
    subtractive_attention_backward_cuda(grad_output, inputs, keys, grad_inputs, grad_keys);

    return std::make_tuple(grad_inputs, grad_keys);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("subtractive_attention", &subtractive_attention, "Subtractive Attention CUDA");
    m.def("subtractive_attention_backward", &subtractive_attention_backward, "Subtractive Attention Backward CUDA");
}
