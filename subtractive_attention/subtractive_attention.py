import subtractive_attention_cuda
import torch
from torch.library import Library, impl
from torch.autograd import Function

# Create library and register operators
lib = Library("my_ops", "DEF")
lib.define("subtractive_attention(Tensor inputs, Tensor keys) -> Tensor")
lib.define("subtractive_attention_backward(Tensor grad_output, Tensor inputs, Tensor keys) -> (Tensor, Tensor)")

@impl(lib, "subtractive_attention", "CUDA")
def subtractive_attention_cuda_impl(inputs: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    return subtractive_attention_cuda.subtractive_attention(inputs, keys)

@impl(lib, "subtractive_attention_backward", "CUDA")
def subtractive_attention_backward_cuda_impl(grad_output: torch.Tensor, inputs: torch.Tensor, keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return subtractive_attention_cuda.subtractive_attention_backward(grad_output, inputs, keys)

class SubtractiveAttention(Function):
    @staticmethod
    def forward(ctx, inputs, keys):
        ctx.save_for_backward(inputs, keys)
        return torch.ops.my_ops.subtractive_attention(inputs, keys)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, keys = ctx.saved_tensors
        grad_inputs, grad_keys = torch.ops.my_ops.subtractive_attention_backward(
            grad_output, inputs, keys
        )
        return grad_inputs, grad_keys

def subtractive_attention(inputs, keys):
    return SubtractiveAttention.apply(inputs, keys)
