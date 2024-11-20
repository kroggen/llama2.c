import torch
import subtractive_attention_cuda  # Import the CUDA extension with its distinct name

def subtractive_attention_custom(inputs, keys):
    """
    Custom CUDA-accelerated subtractive_attention function.

    Args:
        inputs (torch.Tensor): Tensor of shape (batch, seqlen, channels).
        keys (torch.Tensor): Tensor of shape (num_tokens, channels).

    Returns:
        torch.Tensor: Similarity scores of shape (batch, seqlen, num_tokens).
    """
    return subtractive_attention_cuda.subtractive_attention(inputs, keys)  # Call the CUDA function directly
