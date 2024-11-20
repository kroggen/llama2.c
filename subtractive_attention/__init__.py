from .subtractive_attention import subtractive_attention_custom

# Export the function with a cleaner name
subtractive_attention = subtractive_attention_custom

__all__ = ['subtractive_attention', 'subtractive_attention_custom']
