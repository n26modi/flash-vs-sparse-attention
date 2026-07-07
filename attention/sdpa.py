import torch
import torch.nn.functional as F


def attention(Q, K, V, causal=False):
    """
    Q, K, V: (batch, heads, seq_len, head_dim)
    returns: (batch, heads, seq_len, head_dim)

    Thin wrapper around F.scaled_dot_product_attention. On CPU this uses the
    math (naive) backend. On a CUDA GPU with the right capabilities it dispatches
    to FlashAttention automatically.
    """
    return F.scaled_dot_product_attention(Q, K, V, is_causal=causal)
