import torch

try:
    from flash_attn import flash_attn_func
    FA2_AVAILABLE = True
except ImportError:
    FA2_AVAILABLE = False


def attention(Q, K, V, causal=False):
    """
    Q, K, V: (batch, heads, seq_len, head_dim)
    returns: (batch, heads, seq_len, head_dim)

    flash_attn_func expects (batch, seq_len, heads, head_dim) - note the
    transposed layout. We transpose in, call, and transpose back so the
    caller interface stays consistent with naive and sdpa.
    """
    if not FA2_AVAILABLE:
        raise RuntimeError(
            "FlashAttention-2 requires CUDA and the flash-attn package. "
            "Install requirements-gpu.txt on a GPU instance."
        )

    # (B, H, N, D) -> (B, N, H, D)
    Q = Q.transpose(1, 2)
    K = K.transpose(1, 2)
    V = V.transpose(1, 2)

    out = flash_attn_func(Q, K, V, causal=causal)

    # (B, N, H, D) -> (B, H, N, D)
    return out.transpose(1, 2)
