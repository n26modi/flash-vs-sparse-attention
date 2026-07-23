import torch

try:
    import xformers.ops as xops
    from xformers.ops.fmha.attn_bias import LocalAttentionFromBottomRightMask
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False


def attention(Q, K, V, causal=False, block_size=64, top_k=16):
    """
    Q, K, V: (batch, heads, seq_len, head_dim)
    returns: (batch, heads, seq_len, head_dim)

    xformers memory_efficient_attention with a local sliding-window mask
    at the same token density as block_indexer (top_k * block_size tokens
    per query). Fixed-pattern sparse - mask is static, not content-adaptive.

    xformers expects (B, N, H, D) - wrapper transposes in and out.

    Requires CUDA and xformers (pip install xformers).
    """
    if not XFORMERS_AVAILABLE or not torch.cuda.is_available():
        raise RuntimeError(
            "xformers_sparse requires CUDA and xformers. "
            "Install requirements-gpu.txt on a GPU instance."
        )

    B, H, N, D = Q.shape
    assert N % block_size == 0, f"seq_len {N} must be divisible by block_size {block_size}"

    window = top_k * block_size

    # xformers expects (B, N, H, D)
    Q_t = Q.transpose(1, 2).contiguous()
    K_t = K.transpose(1, 2).contiguous()
    V_t = V.transpose(1, 2).contiguous()

    attn_bias = LocalAttentionFromBottomRightMask(
        window_left=window,
        window_right=0 if causal else window,
    )

    out = xops.memory_efficient_attention(Q_t, K_t, V_t, attn_bias=attn_bias)
    return out.transpose(1, 2)
