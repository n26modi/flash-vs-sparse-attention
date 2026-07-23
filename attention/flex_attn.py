import torch

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    FLEX_AVAILABLE = True
except ImportError:
    FLEX_AVAILABLE = False


def attention(Q, K, V, causal=False, block_size=64, top_k=16):
    """
    Q, K, V: (batch, heads, seq_len, head_dim)
    returns: (batch, heads, seq_len, head_dim)

    FlexAttention with a fixed sliding-window block mask at the same token
    density as block_indexer (top_k * block_size tokens per query).

    Fixed-pattern sparse: mask defined statically, not per-query content.
    PyTorch JIT-compiles the mask into a Triton kernel at first call.

    Requires PyTorch 2.5+ and CUDA.
    """
    if not FLEX_AVAILABLE or not torch.cuda.is_available():
        raise RuntimeError(
            "flex_attn requires PyTorch 2.5+ and CUDA."
        )

    B, H, N, D = Q.shape
    assert N % block_size == 0, f"seq_len {N} must be divisible by block_size {block_size}"

    # Window covers top_k * block_size tokens - same density as block_indexer
    half_window = top_k * block_size // 2

    if causal:
        def mask_mod(b, h, q_idx, kv_idx):
            return (kv_idx <= q_idx) & ((q_idx - kv_idx) <= top_k * block_size)
    else:
        def mask_mod(b, h, q_idx, kv_idx):
            diff = q_idx - kv_idx
            return (diff >= -half_window) & (diff <= half_window)

    block_mask = create_block_mask(mask_mod, B, H, N, N, device=Q.device)

    Q_c = Q.contiguous()
    K_c = K.contiguous()
    V_c = V.contiguous()

    return flex_attention(Q_c, K_c, V_c, block_mask=block_mask)
