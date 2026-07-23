def compute_flops(batch, heads, seq_len, head_dim, variant='dense', block_size=64, top_k=16):
    B, H, N, D = batch, heads, seq_len, head_dim

    if variant in ('naive', 'sdpa', 'flash'):
        # Two matmuls (Q*Kt and Attn*V): 2*M*K*N each
        matmul_flops = 4 * B * H * N * N * D
        softmax_flops = 5 * B * H * N * N
        return matmul_flops + softmax_flops

    elif variant in ('block_indexer', 'block_indexer_triton'):
        num_blocks = N // block_size
        # Coarse: Q @ K_repr.T where K_repr is (num_blocks, D)
        coarse_flops = 2 * B * H * N * num_blocks * D
        # Fine: full attention over top_k * block_size tokens per query
        fine_matmul = 4 * B * H * N * top_k * block_size * D
        fine_softmax = 5 * B * H * N * top_k * block_size
        return coarse_flops + fine_matmul + fine_softmax

    else:
        raise ValueError(f"Unknown variant: {variant}")


def compute_io_bytes(batch, heads, seq_len, head_dim, variant='naive', bytes_per_element=2, block_size=64):
    B, H, N, D = batch, heads, seq_len, head_dim
    e = bytes_per_element

    qkv = 3 * B * H * N * D * e
    output = B * H * N * D * e

    if variant == 'naive':
        # Scores matrix written once, read twice (softmax + Attn*V); softmax output written+read
        scores_io = 4 * B * H * N * N * e
        return qkv + scores_io + output

    elif variant in ('sdpa', 'flash'):
        # Tiled - N*N scores matrix never written to HBM
        return qkv + output

    elif variant in ('block_indexer', 'block_indexer_triton'):
        # Coarse scores: N x num_blocks (much smaller than N x N)
        num_blocks = N // block_size
        coarse_io = 2 * B * H * N * num_blocks * e
        return qkv + coarse_io + output

    else:
        raise ValueError(f"Unknown variant: {variant}")


def arithmetic_intensity(batch, heads, seq_len, head_dim, variant, bytes_per_element=2, block_size=64, top_k=16):
    flops = compute_flops(batch, heads, seq_len, head_dim, variant, block_size, top_k)
    io = compute_io_bytes(batch, heads, seq_len, head_dim, variant, bytes_per_element, block_size)
    return flops / io
