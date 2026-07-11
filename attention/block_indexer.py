import torch


def attention(Q, K, V, causal=False, block_size=64, top_k=16):
    """
    Q, K, V: (batch, heads, seq_len, head_dim)
    returns: (batch, heads, seq_len, head_dim)

    Two-stage Longcat-style sparse attention:
      Coarse: score each query against per-block mean-pool representatives.
      Fine:   full attention only within the top-K surviving blocks per query.

    Approximate - not numerically identical to dense attention. The mean-pool
    coarse scorer is a heuristic; a trained scorer (real Longcat) would be more
    accurate. Set top_k = seq_len // block_size to recover dense attention.

    Q is processed in chunks of 512 to avoid materialising
    (B, H, N, top_k * block_size, D) all at once (~34 GB at N=32768).
    """
    B, H, N, D = Q.shape
    assert N % block_size == 0, f"seq_len {N} must be divisible by block_size {block_size}"
    num_blocks = N // block_size
    assert top_k <= num_blocks, f"top_k {top_k} exceeds num_blocks {num_blocks}"

    scale = D ** -0.5
    chunk_size = min(512, N)

    # Reshape K, V into blocks: (B, H, num_blocks, block_size, D)
    K_blocks = K.reshape(B, H, num_blocks, block_size, D)
    V_blocks = V.reshape(B, H, num_blocks, block_size, D)

    # Block representatives via mean pool: (B, H, num_blocks, D)
    K_repr = K_blocks.mean(dim=3)

    output = torch.zeros_like(Q)

    for chunk_start in range(0, N, chunk_size):
        chunk_end = min(chunk_start + chunk_size, N)
        c = chunk_end - chunk_start
        Q_chunk = Q[:, :, chunk_start:chunk_end, :]  # (B, H, c, D)

        # --- Coarse stage ---
        # (B, H, c, D) @ (B, H, D, num_blocks) -> (B, H, c, num_blocks)
        coarse = torch.matmul(Q_chunk, K_repr.transpose(-2, -1)) * scale

        if causal:
            # Mask block j for query i if block j starts after position i
            block_starts = torch.arange(num_blocks, device=Q.device) * block_size
            q_pos = torch.arange(chunk_start, chunk_end, device=Q.device).view(1, 1, -1, 1)
            coarse = coarse.masked_fill(block_starts.view(1, 1, 1, -1) > q_pos, float('-inf'))

        # Top-K block indices per query: (B, H, c, top_k)
        _, top_indices = coarse.topk(top_k, dim=-1)

        # --- Gather K, V for selected blocks ---
        # flat_idx: (B, H, c * top_k) - flattened block indices
        flat_idx = top_indices.reshape(B, H, c * top_k)

        gather_idx = flat_idx.unsqueeze(-1).unsqueeze(-1)  # (B, H, c*top_k, 1, 1)
        gather_idx = gather_idx.expand(-1, -1, -1, block_size, D)  # (B, H, c*top_k, block_size, D)

        # Gather from K_blocks / V_blocks dim 2 (num_blocks dim)
        K_sel = K_blocks.gather(dim=2, index=gather_idx)  # (B, H, c*top_k, block_size, D)
        V_sel = V_blocks.gather(dim=2, index=gather_idx)
        K_sel = K_sel.reshape(B, H, c, top_k * block_size, D)
        V_sel = V_sel.reshape(B, H, c, top_k * block_size, D)

        # --- Fine stage ---
        # Per-query dot product: each query has its own K subset
        # (B, H, c, D) x (B, H, c, top_k*block_size, D) -> (B, H, c, top_k*block_size)
        fine = torch.einsum('bhqd,bhqkd->bhqk', Q_chunk, K_sel) * scale

        if causal:
            # Token-level mask: exclude positions beyond the query's position
            token_pos = (
                top_indices.unsqueeze(-1) * block_size
                + torch.arange(block_size, device=Q.device)
            )  # (B, H, c, top_k, block_size)
            token_pos = token_pos.reshape(B, H, c, top_k * block_size)
            q_pos_fine = torch.arange(chunk_start, chunk_end, device=Q.device).view(1, 1, -1, 1)
            fine = fine.masked_fill(token_pos > q_pos_fine, float('-inf'))

        weights = torch.softmax(fine, dim=-1)  # (B, H, c, top_k*block_size)
        output[:, :, chunk_start:chunk_end, :] = torch.einsum('bhqk,bhqkd->bhqd', weights, V_sel)

    return output
