import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _fine_attn_fwd(
        Q_ptr, K_blocks_ptr, V_blocks_ptr, Out_ptr, top_indices_ptr,
        # Q / Out strides: (B, H, N, D) - Out has identical layout to Q
        stride_qb, stride_qh, stride_qn, stride_qd,
        # K_blocks / V_blocks strides: (B, H, num_blocks, block_size, D)
        stride_kb, stride_kh, stride_kbl, stride_kbs, stride_kd,
        # top_indices strides: (B, H, N, top_k)
        stride_ib, stride_ih, stride_in, stride_ik,
        B, H, N, scale,
        D: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        TOP_K: tl.constexpr,
        causal: tl.constexpr,
    ):
        """
        One program per (batch, head, query position).

        Iterates over TOP_K selected K/V blocks on-GPU using online softmax -
        the same numerically stable accumulation trick used in FlashAttention-2.
        The N×N scores matrix is never materialised; intermediates stay in registers.
        """
        pid = tl.program_id(0)
        n   = pid % N
        bh  = pid // N
        h   = bh  % H
        b   = bh  // H

        # Load Q[b, h, n, :] into registers (D,), upcast to fp32 for accumulation
        q_base = b * stride_qb + h * stride_qh + n * stride_qn
        q = tl.load(Q_ptr + q_base + tl.arange(0, D) * stride_qd).to(tl.float32)

        # Online softmax state
        m_i = tl.zeros([1], dtype=tl.float32) - float('inf')   # running max
        l_i = tl.zeros([1], dtype=tl.float32)                   # running sum(exp)
        acc = tl.zeros([D],  dtype=tl.float32)                   # unnorm output

        idx_base = b * stride_ib + h * stride_ih + n * stride_in

        for k in range(TOP_K):
            block_idx = tl.load(top_indices_ptr + idx_base + k * stride_ik)

            # Load K_block[b, h, block_idx, :, :] -> (BLOCK_SIZE, D)
            kv_base = b * stride_kb + h * stride_kh + block_idx * stride_kbl
            kv_offs = (tl.arange(0, BLOCK_SIZE)[:, None] * stride_kbs
                       + tl.arange(0, D)[None, :]         * stride_kd)
            K_block = tl.load(K_blocks_ptr + kv_base + kv_offs).to(tl.float32)

            # Attention scores: dot(q, K_block[s, :]) for each row s -> (BLOCK_SIZE,)
            scores = tl.sum(K_block * q[None, :], axis=1) * scale

            # Token-level causal mask
            if causal:
                tok_pos = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                scores = tl.where(tok_pos > n, float('-inf'), scores)

            # Online softmax update (FA2 algorithm)
            m_new = tl.maximum(m_i, tl.max(scores, axis=0))
            exp_scores = tl.exp(scores - m_new)
            l_new = l_i * tl.exp(m_i - m_new) + tl.sum(exp_scores, axis=0)

            V_block = tl.load(V_blocks_ptr + kv_base + kv_offs).to(tl.float32)
            acc = (acc * tl.exp(m_i - m_new)
                   + tl.sum(exp_scores[:, None] * V_block, axis=0))

            m_i = m_new
            l_i = l_new

        # Normalise, cast back to bf16, and write output
        out = (acc / l_i).to(tl.bfloat16)
        out_base = b * stride_qb + h * stride_qh + n * stride_qn
        tl.store(Out_ptr + out_base + tl.arange(0, D) * stride_qd, out)


def attention(Q, K, V, causal=False, block_size=64, top_k=16):
    """
    Q, K, V: (batch, heads, seq_len, head_dim)
    returns: (batch, heads, seq_len, head_dim)

    Triton-fused version of block_indexer. Same two-stage algorithm:
      Coarse: PyTorch matmul Q @ mean-pool(K) -> top-K block selection.
      Fine:   Triton kernel with online softmax over selected blocks only.

    No Python loop over Q chunks - all N queries run in parallel as Triton
    programs. The N×N attention matrix is never written to HBM.

    Requires CUDA + triton (bundled with PyTorch 2.x on GPU instances).
    """
    if not TRITON_AVAILABLE or not torch.cuda.is_available():
        raise RuntimeError(
            "block_indexer_triton requires CUDA and the triton package. "
            "Install requirements-gpu.txt on a GPU instance, or use "
            "block_indexer for CPU testing."
        )

    B, H, N, D = Q.shape
    assert N % block_size == 0, f"seq_len {N} must be divisible by block_size {block_size}"
    num_blocks = N // block_size
    assert top_k <= num_blocks, f"top_k {top_k} exceeds num_blocks {num_blocks}"

    scale = D ** -0.5
    Q_c = Q.contiguous()

    # Reshape into blocks - contiguous views, no data copy
    K_blocks = K.reshape(B, H, num_blocks, block_size, D).contiguous()
    V_blocks = V.reshape(B, H, num_blocks, block_size, D).contiguous()

    # --- Coarse stage (PyTorch) ---
    K_repr = K_blocks.float().mean(dim=3)
    coarse = torch.matmul(Q_c.float(), K_repr.transpose(-2, -1)) * scale

    if causal:
        block_starts = torch.arange(num_blocks, device=Q.device) * block_size
        q_pos = torch.arange(N, device=Q.device).view(1, 1, -1, 1)
        coarse = coarse.masked_fill(block_starts.view(1, 1, 1, -1) > q_pos, float('-inf'))

    _, top_indices = coarse.topk(top_k, dim=-1)
    top_indices = top_indices.contiguous().to(torch.int32)

    # --- Fine stage (Triton) ---
    Out = torch.empty_like(Q_c)
    grid = (B * H * N,)

    _fine_attn_fwd[grid](
        Q_c, K_blocks, V_blocks, Out, top_indices,
        Q_c.stride(0),      Q_c.stride(1),      Q_c.stride(2),      Q_c.stride(3),
        K_blocks.stride(0), K_blocks.stride(1), K_blocks.stride(2), K_blocks.stride(3), K_blocks.stride(4),
        top_indices.stride(0), top_indices.stride(1), top_indices.stride(2), top_indices.stride(3),
        B, H, N, scale,
        D=D,
        BLOCK_SIZE=block_size,
        TOP_K=top_k,
        causal=causal,
    )
    return Out
