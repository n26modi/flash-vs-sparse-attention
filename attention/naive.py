import torch


def attention(Q, K, V, causal=False):
    """
    Q, K, V: (batch, heads, seq_len, head_dim)
    returns: (batch, heads, seq_len, head_dim)

    Materializes the full N×N scores matrix in HBM. At seq_len=32768 with 8
    heads in float32 that is ~32GB - expected to OOM on a 24GB A10.
    """
    scale = Q.shape[-1] ** -0.5
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, H, N, N)

    if causal:
        N = Q.shape[-2]
        mask = torch.triu(torch.ones(N, N, device=Q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, V)
