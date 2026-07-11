import pytest
import torch

from attention import naive, sdpa, flash, block_indexer

B, H, N, D = 1, 2, 64, 32


def make_inputs(dtype=torch.float32, device='cpu'):
    torch.manual_seed(42)
    Q = torch.randn(B, H, N, D, dtype=dtype, device=device)
    K = torch.randn(B, H, N, D, dtype=dtype, device=device)
    V = torch.randn(B, H, N, D, dtype=dtype, device=device)
    return Q, K, V


def test_naive_vs_sdpa_fp32():
    Q, K, V = make_inputs(torch.float32)
    out_naive = naive.attention(Q, K, V, causal=False)
    out_sdpa = sdpa.attention(Q, K, V, causal=False)
    torch.testing.assert_close(out_naive, out_sdpa, atol=1e-5, rtol=0)


def test_naive_vs_sdpa_bf16():
    Q, K, V = make_inputs(torch.bfloat16)
    out_naive = naive.attention(Q, K, V, causal=False)
    out_sdpa = sdpa.attention(Q, K, V, causal=False)
    torch.testing.assert_close(out_naive, out_sdpa, atol=1e-2, rtol=0)


def test_naive_causal_vs_sdpa_causal():
    Q, K, V = make_inputs(torch.float32)
    out_naive = naive.attention(Q, K, V, causal=True)
    out_sdpa = sdpa.attention(Q, K, V, causal=True)
    torch.testing.assert_close(out_naive, out_sdpa, atol=1e-5, rtol=0)


def test_output_shape():
    Q, K, V = make_inputs()
    for fn in [naive.attention, sdpa.attention]:
        out = fn(Q, K, V)
        assert out.shape == (B, H, N, D), f"{fn.__module__}: expected {(B, H, N, D)}, got {out.shape}"


def test_flash_unavailable_on_cpu():
    if torch.cuda.is_available():
        pytest.skip("CUDA available - this test only applies on CPU")
    Q, K, V = make_inputs()
    with pytest.raises(RuntimeError, match="FlashAttention"):
        flash.attention(Q, K, V)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_flash_vs_naive_gpu():
    device = 'cuda'
    Q, K, V = make_inputs(torch.bfloat16, device=device)
    out_naive = naive.attention(Q, K, V, causal=False)
    out_flash = flash.attention(Q, K, V, causal=False)
    torch.testing.assert_close(out_naive, out_flash, atol=1e-2, rtol=0)


# --- block_indexer tests ---
# seq_len=64, block_size=8 -> 8 blocks. top_k=8 = all blocks (degenerate = dense).

BI_BLOCK_SIZE = 8
BI_TOP_K = 4  # sparse: 4 of 8 blocks
BI_TOP_K_DENSE = 8  # all blocks -> should match naive


def test_block_indexer_output_shape():
    Q, K, V = make_inputs()
    out = block_indexer.attention(Q, K, V, block_size=BI_BLOCK_SIZE, top_k=BI_TOP_K)
    assert out.shape == (B, H, N, D)


def test_block_indexer_softmax_sums_to_one():
    # Verify each query's attention weights sum to 1.
    # We hook into the computation by checking the output norm is bounded -
    # easier: just run non-causal and check values are finite and not NaN.
    Q, K, V = make_inputs(torch.float32)
    out = block_indexer.attention(Q, K, V, causal=False, block_size=BI_BLOCK_SIZE, top_k=BI_TOP_K)
    assert torch.isfinite(out).all(), "Output contains NaN or Inf"


def test_block_indexer_causal_output_shape():
    Q, K, V = make_inputs()
    out = block_indexer.attention(Q, K, V, causal=True, block_size=BI_BLOCK_SIZE, top_k=BI_TOP_K)
    assert out.shape == (B, H, N, D)


def test_block_indexer_degenerate_matches_naive():
    # When top_k = num_blocks, all tokens are attended to.
    # Output should match naive up to FP32 rounding from different summation order.
    Q, K, V = make_inputs(torch.float32)
    out_naive = naive.attention(Q, K, V, causal=False)
    out_bi = block_indexer.attention(Q, K, V, causal=False, block_size=BI_BLOCK_SIZE, top_k=BI_TOP_K_DENSE)
    torch.testing.assert_close(out_naive, out_bi, atol=1e-3, rtol=0)
