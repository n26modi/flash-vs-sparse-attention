import pytest
import torch

from attention import naive, sdpa, flash

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
