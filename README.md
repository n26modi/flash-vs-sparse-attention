# sparse-attention-triton-bench

Benchmarking study of attention at two scales, plus a novel Triton kernel for content-adaptive sparse attention.

**Key result:** at 32k context, the Triton kernel computes 87.2B FLOPs vs 2.24T for dense (3.9%). Peak HBM stays at 1.13GB. Naive attention OOMs above 8k. FlexAttention OOMs at 32k.

Full write-up: [nishantmodi.me/posts/sparse-attention-triton-bench.html](https://nishantmodi.me/posts/sparse-attention-triton-bench.html)

---

## The two-phase problem

Attention has two scaling problems, which occur at different sequence lengths.

**Phase 1: memory bandwidth (short context).** Naive attention materializes an NxN score matrix in HBM. At N=8k that's 8.6GB of reads and writes. FlashAttention-2 tiles into SRAM so the matrix never gets written - same FLOPs, 38x lower latency, 46x less peak HBM. FA2's win is IO reduction, not fewer FLOPs.

**Phase 2: compute (long context).** Even with perfect IO efficiency, dense attention is O(N²) FLOPs. At N=32,768 that's 2.24T FLOPs per forward pass. Tiling doesn't fix this. You need fewer FLOPs.

---

## Variants

All share the same signature: `(Q, K, V, causal) -> output`

| variant | description |
|---|---|
| `naive` | raw PyTorch matmul + softmax. materializes NxN in HBM. OOMs above N=8k |
| `sdpa` | `F.scaled_dot_product_attention`. dispatches to FA2 on CUDA automatically |
| `flash` | direct `flash_attn_func`. wrapper handles (B,N,H,D) layout |
| `block_indexer` | Python two-stage sparse attention. correct math, ~450 kernel launches |
| `block_indexer_triton` | Triton-fused fine stage. online softmax. 3 kernel launches. novel contribution |
| `flex_attn` | FlexAttention with fixed sliding-window mask. compile-time sparsity baseline |

---

## Results

Benchmarked on NVIDIA L4 (sm_89, 22GB HBM). `batch=1`, `heads=8`, `head_dim=64`, `BF16`, `block_size=64`, `top_k=16`.

![Latency vs sequence length](https://raw.githubusercontent.com/n26modi/sparse-attention-triton-bench/main/results/latency_causal.png)

![Peak HBM vs sequence length](https://raw.githubusercontent.com/n26modi/sparse-attention-triton-bench/main/results/hbm_causal.png)

![FLOPs ratio vs sequence length](https://raw.githubusercontent.com/n26modi/sparse-attention-triton-bench/main/results/flops_ratio.png)

---

## Running it

**Local (CPU, no GPU required):**
```bash
pip install -r requirements-dev.txt
pytest tests/
```

**Full benchmark sweep (CUDA GPU required):**
```bash
# on a CUDA machine
pip install -r requirements-gpu.txt
python -m benchmark.runner --smoke   # sanity check: N=1024, all variants
python -m benchmark.runner           # full sweep
```

**Fixed parameters:** seq_lens 512-32768, causal True/False, naive skipped above N=8192 (OOM).

---

## Repo layout

```
attention/    # five implementations
benchmark/    # timer, memory, flops, runner
analysis/     # plotting and IO theory
tests/        # correctness tests (CPU-runnable, GPU tests skip cleanly)
results/      # CSVs and PNGs
```

---

## Stack

Python, PyTorch, Triton, GCP L4 (sm_89)
