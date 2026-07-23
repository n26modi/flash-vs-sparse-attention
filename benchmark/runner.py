import argparse
import csv
import datetime
import torch

import functools

from attention import naive, sdpa, flash, block_indexer, block_indexer_triton, flex_attn, xformers_sparse
from benchmark.timer import measure_latency
from benchmark.memory import measure_peak_hbm
from benchmark.flops import compute_flops, compute_io_bytes

VARIANTS = {
    'naive': naive.attention,
    'sdpa': sdpa.attention,
    'flash': flash.attention,
    'block_indexer': functools.partial(block_indexer.attention, block_size=64, top_k=16),
    'block_indexer_triton': functools.partial(block_indexer_triton.attention, block_size=64, top_k=16),
    'flex_attn': functools.partial(flex_attn.attention, block_size=64, top_k=16),
    'xformers_sparse': functools.partial(xformers_sparse.attention, block_size=64, top_k=16),
}

SEQ_LENS = [512, 1024, 2048, 4096, 8192, 16384, 32768]
BATCH = 1
HEADS = 8
HEAD_DIM = 64
DTYPE = torch.bfloat16
NAIVE_OOM_THRESHOLD = 8192

CSV_FIELDS = [
    'variant', 'seq_len', 'causal', 'skipped_oom',
    'mean_ms', 'std_ms', 'min_ms', 'max_ms',
    'peak_hbm_mb', 'theoretical_flops',
    'theoretical_io_bytes_naive', 'theoretical_io_bytes_fa2',
    'error',
]


def run(smoke=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("WARNING: Running on CPU. Timing and memory results not meaningful.\n")

    seq_lens = [1024] if smoke else SEQ_LENS
    causal_opts = [True] if smoke else [True, False]

    results = []
    total = len(seq_lens) * len(causal_opts) * len(VARIANTS)
    done = 0

    for seq_len in seq_lens:
        for causal in causal_opts:
            for variant_name, attn_fn in VARIANTS.items():
                done += 1
                print(f"[{done}/{total}] {variant_name} | seq_len={seq_len} | causal={causal}")

                row = {'variant': variant_name, 'seq_len': seq_len, 'causal': causal, 'skipped_oom': False}

                if variant_name == 'naive' and seq_len > NAIVE_OOM_THRESHOLD:
                    row['skipped_oom'] = True
                    print("  -> skipped (expected OOM)")
                    results.append(row)
                    continue

                shape = (BATCH, HEADS, seq_len, HEAD_DIM)
                Q = torch.randn(shape, dtype=DTYPE, device=device)
                K = torch.randn(shape, dtype=DTYPE, device=device)
                V = torch.randn(shape, dtype=DTYPE, device=device)
                fn = lambda: attn_fn(Q, K, V, causal=causal)

                try:
                    timing = measure_latency(fn)
                    peak_hbm = measure_peak_hbm(fn)
                except RuntimeError as e:
                    row['error'] = str(e)
                    print(f"  -> ERROR: {e}")
                    results.append(row)
                    continue

                row.update({
                    'mean_ms': round(timing['mean_ms'], 4),
                    'std_ms': round(timing['std_ms'], 4),
                    'min_ms': round(timing['min_ms'], 4),
                    'max_ms': round(timing['max_ms'], 4),
                    'peak_hbm_mb': round(peak_hbm, 2) if peak_hbm is not None else None,
                    'theoretical_flops': compute_flops(BATCH, HEADS, seq_len, HEAD_DIM, variant=variant_name),
                    'theoretical_io_bytes_naive': compute_io_bytes(BATCH, HEADS, seq_len, HEAD_DIM, variant='naive'),
                    'theoretical_io_bytes_fa2': compute_io_bytes(BATCH, HEADS, seq_len, HEAD_DIM, variant='flash'),
                })

                hbm_str = f" | {peak_hbm:.1f}MB" if peak_hbm is not None else ""
                print(f"  -> {timing['mean_ms']:.2f}ms ± {timing['std_ms']:.2f}ms{hbm_str}")
                results.append(row)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    mode = 'smoke' if smoke else 'full'
    outpath = f"results/benchmark_{mode}_{timestamp}.csv"

    with open(outpath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, '') for k in CSV_FIELDS})

    print(f"\nSaved {len(results)} rows -> {outpath}")
    return outpath


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true', help='Quick sanity: N=1024, causal=True only')
    args = parser.parse_args()
    run(smoke=args.smoke)
