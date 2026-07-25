import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import glob
import os

RESULTS_DIR = 'results'
OUT_DIR = 'results'

COLORS = {
    'naive':               '#aaaaaa',
    'sdpa':                '#4c9be8',
    'block_indexer':       '#f4a623',
    'block_indexer_triton':'#e05c3a',
    'flex_attn':           '#7c5cbf',
}

LABELS = {
    'naive':               'Naive (dense)',
    'sdpa':                'SDPA / FA2',
    'block_indexer':       'Block indexer (Python)',
    'block_indexer_triton':'Block indexer (Triton)',
    'flex_attn':           'FlexAttention (fixed window)',
}

L4_HBM_MB = 24 * 1024  # 24 GB


def load_latest():
    files = sorted(glob.glob(f'{RESULTS_DIR}/benchmark_full_*.csv'))
    if not files:
        raise FileNotFoundError(f"No benchmark_full_*.csv found in {RESULTS_DIR}/")
    df = pd.read_csv(files[-1])
    df = df[df['error'].isna() & ~df['skipped_oom']]
    df['mean_ms'] = pd.to_numeric(df['mean_ms'], errors='coerce')
    df['peak_hbm_mb'] = pd.to_numeric(df['peak_hbm_mb'], errors='coerce')
    df['theoretical_flops'] = pd.to_numeric(df['theoretical_flops'], errors='coerce')
    return df


def get(df, variant, causal):
    sub = df[(df['variant'] == variant) & (df['causal'] == causal)].sort_values('seq_len')
    return sub['seq_len'].values, sub['mean_ms'].values


def get_hbm(df, variant, causal):
    sub = df[(df['variant'] == variant) & (df['causal'] == causal)].sort_values('seq_len')
    return sub['seq_len'].values, sub['peak_hbm_mb'].values


def get_cols(df, variant, causal, *cols):
    sub = df[(df['variant'] == variant) & (df['causal'] == causal)].sort_values('seq_len')
    return (sub['seq_len'].values,) + tuple(sub[c].values for c in cols)


def savefig(name):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"saved {path}")


def plot_latency(df, causal):
    fig, ax = plt.subplots(figsize=(8, 5))
    tag = 'causal' if causal else 'non-causal'

    for v in ['naive', 'sdpa', 'block_indexer', 'block_indexer_triton', 'flex_attn']:
        xs, ys = get(df, v, causal)
        if len(xs) == 0:
            continue
        ax.plot(xs, ys, marker='o', label=LABELS[v], color=COLORS[v], linewidth=2)

    ax.set_xlabel('Sequence length')
    ax.set_ylabel('Latency (ms)')
    ax.set_title(f'Attention latency vs sequence length ({tag}, NVIDIA L4)')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    savefig(f'latency_{tag}.png')


def plot_hbm(df, causal):
    fig, ax = plt.subplots(figsize=(8, 5))
    tag = 'causal' if causal else 'non-causal'

    for v in ['naive', 'sdpa', 'block_indexer', 'block_indexer_triton', 'flex_attn']:
        xs, ys = get_hbm(df, v, causal)
        if len(xs) == 0:
            continue
        ax.plot(xs, ys, marker='o', label=LABELS[v], color=COLORS[v], linewidth=2)

    ax.axhline(L4_HBM_MB, color='red', linestyle='--', linewidth=1.2, label='L4 HBM limit (24 GB)')
    ax.set_xlabel('Sequence length')
    ax.set_ylabel('Peak HBM (MB)')
    ax.set_title(f'Peak GPU memory vs sequence length ({tag}, NVIDIA L4)')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    savefig(f'hbm_{tag}.png')


def plot_tflops(df, causal):
    """Achieved GFLOP/s = theoretical FLOPs / wall time. Each variant uses its own FLOPs budget."""
    fig, ax = plt.subplots(figsize=(8, 5))
    tag = 'causal' if causal else 'non-causal'

    for v in ['naive', 'sdpa', 'block_indexer', 'block_indexer_triton', 'flex_attn']:
        xs, flops, ms = get_cols(df, v, causal, 'theoretical_flops', 'mean_ms')
        if len(xs) == 0:
            continue
        mask = (ms > 0) & np.isfinite(ms) & np.isfinite(flops)
        gflops = np.where(mask, flops / (ms * 1e-3) / 1e9, np.nan)
        ax.plot(xs, gflops, marker='o', label=LABELS[v], color=COLORS[v], linewidth=2)

    ax.set_xlabel('Sequence length')
    ax.set_ylabel('Achieved GFLOP/s')
    ax.set_title(f'Compute throughput vs sequence length ({tag}, NVIDIA L4)')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    savefig(f'tflops_{tag}.png')


def plot_flops_ratio(df):
    """Block indexer FLOPs as % of dense FLOPs. Shows sparsity benefit vs crossover at small N."""
    fig, ax = plt.subplots(figsize=(8, 5))

    causal = True
    # sdpa has dense FLOPs at all seq_lens (not OOM), use it as the dense reference
    sdpa_dense = dict(zip(
        df[(df['variant'] == 'sdpa') & (df['causal'] == causal)]['seq_len'],
        df[(df['variant'] == 'sdpa') & (df['causal'] == causal)]['theoretical_flops'],
    ))

    bi = df[(df['variant'] == 'block_indexer') & (df['causal'] == causal)].sort_values('seq_len')
    xs = bi['seq_len'].values
    ratios = [bi[bi['seq_len'] == x]['theoretical_flops'].values[0] / sdpa_dense[x] * 100 for x in xs]

    ax.plot(xs, ratios, marker='o', color=COLORS['block_indexer_triton'], linewidth=2,
            label='Block indexer FLOPs / dense FLOPs')
    ax.axhline(100, color='#aaaaaa', linestyle='--', linewidth=1.0, label='Dense baseline (100%)')

    for x, r in zip(xs, ratios):
        offset = 10 if r < 100 else -15
        ax.annotate(f'{r:.0f}%', (x, r), textcoords='offset points',
                    xytext=(0, offset), fontsize=8, ha='center')

    ax.set_xlabel('Sequence length')
    ax.set_ylabel('FLOPs vs dense attention (%)')
    ax.set_title('Block indexer sparsity: FLOPs as fraction of dense (causal, NVIDIA L4)')
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:.0f}%'))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    savefig('flops_ratio.png')


def plot_fusion_speedup(df):
    fig, ax = plt.subplots(figsize=(7, 4))

    causal = True
    xs_py, ys_py = get(df, 'block_indexer', causal)
    xs_tr, ys_tr = get(df, 'block_indexer_triton', causal)

    common = np.intersect1d(xs_py, xs_tr)
    py_vals = [ys_py[np.where(xs_py == x)[0][0]] for x in common]
    tr_vals = [ys_tr[np.where(xs_tr == x)[0][0]] for x in common]
    speedup = [p / t for p, t in zip(py_vals, tr_vals)]

    bars = ax.bar([str(int(x)) for x in common], speedup, color='#e05c3a', alpha=0.85)
    for bar, s in zip(bars, speedup):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f'{s:.1f}x', ha='center', va='bottom', fontsize=8)

    ax.axhline(1, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Sequence length')
    ax.set_ylabel('Speedup (Python / Triton)')
    ax.set_title('Triton fusion speedup over Python block indexer (causal, NVIDIA L4)')
    ax.grid(True, axis='y', alpha=0.3)
    savefig('fusion_speedup.png')


if __name__ == '__main__':
    df = load_latest()
    plot_latency(df, causal=True)
    plot_hbm(df, causal=True)
    plot_tflops(df, causal=True)
    plot_flops_ratio(df)
    plot_fusion_speedup(df)
    print("done")
