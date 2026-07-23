import time
import torch


def measure_latency(fn, n_warmup=10, n_trials=50):
    """
    Returns timing stats in milliseconds.

    On GPU: uses CUDA events inserted into the GPU command stream so we measure
    actual kernel execution time, not CPU launch overhead.

    On CPU: falls back to time.perf_counter(). Numbers won't be meaningful for
    benchmarking but allow the infrastructure to be tested locally.
    """
    if torch.cuda.is_available():
        return _measure_gpu(fn, n_warmup, n_trials)
    else:
        return _measure_cpu(fn, n_warmup, n_trials)


def _measure_gpu(fn, n_warmup, n_trials):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(n_trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    mean = sum(times) / len(times)
    return {
        "mean_ms": mean,
        "std_ms": (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5,
        "min_ms": min(times),
        "max_ms": max(times),
    }


def _measure_cpu(fn, n_warmup, n_trials):
    print("WARNING: CPU timing - results not meaningful for benchmarking")
    for _ in range(n_warmup):
        fn()

    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)

    mean = sum(times) / len(times)
    return {
        "mean_ms": mean,
        "std_ms": (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5,
        "min_ms": min(times),
        "max_ms": max(times),
    }
