import torch


def measure_peak_hbm(fn):
    """
    Returns peak HBM allocated during fn() in MB, or None on CPU.

    Resets peak stats before every call so readings don't accumulate across runs.
    """
    if not torch.cuda.is_available():
        print("WARNING: HBM measurement requires CUDA - returning None")
        return None

    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)
