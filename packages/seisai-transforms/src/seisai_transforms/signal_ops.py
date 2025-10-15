import numpy as np

def standardize_per_trace(x_hw: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Row-wise (per-trace) zero-mean, unit-std standardization.
    x_hw: (H, W)  H=traces, W=time
    """
    x = x_hw.astype(np.float32, copy=False)
    m = x.mean(axis=1, keepdims=True)
    s = x.std(axis=1, keepdims=True) + float(eps)
    return (x - m) / s