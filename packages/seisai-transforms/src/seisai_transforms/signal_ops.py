import numpy as np
from scipy.signal import hilbert


def standardize_per_trace(x_hw: np.ndarray, eps: float = 1e-10) -> np.ndarray:
	"""Row-wise (per-trace) zero-mean, unit-std standardization.
	x_hw: (H, W)  H=traces, W=time
	"""
	x = x_hw.astype(np.float32, copy=False)
	m = x.mean(axis=1, keepdims=True)
	s = x.std(axis=1, keepdims=True) + float(eps)
	return (x - m) / s


def compute_envelope(x: np.ndarray, axis: int = -1) -> np.ndarray:
	z = hilbert(x, axis=axis)  # complex analytic signal
	env = np.abs(np.asarray(z))  # envelope = magnitude
	return env.astype(x.dtype, copy=False)
