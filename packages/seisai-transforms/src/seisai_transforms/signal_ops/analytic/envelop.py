import numpy as np
from scipy.signal import hilbert


def compute_envelope(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = hilbert(x, axis=axis)  # complex analytic signal
    env = np.abs(np.asarray(z))  # envelope = magnitude
    return env.astype(x.dtype, copy=False)
