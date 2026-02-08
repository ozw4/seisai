import numpy as np
import torch


# --- 共通: ビン(index)基準のガウス分布生成 ---
def gaussian_pulse1d_np(
    mu: np.ndarray | float, sigma_bins: np.ndarray | float, W: int
) -> np.ndarray:
    """Peak-normalized 1D Gaussian pulse (peak=1) on discrete bins.

    Parameters
    ----------
    mu
            Center bin index (can be scalar or array-like). For integer mu and if that bin
            exists, the maximum value becomes exactly 1.
    sigma_bins
            Standard deviation in bins. Must be > 0 (broadcastable to mu).
    W
            Output width (number of bins). Must be > 0.

    Returns
    -------
    np.ndarray
            Shape (..., W) where ... is broadcasted from mu/sigma. Dtype is float32.

    Raises
    ------
    ValueError
            If W <= 0 or sigma_bins <= 0, or if mu/sigma contain non-finite values.

    """
    if W <= 0:
        msg = 'W must be positive'
        raise ValueError(msg)

    m = np.asarray(mu, dtype=np.float64)
    s = np.asarray(sigma_bins, dtype=np.float64)
    if np.any(~np.isfinite(m)):
        msg = 'mu must be finite'
        raise ValueError(msg)
    if np.any(~np.isfinite(s)):
        msg = 'sigma must be finite (bins)'
        raise ValueError(msg)
    if np.any(s <= 0):
        msg = 'sigma must be positive (bins)'
        raise ValueError(msg)

    xs = np.arange(W, dtype=np.float64)
    z = -0.5 * ((xs - m[..., None]) / s[..., None]) ** 2
    g = np.exp(z).astype(np.float32, copy=False)
    return g


def gaussian_probs1d_np(
    mu: np.ndarray, sigma_bins: np.ndarray | float, W: int
) -> np.ndarray:
    """mu: (...), sigma_bins: broadcastable to mu, return shape (..., W), sum=1 along last axis"""
    if W <= 0:
        msg = 'W must be positive'
        raise ValueError(msg)
    mu = np.asarray(mu, dtype=np.float32)
    s = np.asarray(sigma_bins, dtype=np.float32)
    if np.any(s <= 0):
        msg = 'sigma must be positive (bins)'
        raise ValueError(msg)
    xs = np.arange(W, dtype=np.float32)
    z = -0.5 * ((xs - mu[..., None]) / s[..., None]) ** 2
    g = (
        np.exp(z, dtype=np.float32)
        if hasattr(np, 'exp')
        else np.exp(z).astype(np.float32)
    )  # safety
    denom = g.sum(axis=-1, keepdims=True)
    if np.any(denom <= 0.0):
        msg = 'invalid gaussian row (zero area)'
        raise ValueError(msg)
    return (g / denom).astype(np.float32)


def gaussian_probs1d_torch(
    mu: torch.Tensor, sigma_bins: torch.Tensor | float, W: int
) -> torch.Tensor:
    """mu: (...), sigma_bins: broadcastable to mu, return shape (..., W), sum=1 along last axis"""
    if W <= 0:
        msg = 'W must be positive'
        raise ValueError(msg)
    mu = mu.to(dtype=torch.float32)
    s = torch.as_tensor(sigma_bins, dtype=mu.dtype, device=mu.device)
    if torch.any(s <= 0):
        msg = 'sigma must be positive (bins)'
        raise ValueError(msg)
    s = torch.clamp(s, min=1e-6)
    bins = torch.arange(W, device=mu.device, dtype=mu.dtype).view(*([1] * mu.dim()), W)
    logits = -0.5 * ((bins - mu.unsqueeze(-1)) / s.unsqueeze(-1)) ** 2
    return torch.softmax(logits, dim=-1)
