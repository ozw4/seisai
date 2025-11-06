import numpy as np
import torch


# --- 共通: ビン（index）基準のガウス分布生成 ---
def gaussian_probs1d_np(
	mu: np.ndarray, sigma_bins: np.ndarray | float, W: int
) -> np.ndarray:
	"""mu: (...), sigma_bins: broadcastable to mu, return shape (..., W), sum=1 along last axis"""
	if W <= 0:
		raise ValueError('W must be positive')
	mu = np.asarray(mu, dtype=np.float32)
	s = np.asarray(sigma_bins, dtype=np.float32)
	if np.any(s <= 0):
		raise ValueError('sigma must be positive (bins)')
	xs = np.arange(W, dtype=np.float32)
	z = -0.5 * ((xs - mu[..., None]) / s[..., None]) ** 2
	g = (
		np.exp(z, dtype=np.float32)
		if hasattr(np, 'exp')
		else np.exp(z).astype(np.float32)
	)  # safety
	denom = g.sum(axis=-1, keepdims=True)
	if np.any(denom <= 0.0):
		raise ValueError('invalid gaussian row (zero area)')
	return (g / denom).astype(np.float32)


def gaussian_probs1d_torch(
	mu: torch.Tensor, sigma_bins: torch.Tensor | float, W: int
) -> torch.Tensor:
	"""mu: (...), sigma_bins: broadcastable to mu, return shape (..., W), sum=1 along last axis"""
	if W <= 0:
		raise ValueError('W must be positive')
	mu = mu.to(dtype=torch.float32)
	s = torch.as_tensor(sigma_bins, dtype=mu.dtype, device=mu.device)
	if torch.any(s <= 0):
		raise ValueError('sigma must be positive (bins)')
	s = torch.clamp(s, min=1e-6)
	bins = torch.arange(W, device=mu.device, dtype=mu.dtype).view(*([1] * mu.dim()), W)
	logits = -0.5 * ((bins - mu.unsqueeze(-1)) / s.unsqueeze(-1)) ** 2
	return torch.softmax(logits, dim=-1)
