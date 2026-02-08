import torch
from seisai_pick.gaussian_prob import gaussian_probs1d_torch


def gaussian_prior_from_trend(
    t_trend_sec: torch.Tensor,  # (B,H)
    dt_sec: torch.Tensor,  # (B,1) or (B,)
    W: int,
    sigma_ms: float,
    ref_tensor: torch.Tensor,
    covered_mask: torch.Tensor | None = None,  # (B,H)
) -> torch.Tensor:
    """Per-trace Gaussian prior over W bins centered at t_trend, normalized per (B,H,*).
    ガウスは秒ではなくビンindex基準。mu_bin = t_trend_sec / dt_sec, sigma_bin = (sigma_ms/1000) / dt_sec。.
    """
    if sigma_ms <= 0.0:
        msg = 'sigma_ms must be positive'
        raise ValueError(msg)
    B, H = t_trend_sec.shape
    if dt_sec.dim() == 1:
        dt = dt_sec.view(B, 1).to(ref_tensor)
    else:
        dt = dt_sec.to(ref_tensor).view(B, 1)

    mu_bins = t_trend_sec.to(ref_tensor) / dt  # (B,H)
    sigma_bins = ((sigma_ms * 1e-3) / dt).expand(B, H)  # (B,H)
    prior = gaussian_probs1d_torch(mu_bins, sigma_bins, W)  # (B,H,W)

    if covered_mask is not None:
        m = covered_mask.to(torch.bool).to(prior.device).unsqueeze(-1)  # (B,H,1)
        uni = prior.new_full(prior.shape, 1.0 / W)
        prior = torch.where(m, prior, uni)
    return prior
