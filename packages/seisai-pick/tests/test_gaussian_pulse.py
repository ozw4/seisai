import numpy as np
import pytest
from seisai_pick.gaussian_prob import gaussian_pulse1d_np


def test_gaussian_pulse_mu_integer_peak_is_one() -> None:
    g = gaussian_pulse1d_np(mu=5, sigma_bins=2.0, W=32)
    assert g.dtype == np.float32
    assert g.shape == (32,)
    assert float(g.max()) == pytest.approx(1.0, abs=1e-7)
    assert float(g[5]) == pytest.approx(1.0, abs=1e-7)


def test_gaussian_pulse_is_symmetric_around_integer_mu() -> None:
    mu = 10
    W = 41
    g = gaussian_pulse1d_np(mu=mu, sigma_bins=3.0, W=W)
    assert float(g[mu]) == 1.0

    for d in range(1, 10):
        assert float(g[mu - d]) == pytest.approx(float(g[mu + d]), abs=1e-6)

    # simple unimodality check around the peak
    assert float(g[mu - 1]) < float(g[mu])
    assert float(g[mu + 1]) < float(g[mu])


@pytest.mark.parametrize(
    ('mu', 'sigma_bins', 'W'),
    [
        (0, 0.0, 8),  # sigma <= 0
        (0, -1.0, 8),  # sigma <= 0
        (0, 1.0, 0),  # W <= 0
        (0, 1.0, -3),  # W <= 0
    ],
)
def test_gaussian_pulse_invalid_inputs_raise(mu, sigma_bins, W) -> None:
    with pytest.raises(ValueError):
        gaussian_pulse1d_np(mu=mu, sigma_bins=sigma_bins, W=W)
