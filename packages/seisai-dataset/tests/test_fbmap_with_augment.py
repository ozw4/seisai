# %%
# tests/test_fbgmap_e2e_alignment.py
from __future__ import annotations

import numpy as np
from seisai_dataset.builder.builder import (
    FBGaussMap,
)  # factor_h/hflip/factor/start を解釈する最新版
from seisai_transforms.augment import (
    DeterministicCropOrPad,
    RandomSpatialStretchSameH,
    RandomTimeStretch,
    ViewCompose,
)

# RandomSpatialStretchSameH は did_space を廃し factor_h のみ返す版を想定
from seisai_transforms.config import SpaceAugConfig, TimeAugConfig
from seisai_transforms.view_projection import (
    project_fb_idx_view,
)


def _make_pulse_stack(H=32, W=256, sigma_samples=3.0, seed=0):
    r = np.random.default_rng(seed)
    x = np.zeros((H, W), dtype=np.float32)
    fb_idx = r.integers(low=W // 8, high=W - W // 8, size=H, endpoint=False).astype(
        np.int64
    )
    fb_idx[:3] = -1
    fb_idx[-2:] = -1
    xs = np.arange(W, dtype=np.float32)
    for h in range(H):
        if fb_idx[h] >= 0:
            c = float(fb_idx[h])
            g = np.exp(-0.5 * ((xs - c) / float(sigma_samples)) ** 2).astype(np.float32)
            g /= max(g.max(), 1e-6)
            x[h] += (0.8 + 0.4 * r.random()) * g
        x[h] += 0.03 * r.standard_normal(W).astype(np.float32)
    return x, fb_idx


def _argmax_per_row(a: np.ndarray) -> np.ndarray:
    return np.argmax(a, axis=1).astype(np.int64)


def _row_sum(a: np.ndarray) -> np.ndarray:
    return a.sum(axis=1)


def _maybe_visualize(
    case_name: str,
    x_view: np.ndarray,
    fb_map: np.ndarray,
    t_amp: np.ndarray,
    t_gauss: np.ndarray,
    viz_show: bool,
) -> None:
    if not viz_show:
        return
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    fig.suptitle(f'{case_name} — amplitude vs. FBGaussMap peak alignment')

    ax = axes[0]
    ax.imshow(
        x_view.T, aspect='auto', origin='lower', cmap='seismic', vmin=-3.0, vmax=+3.0
    )
    ax.scatter(
        np.arange(x_view.shape[0]), t_amp, s=8, c='k', marker='x', label='amp-peak'
    )
    ax.set_title('x_view')
    ax.set_xlabel('Trace (H)')
    ax.set_ylabel('Time (samples)')
    ax.legend(loc='upper right', fontsize=8)

    ax = axes[1]
    im = ax.imshow(fb_map.T, aspect='auto', origin='lower', cmap='jet')
    ax.scatter(
        np.arange(fb_map.shape[0]), t_gauss, s=8, c='w', marker='o', label='gauss-peak'
    )
    ax.set_title('fb_map (area=1)')
    ax.set_xlabel('Trace (H)')
    ax.set_ylabel('Time (samples)')
    ax.legend(loc='upper right', fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.show()


def test_alignment_time_only(viz_show=False) -> None:
    np.random.seed(0)
    H, W = 32, 256
    x, fb = _make_pulse_stack(H, W, sigma_samples=4.0, seed=42)
    dt_sec = 0.002
    f_t, L = 1.5, 256

    tf = ViewCompose(
        [
            RandomTimeStretch(TimeAugConfig(prob=1.0, factor_range=(f_t, f_t))),
            DeterministicCropOrPad(target_len=L),
        ]
    )
    x_view, meta = tf(x, rng=np.random.default_rng(0), return_meta=True)
    assert x_view.shape == (H, L)

    H, W = x_view.shape
    x.shape[1]

    meta['fb_idx_view'] = project_fb_idx_view(fb, H, W, meta)
    t_peak_amp = _argmax_per_row(np.abs(x_view))
    sample = {'x_view': x_view, 'dt_sec': dt_sec, 'meta': meta}

    op = FBGaussMap(dst='fb_map', sigma=4.0)
    op(sample)
    fb_map = sample['fb_map']
    assert fb_map.shape == (H, L)

    valid = _row_sum(fb_map) > 0
    assert valid.any()

    t_peak_gauss = _argmax_per_row(fb_map)
    tol = 2

    diff = np.abs(t_peak_amp[valid] - t_peak_gauss[valid])
    _maybe_visualize('time_only', x_view, fb_map, t_peak_amp, t_peak_gauss, viz_show)

    assert np.all(diff <= tol), (
        f'peak misalignment (time-only): max diff={diff.max()} > {tol}'
    )

    area = _row_sum(fb_map[valid])
    assert np.allclose(area, 1.0, atol=1e-5)


def test_alignment_h_and_t(viz_show=False) -> None:
    H, W = 24, 192
    x, fb = _make_pulse_stack(H, W, sigma_samples=3.0, seed=7)
    dt_sec = 0.004
    f_h, f_t, L = 0.9, 1.1, 192

    tf = ViewCompose(
        [
            # RandomHFlip(prob=1.0),
            RandomSpatialStretchSameH(
                SpaceAugConfig(prob=1.0, factor_range=(f_h, f_h))
            ),
            RandomTimeStretch(TimeAugConfig(prob=1.0, factor_range=(f_t, f_t))),
            DeterministicCropOrPad(target_len=L),
        ]
    )
    x_view, meta = tf(x, rng=np.random.default_rng(0), return_meta=True)
    assert x_view.shape == (H, L)

    t_peak_amp = _argmax_per_row(np.abs(x_view))

    sample = {'x_view': x_view, 'fb_idx': fb.copy(), 'dt_sec': dt_sec, 'meta': meta}
    meta['fb_idx_view'] = project_fb_idx_view(fb, H, L, meta)
    op = FBGaussMap(dst='fb_map', sigma=3.0)
    op(sample)
    fb_map = sample['fb_map']

    valid = _row_sum(fb_map) > 0
    assert valid.any()

    t_peak_gauss = _argmax_per_row(fb_map)
    tol = 2
    diff = np.abs(t_peak_amp[valid] - t_peak_gauss[valid])
    _maybe_visualize('h_and_time', x_view, fb_map, t_peak_amp, t_peak_gauss, viz_show)

    assert np.all(diff <= tol), (
        f'peak misalignment (h+time): max diff={diff.max()} > {tol}'
    )

    area = _row_sum(fb_map[valid])
    assert np.allclose(area, 1.0, atol=1e-5)


if __name__ == '__main__':
    test_alignment_time_only(viz_show=True)
    test_alignment_h_and_t(viz_show=True)
