# %%
"""Quick check for SegyGatherPhasePipelineDataset (with waveform + P/S pick visualization).

Requires:
  - seisai-dataset
  - seisai-pick
  - seisai-transforms
  - seisai-utils
  - torch, numpy, segyio, matplotlib

Run:
  python packages/seisai-dataset/examples/phase_dataset_quick_check.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import segyio
import torch
from seisai_dataset import (
    BuildPlan,
    FirstBreakGate,
    FirstBreakGateConfig,
    SegyGatherPhasePipelineDataset,
)
from seisai_dataset.builder.builder import IdentitySignal, PhasePSNMap, SelectStack
from seisai_transforms.augment import (
    DeterministicCropOrPad,
    PerTraceStandardize,
    ViewCompose,
)
from seisai_utils.viz import ImshowPanel, imshow_hw, save_imshow_row
from torch.utils.data import DataLoader


def _write_empty_phase_picks_npz(path: Path, *, n_traces: int) -> None:
    """Write an NPZ file containing empty (no picks) phase-pick data for `n_traces` traces."""
    n_traces = int(n_traces)
    if n_traces <= 0:
        raise ValueError(f'n_traces must be > 0, got {n_traces}')
    indptr = np.zeros(n_traces + 1, dtype=np.int64)
    data = np.zeros(0, dtype=np.int64)
    np.savez_compressed(
        path,
        p_indptr=indptr,
        p_data=data,
        s_indptr=indptr.copy(),
        s_data=data.copy(),
    )


def _describe(name: str, v: object) -> None:
    shape = getattr(v, 'shape', None)
    dtype = getattr(v, 'dtype', None)
    print(f'{name}: type={type(v).__name__}, dtype={dtype}, shape={shape}')


def _to_numpy_hw_from_input(x: torch.Tensor) -> np.ndarray:
    """input: (C,H,W) torch -> (H,W) float32 numpy (channel 0)."""
    if not isinstance(x, torch.Tensor):
        raise ValueError(f'input must be torch.Tensor, got {type(x).__name__}')
    if x.ndim != 3:
        raise ValueError(f'input must be (C,H,W), got shape={tuple(x.shape)}')
    return x[0].detach().cpu().numpy().astype(np.float32, copy=False)


def _to_numpy_chw_from_target(y: torch.Tensor) -> np.ndarray:
    """target: (C,H,W) torch -> (C,H,W) float32 numpy."""
    if not isinstance(y, torch.Tensor):
        raise ValueError(f'target must be torch.Tensor, got {type(y).__name__}')
    if y.ndim != 3:
        raise ValueError(f'target must be (C,H,W), got shape={tuple(y.shape)}')
    return y.detach().cpu().numpy().astype(np.float32, copy=False)


def _save_wave_with_picks(
    *,
    wave_hw: np.ndarray,
    p_idx_view: np.ndarray,
    s_idx_view: np.ndarray,
    out_path: Path,
    title: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Save waveform image with overlaid P/S first-pick markers (in view coords)."""
    wave_hw = np.asarray(wave_hw, dtype=np.float32)
    if wave_hw.ndim != 2:
        msg = f'wave_hw must be 2D (H,W), got shape={wave_hw.shape}'
        raise ValueError(msg)

    p = np.asarray(p_idx_view, dtype=np.int64)
    s = np.asarray(s_idx_view, dtype=np.int64)
    H, _ = wave_hw.shape
    if p.shape != (H,) or s.shape != (H,):
        msg = f'p/s_idx_view must be (H,), got p={p.shape}, s={s.shape}, H={H}'
        raise ValueError(msg)

    fig, ax = plt.subplots(figsize=(12.0, 6.0), dpi=150)
    imshow_hw(
        ax,
        wave_hw,
        title=title,
        cmap='gray',
        vmin=vmin,
        vmax=vmax,
    )

    # imshow_hw transposes (H,W)->(W,H), so x-axis=trace index, y-axis=time sample
    xs = np.arange(H, dtype=np.int64)

    p_mask = p > 0
    if np.any(p_mask):
        ax.scatter(
            xs[p_mask],
            p[p_mask],
            s=10,
            marker='x',
            color='blue',
            label='P first (view)',
        )

    s_mask = s > 0
    if np.any(s_mask):
        ax.scatter(
            xs[s_mask],
            s[s_mask],
            s=10,
            marker='o',
            color='red',
            label='S first (view)',
        )

    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    # ====== Edit parameters ======
    segy_files = [
        '/home/dcuser/data/ridgecrest_das/event/20200623002546.sgy',
    ]
    phase_pick_files = [
        '/home/dcuser/data/ridgecrest_das/event/npz/20200623002546_phase_picks.npz',
    ]
    target_len = 6000
    subset_traces = 128
    include_empty_gathers = True
    use_header_cache = False
    seed = 0

    # where to save PNGs
    out_dir = Path('/tmp/phase_dataset_quick_check')
    # ============================

    out_dir.mkdir(parents=True, exist_ok=True)

    for p in segy_files:
        if not Path(p).exists():
            raise FileNotFoundError(f'SEGY not found: {p}')

    # Create an empty CSR pick file if missing (useful for smoke checks).
    pick_path = Path(phase_pick_files[0])
    if not pick_path.exists():
        pick_path.parent.mkdir(parents=True, exist_ok=True)
        with segyio.open(segy_files[0], 'r', ignore_geometry=True) as f:
            n_traces = int(f.tracecount)
        _write_empty_phase_picks_npz(pick_path, n_traces=n_traces)

    transform = ViewCompose([DeterministicCropOrPad(target_len), PerTraceStandardize()])
    fbgate = FirstBreakGate(
        FirstBreakGateConfig(
            apply_on='off',
            min_pick_ratio=0.0,
        )
    )
    plan = BuildPlan(
        wave_ops=[IdentitySignal(src='x_view', dst='x', copy=False)],
        label_ops=[PhasePSNMap(dst='psn_map', sigma=1.5)],
        input_stack=SelectStack(keys='x', dst='input'),
        target_stack=SelectStack(keys='psn_map', dst='target'),
    )

    ds = SegyGatherPhasePipelineDataset(
        segy_files=segy_files,
        phase_pick_files=phase_pick_files,
        transform=transform,
        fbgate=fbgate,
        plan=plan,
        subset_traces=int(subset_traces),
        include_empty_gathers=bool(include_empty_gathers),
        use_header_cache=bool(use_header_cache),
        secondary_key_fixed=True,
        verbose=False,
        primary_keys=('ffid',),
        max_trials=256,
    )
    ds._rng = np.random.default_rng(int(seed))

    try:
        sample = ds[0]
    finally:
        ds.close()

    print('=== Sample keys ===')
    print(sorted(sample.keys()))
    _describe('input', sample['input'])
    _describe('target', sample['target'])
    _describe('trace_valid', sample['trace_valid'])
    _describe('label_valid', sample['label_valid'])
    _describe('fb_idx', sample['fb_idx'])
    _describe('p_idx', sample['p_idx'])
    _describe('s_idx', sample['s_idx'])

    meta = sample['meta']
    print('=== Meta keys ===')
    print(sorted(meta.keys()))
    _describe("meta['fb_idx_view']", meta.get('fb_idx_view'))
    _describe("meta['p_idx_view']", meta.get('p_idx_view'))
    _describe("meta['s_idx_view']", meta.get('s_idx_view'))
    _describe("meta['time_view']", meta.get('time_view'))

    # ---- Visualization (waveform + pick markers) ----
    x_in = sample['input']
    y_tg = sample['target']

    wave_hw = _to_numpy_hw_from_input(x_in)
    y_chw = _to_numpy_chw_from_target(y_tg)

    p_idx_view = meta['p_idx_view']
    s_idx_view = meta['s_idx_view']

    png_wave = out_dir / 'wave_with_ps_p_srclocs.png'
    _save_wave_with_picks(
        wave_hw=wave_hw,
        p_idx_view=p_idx_view,
        s_idx_view=s_idx_view,
        out_path=png_wave,
        title='Input waveform (view) + P/S first picks (view coords)',
        vmin=-2.0,
        vmax=2.0,
    )
    print(f'[saved] {png_wave}')

    # ---- Visualization (row panels) using seisai_utils.viz.save_imshow_row ----
    png_panels = out_dir / 'input_and_target_panels.png'
    panels = [
        ImshowPanel(
            title='input (ch0)', data_hw=wave_hw, cmap='gray', vmin=-2.0, vmax=2.0
        ),
        ImshowPanel(title='target P', data_hw=y_chw[0], cmap=None, vmin=0.0, vmax=0.5),
        ImshowPanel(title='target S', data_hw=y_chw[1], cmap=None, vmin=0.0, vmax=0.5),
        ImshowPanel(
            title='target Noise', data_hw=y_chw[2], cmap=None, vmin=0.0, vmax=1.0
        ),
    ]
    save_imshow_row(
        png_panels,
        panels,
        suptitle='SegyGatherPhasePipelineDataset quick check',
        transpose_for_trace_time=True,
        figsize=(22.0, 6.0),
        dpi=150,
    )
    print(f'[saved] {png_panels}')

    # Optional: show a single-batch view via default DataLoader collation.
    ds2 = SegyGatherPhasePipelineDataset(
        segy_files=segy_files,
        phase_pick_files=phase_pick_files,
        transform=transform,
        fbgate=fbgate,
        plan=plan,
        subset_traces=int(subset_traces),
        include_empty_gathers=bool(include_empty_gathers),
        use_header_cache=bool(use_header_cache),
        secondary_key_fixed=True,
        verbose=False,
        max_trials=256,
    )
    ds2._rng = np.random.default_rng(int(seed))
    try:
        loader = DataLoader(ds2, batch_size=1, num_workers=0)
        batch = next(iter(loader))
    finally:
        ds2.close()

    print('=== Batch keys (batch_size=1) ===')
    print(sorted(batch.keys()))
    _describe('batch[input]', batch['input'])
    _describe('batch[target]', batch['target'])


if __name__ == '__main__':
    main()
