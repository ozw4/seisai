from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .config import CoarseQCCfg

__all__ = [
    'plot_anchor_grid_qc',
    'plot_confidence_qc',
    'plot_error_qc',
    'plot_original_gather_qc',
    'select_display_indices',
    'write_global_anchor_coarse_qc',
]

_SAFE_ID_PATTERN = re.compile(r'[^A-Za-z0-9_.-]+')


def _sanitize_id(value: str) -> str:
    token = _SAFE_ID_PATTERN.sub('-', str(value).strip()).strip('-')
    return token or 'gather'


def select_display_indices(length: int, max_count: int) -> np.ndarray:
    n = int(length)
    limit = int(max_count)
    if n <= 0:
        msg = 'length must be positive'
        raise ValueError(msg)
    if limit <= 0:
        msg = 'max_count must be positive'
        raise ValueError(msg)
    if n <= limit:
        return np.arange(n, dtype=np.int64)
    return np.unique(
        np.linspace(0, n - 1, limit, dtype=np.float64).round().astype(np.int64)
    )


def _as_2d_float(name: str, value: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    if arr.ndim != 2:
        msg = f'{name} must be 2D, got shape={arr.shape}'
        raise ValueError(msg)
    if arr.shape[0] <= 0 or arr.shape[1] <= 0:
        msg = f'{name} must be non-empty'
        raise ValueError(msg)
    return arr


def _as_vector(name: str, value: np.ndarray, *, length: int | None = None) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 1:
        msg = f'{name} must be 1D, got shape={arr.shape}'
        raise ValueError(msg)
    if length is not None and int(arr.shape[0]) != int(length):
        msg = f'{name} length {int(arr.shape[0])} != {int(length)}'
        raise ValueError(msg)
    return arr


def _resolve_clip(wave_hw: np.ndarray, *, clip_percentile: float) -> float:
    percentile = float(clip_percentile)
    if percentile <= 0.0 or percentile > 100.0:
        msg = 'clip_percentile must lie in (0, 100]'
        raise ValueError(msg)
    finite_abs = np.abs(np.asarray(wave_hw, dtype=np.float32))
    finite_abs = finite_abs[np.isfinite(finite_abs)]
    if finite_abs.size == 0:
        return 1.0
    clip_value = float(np.percentile(finite_abs, percentile))
    if np.isfinite(clip_value) and clip_value > 0.0:
        return clip_value
    max_value = float(np.max(finite_abs))
    if np.isfinite(max_value) and max_value > 0.0:
        return max_value
    return 1.0


def _coerce_segment_spans(segments: Sequence[object] | None) -> list[tuple[int, int]]:
    if segments is None:
        return []
    spans: list[tuple[int, int]] = []
    for segment in segments:
        if isinstance(segment, Mapping):
            start = int(segment['start_pos'])
            stop = int(segment['stop_pos'])
        else:
            start = int(getattr(segment, 'start_pos'))
            stop = int(getattr(segment, 'stop_pos'))
        if stop <= start:
            msg = f'invalid segment span: [{start}, {stop})'
            raise ValueError(msg)
        spans.append((start, stop))
    return spans


def _draw_segment_boundaries(ax, segments: Sequence[object] | None) -> None:
    spans = _coerce_segment_spans(segments)
    for start, _stop in spans[1:]:
        ax.axvline(float(start) - 0.5, color='orange', lw=0.9, ls='--', alpha=0.8)


def _mask_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=np.bool_)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(values.tolist()):
        if flag and start is None:
            start = int(idx)
        elif not flag and start is not None:
            runs.append((start, int(idx)))
            start = None
    if start is not None:
        runs.append((start, int(values.size)))
    return runs


def _save_and_close(fig, out_png: str | Path, *, dpi: int) -> Path:
    import matplotlib.pyplot as plt

    out_path = Path(out_png).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)
    return out_path


def plot_anchor_grid_qc(
    out_png: str | Path,
    *,
    input_waveform_hw: np.ndarray,
    anchor_pick_j: np.ndarray,
    anchor_pmax: np.ndarray,
    trace_valid: np.ndarray,
    segment_id: np.ndarray | None = None,
    title: str | None = None,
    dpi: int = 150,
    clip_percentile: float = 99.0,
) -> Path:
    import matplotlib.pyplot as plt

    wave = _as_2d_float('input_waveform_hw', input_waveform_hw)
    h, w = wave.shape
    pick = _as_vector('anchor_pick_j', anchor_pick_j, length=h).astype(np.float32)
    pmax = _as_vector('anchor_pmax', anchor_pmax, length=h).astype(np.float32)
    valid = _as_vector('trace_valid', trace_valid, length=h).astype(np.bool_)
    if segment_id is not None:
        seg_id = _as_vector('segment_id', segment_id, length=h).astype(np.int64)
    else:
        seg_id = None

    finite_pick = np.isfinite(pick)
    plot_mask = valid & finite_pick & (pick >= 0.0) & (pick < float(w))
    rows = np.arange(h, dtype=np.float32)
    clip_value = _resolve_clip(wave, clip_percentile=clip_percentile)

    fig, ax = plt.subplots(figsize=(12.0, 6.0))
    ax.imshow(
        wave,
        cmap='gray',
        aspect='auto',
        interpolation='nearest',
        origin='upper',
        vmin=-clip_value,
        vmax=clip_value,
    )
    for start, stop in _mask_runs(~valid):
        ax.axhspan(
            float(start) - 0.5,
            float(stop) - 0.5,
            color='red',
            alpha=0.12,
            lw=0.0,
        )

    if np.any(plot_mask):
        ax.plot(
            pick[plot_mask],
            rows[plot_mask],
            color='#00d7ff',
            lw=1.0,
            alpha=0.85,
            label='anchor_pick_j',
        )
        scatter = ax.scatter(
            pick[plot_mask],
            rows[plot_mask],
            c=np.clip(pmax[plot_mask], 0.0, 1.0),
            cmap='viridis',
            s=14.0,
            vmin=0.0,
            vmax=1.0,
            alpha=0.95,
            label='anchor_pmax',
            zorder=4,
        )
        fig.colorbar(scatter, ax=ax, fraction=0.025, pad=0.01, label='pmax')

    if seg_id is not None and np.any(valid):
        valid_rows = np.flatnonzero(valid)
        valid_seg = seg_id[valid_rows]
        change_at = np.flatnonzero(valid_seg[1:] != valid_seg[:-1]) + 1
        for change in change_at.tolist():
            row = int(valid_rows[int(change)])
            ax.axhline(float(row) - 0.5, color='orange', lw=0.9, ls='--', alpha=0.8)

    ax.set_xlim(-0.5, float(w) - 0.5)
    ax.set_ylim(float(h) - 0.5, -0.5)
    ax.set_xlabel('Coarse Time Index')
    ax.set_ylabel('Anchor Row')
    ax.set_title('anchor grid QC' if title is None else str(title))
    if np.any(plot_mask):
        ax.legend(loc='upper right', fontsize=8)
    return _save_and_close(fig, out_png, dpi=dpi)


def plot_original_gather_qc(
    out_png: str | Path,
    *,
    raw_wave_hw: np.ndarray,
    coarse_pick_i: np.ndarray,
    coarse_pmax: np.ndarray | None = None,
    fb_pick_i: np.ndarray | None = None,
    trace_positions: np.ndarray | None = None,
    sample_indices: np.ndarray | None = None,
    segments: Sequence[object] | None = None,
    title: str | None = None,
    dpi: int = 150,
    clip_percentile: float = 99.0,
) -> Path:
    import matplotlib.pyplot as plt

    wave = _as_2d_float('raw_wave_hw', raw_wave_hw)
    h_disp, w_disp = wave.shape
    pick = _as_vector('coarse_pick_i', coarse_pick_i).astype(np.float32)
    n_traces = int(pick.shape[0])
    if coarse_pmax is not None:
        pmax = _as_vector('coarse_pmax', coarse_pmax, length=n_traces).astype(
            np.float32
        )
    else:
        pmax = None
    if fb_pick_i is not None:
        fb = _as_vector('fb_pick_i', fb_pick_i, length=n_traces).astype(np.float32)
    else:
        fb = None

    if trace_positions is None:
        trace_pos = np.arange(h_disp, dtype=np.float32)
    else:
        trace_pos = _as_vector(
            'trace_positions',
            trace_positions,
            length=h_disp,
        ).astype(np.float32)
    if sample_indices is None:
        sample_pos = np.arange(w_disp, dtype=np.float32)
    else:
        sample_pos = _as_vector(
            'sample_indices',
            sample_indices,
            length=w_disp,
        ).astype(np.float32)
    if not np.all(np.diff(trace_pos) >= 0.0):
        msg = 'trace_positions must be sorted'
        raise ValueError(msg)
    if not np.all(np.diff(sample_pos) >= 0.0):
        msg = 'sample_indices must be sorted'
        raise ValueError(msg)

    clip_value = _resolve_clip(wave, clip_percentile=clip_percentile)
    fig_width = max(10.0, min(18.0, float(n_traces) / 32.0))
    fig, ax = plt.subplots(figsize=(fig_width, 7.0))
    ax.imshow(
        wave.T,
        cmap='gray',
        aspect='auto',
        interpolation='nearest',
        origin='upper',
        extent=(
            float(trace_pos[0]) - 0.5,
            float(trace_pos[-1]) + 0.5,
            float(sample_pos[-1]) + 0.5,
            float(sample_pos[0]) - 0.5,
        ),
        vmin=-clip_value,
        vmax=clip_value,
    )

    x = np.arange(n_traces, dtype=np.float32)
    finite_pick = np.isfinite(pick)
    ax.plot(
        x[finite_pick],
        pick[finite_pick],
        color='#00d7ff',
        lw=1.0,
        alpha=0.9,
        label='coarse_pick_i',
    )
    if pmax is not None and np.any(finite_pick):
        scatter = ax.scatter(
            x[finite_pick],
            pick[finite_pick],
            c=np.clip(pmax[finite_pick], 0.0, 1.0),
            cmap='viridis',
            s=12.0,
            vmin=0.0,
            vmax=1.0,
            alpha=0.9,
            label='coarse_pmax',
            zorder=4,
        )
        fig.colorbar(scatter, ax=ax, fraction=0.025, pad=0.01, label='pmax')
    if fb is not None:
        valid_fb = np.isfinite(fb) & (fb >= 0.0)
        ax.plot(
            x[valid_fb],
            fb[valid_fb],
            color='yellow',
            lw=1.0,
            alpha=0.9,
            label='ground truth',
        )

    _draw_segment_boundaries(ax, segments)
    ax.set_xlim(-0.5, float(max(n_traces - 1, 0)) + 0.5)
    ax.set_ylim(float(sample_pos[-1]) + 0.5, float(sample_pos[0]) - 0.5)
    ax.set_xlabel('Trace Position')
    ax.set_ylabel('Sample Index')
    ax.set_title('original gather QC' if title is None else str(title))
    ax.legend(loc='upper right', fontsize=8)
    return _save_and_close(fig, out_png, dpi=dpi)


def plot_confidence_qc(
    out_png: str | Path,
    *,
    coarse_pmax: np.ndarray,
    segments: Sequence[object] | None = None,
    low_confidence_threshold: float | None = None,
    title: str | None = None,
    dpi: int = 150,
) -> Path:
    import matplotlib.pyplot as plt

    pmax = _as_vector('coarse_pmax', coarse_pmax).astype(np.float32)
    n_traces = int(pmax.shape[0])
    x = np.arange(n_traces, dtype=np.float32)

    fig_width = max(8.0, min(18.0, float(n_traces) / 40.0))
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))
    ax.plot(x, pmax, color='#00d7ff', lw=1.1, label='coarse_pmax')
    if low_confidence_threshold is not None:
        threshold = float(low_confidence_threshold)
        if threshold < 0.0 or threshold > 1.0:
            msg = 'low_confidence_threshold must lie in [0, 1]'
            raise ValueError(msg)
        ax.axhline(
            threshold,
            color='red',
            lw=0.9,
            ls='--',
            alpha=0.75,
            label='low confidence threshold',
        )
    _draw_segment_boundaries(ax, segments)
    ax.set_xlim(-0.5, float(max(n_traces - 1, 0)) + 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('Trace Position')
    ax.set_ylabel('Confidence')
    ax.set_title('coarse confidence QC' if title is None else str(title))
    ax.grid(True, alpha=0.25)
    ax.legend(loc='lower right', fontsize=8)
    return _save_and_close(fig, out_png, dpi=dpi)


def _error_summary(
    err_samples: np.ndarray,
    *,
    dt_sec: float | None,
    fine_window_half_samples: int,
) -> str:
    abs_err = np.abs(err_samples.astype(np.float64, copy=False))
    mae = float(np.mean(abs_err))
    p50, p90, p95 = np.percentile(abs_err, [50.0, 90.0, 95.0]).tolist()
    coverage = float(np.mean(abs_err <= float(fine_window_half_samples)) * 100.0)
    parts = [
        f'MAE={mae:.2f} samples',
        f'P50={p50:.2f}',
        f'P90={p90:.2f}',
        f'P95={p95:.2f}',
        f'within +/-{int(fine_window_half_samples)}={coverage:.1f}%',
    ]
    if dt_sec is not None:
        mae_ms = mae * float(dt_sec) * 1000.0
        parts.insert(1, f'MAE={mae_ms:.2f} ms')
    return ' | '.join(parts)


def plot_error_qc(
    out_png: str | Path,
    *,
    coarse_pick_i: np.ndarray,
    fb_pick_i: np.ndarray,
    n_samples_orig: int | None = None,
    dt_sec: float | None = None,
    fine_window_half_samples: int = 128,
    segments: Sequence[object] | None = None,
    title: str | None = None,
    dpi: int = 150,
) -> Path:
    import matplotlib.pyplot as plt

    coarse = _as_vector('coarse_pick_i', coarse_pick_i).astype(np.float32)
    fb = _as_vector('fb_pick_i', fb_pick_i, length=int(coarse.shape[0])).astype(
        np.float32
    )
    n_traces = int(coarse.shape[0])
    valid = np.isfinite(coarse) & np.isfinite(fb) & (fb >= 0.0)
    if n_samples_orig is not None:
        valid &= fb < float(n_samples_orig)
    err = coarse - fb
    err[~valid] = np.nan

    x = np.arange(n_traces, dtype=np.float32)
    fig_width = max(8.0, min(18.0, float(n_traces) / 40.0))
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    ax.plot(x, err, color='#00d7ff', lw=1.0, label='coarse - ground truth')
    ax.axhline(0.0, color='black', lw=0.9, alpha=0.75)
    half = int(fine_window_half_samples)
    if half > 0:
        ax.axhline(float(half), color='red', lw=0.8, ls='--', alpha=0.65)
        ax.axhline(float(-half), color='red', lw=0.8, ls='--', alpha=0.65)
    _draw_segment_boundaries(ax, segments)
    if np.any(valid):
        summary = _error_summary(
            err[valid],
            dt_sec=dt_sec,
            fine_window_half_samples=half,
        )
        title_text = summary if title is None else f'{title}\n{summary}'
    else:
        ax.text(
            0.5,
            0.5,
            'no valid ground-truth picks',
            ha='center',
            va='center',
            transform=ax.transAxes,
        )
        title_text = 'coarse error QC' if title is None else str(title)

    ax.set_xlim(-0.5, float(max(n_traces - 1, 0)) + 0.5)
    ax.set_xlabel('Trace Position')
    ax.set_ylabel('Sample Error')
    ax.set_title(title_text)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper right', fontsize=8)
    return _save_and_close(fig, out_png, dpi=dpi)


def write_global_anchor_coarse_qc(
    *,
    out_dir: str | Path,
    gather_id: str,
    input_waveform_hw: np.ndarray,
    anchor_pick_j: np.ndarray,
    anchor_pmax: np.ndarray,
    trace_valid: np.ndarray,
    segment_id: np.ndarray,
    segments: Sequence[object] | None,
    coarse_pick_i: np.ndarray,
    coarse_pmax: np.ndarray,
    raw_wave_hw: np.ndarray | None = None,
    raw_trace_positions: np.ndarray | None = None,
    raw_sample_indices: np.ndarray | None = None,
    fb_pick_i: np.ndarray | None = None,
    n_samples_orig: int | None = None,
    dt_sec: float | None = None,
    cfg: CoarseQCCfg,
) -> dict[str, Path]:
    if not cfg.enabled:
        return {}

    root = Path(out_dir).expanduser().resolve()
    safe_id = _sanitize_id(gather_id)
    title = safe_id.replace('-', ' ')
    paths: dict[str, Path] = {}

    if cfg.plot_anchor_grid:
        paths['anchor_grid'] = plot_anchor_grid_qc(
            root / f'{safe_id}_anchor_grid.png',
            input_waveform_hw=input_waveform_hw,
            anchor_pick_j=anchor_pick_j,
            anchor_pmax=anchor_pmax,
            trace_valid=trace_valid,
            segment_id=segment_id,
            title=title,
            dpi=cfg.dpi,
            clip_percentile=cfg.clip_percentile,
        )

    if cfg.plot_original_gather and raw_wave_hw is not None:
        paths['original_gather'] = plot_original_gather_qc(
            root / f'{safe_id}_original_gather.png',
            raw_wave_hw=raw_wave_hw,
            coarse_pick_i=coarse_pick_i,
            coarse_pmax=coarse_pmax,
            fb_pick_i=fb_pick_i,
            trace_positions=raw_trace_positions,
            sample_indices=raw_sample_indices,
            segments=segments,
            title=title,
            dpi=cfg.dpi,
            clip_percentile=cfg.clip_percentile,
        )

    if cfg.plot_confidence:
        paths['confidence'] = plot_confidence_qc(
            root / f'{safe_id}_confidence.png',
            coarse_pmax=coarse_pmax,
            segments=segments,
            low_confidence_threshold=cfg.low_confidence_threshold,
            title=title,
            dpi=cfg.dpi,
        )

    if cfg.plot_error_if_labels_available and fb_pick_i is not None:
        paths['error'] = plot_error_qc(
            root / f'{safe_id}_error.png',
            coarse_pick_i=coarse_pick_i,
            fb_pick_i=fb_pick_i,
            n_samples_orig=n_samples_orig,
            dt_sec=dt_sec,
            fine_window_half_samples=cfg.fine_window_half_samples,
            segments=segments,
            title=title,
            dpi=cfg.dpi,
        )

    return paths
