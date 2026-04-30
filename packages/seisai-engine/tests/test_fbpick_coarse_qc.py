from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import numpy as np

from seisai_engine.pipelines.fbpick.coarse import CoarseQCCfg, TraceSegment
from seisai_engine.pipelines.fbpick.coarse.qc import (
    plot_original_gather_qc,
    select_display_indices,
    write_global_anchor_coarse_qc,
)


def _qc_cfg(**overrides) -> CoarseQCCfg:
    values = {
        'enabled': True,
        'max_gathers': 16,
        'out_subdir': 'vis/coarse_global_anchor',
        'plot_anchor_grid': True,
        'plot_original_gather': True,
        'plot_confidence': True,
        'plot_error_if_labels_available': True,
        'fine_window_half_samples': 4,
        'max_display_traces': 64,
        'max_display_samples': 128,
        'low_confidence_threshold': 0.35,
        'dpi': 80,
        'clip_percentile': 99.0,
    }
    values.update(overrides)
    return CoarseQCCfg(**values)


def _assert_png(path: Path) -> None:
    assert path.is_file()
    assert path.stat().st_size > 0


def test_write_global_anchor_coarse_qc_writes_pngs_and_skips_missing_error(
    tmp_path: Path,
) -> None:
    input_waveform = np.linspace(-1.0, 1.0, 4 * 16, dtype=np.float32).reshape(4, 16)
    raw_wave = np.linspace(-1.0, 1.0, 8 * 32, dtype=np.float32).reshape(8, 32)
    segments = (
        TraceSegment(segment_id=0, start_pos=0, stop_pos=4, n_traces=4),
        TraceSegment(segment_id=1, start_pos=4, stop_pos=8, n_traces=4),
    )

    paths = write_global_anchor_coarse_qc(
        out_dir=tmp_path,
        gather_id='line A ffid=1',
        input_waveform_hw=input_waveform,
        anchor_pick_j=np.array([2, 4, 6, 999], dtype=np.int64),
        anchor_pmax=np.array([0.9, 0.8, 0.6, 1.0], dtype=np.float32),
        trace_valid=np.array([True, True, True, False], dtype=np.bool_),
        segment_id=np.array([0, 0, 1, -1], dtype=np.int64),
        segments=segments,
        coarse_pick_i=np.array([5, 6, 7, 8, 15, 16, 17, 18], dtype=np.int32),
        coarse_pmax=np.linspace(0.2, 0.9, 8, dtype=np.float32),
        raw_wave_hw=raw_wave,
        n_samples_orig=32,
        dt_sec=0.002,
        cfg=_qc_cfg(),
    )

    assert set(paths) == {'anchor_grid', 'original_gather', 'confidence'}
    for path in paths.values():
        _assert_png(path)
    assert not list(tmp_path.glob('*_error.png'))


def test_write_global_anchor_coarse_qc_writes_error_when_labels_exist(
    tmp_path: Path,
) -> None:
    paths = write_global_anchor_coarse_qc(
        out_dir=tmp_path,
        gather_id='ffid-2',
        input_waveform_hw=np.zeros((4, 16), dtype=np.float32),
        anchor_pick_j=np.array([2, 4, 6, 8], dtype=np.int64),
        anchor_pmax=np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32),
        trace_valid=np.ones((4,), dtype=np.bool_),
        segment_id=np.zeros((4,), dtype=np.int64),
        segments=(TraceSegment(segment_id=0, start_pos=0, stop_pos=8, n_traces=8),),
        coarse_pick_i=np.array([5, 6, 7, 8, 9, 10, 11, 12], dtype=np.int32),
        coarse_pmax=np.linspace(0.5, 0.9, 8, dtype=np.float32),
        fb_pick_i=np.array([4, 6, 8, 8, 10, 10, 12, 12], dtype=np.int32),
        n_samples_orig=32,
        dt_sec=0.002,
        cfg=_qc_cfg(plot_original_gather=False),
    )

    assert 'error' in paths
    _assert_png(paths['error'])


def test_original_gather_qc_accepts_decimated_display_image(tmp_path: Path) -> None:
    n_traces = 1000
    n_samples = 5000
    trace_positions = select_display_indices(n_traces, 64)
    sample_indices = select_display_indices(n_samples, 128)
    raw_wave = np.zeros((trace_positions.size, sample_indices.size), dtype=np.float32)
    coarse_pick_i = np.linspace(100, 2000, n_traces, dtype=np.float32).astype(np.int32)

    out_path = plot_original_gather_qc(
        tmp_path / 'decimated_original_gather.png',
        raw_wave_hw=raw_wave,
        coarse_pick_i=coarse_pick_i,
        coarse_pmax=np.linspace(0.2, 0.9, n_traces, dtype=np.float32),
        trace_positions=trace_positions,
        sample_indices=sample_indices,
        segments=(
            TraceSegment(segment_id=0, start_pos=0, stop_pos=500, n_traces=500),
            TraceSegment(segment_id=1, start_pos=500, stop_pos=1000, n_traces=500),
        ),
        dpi=80,
    )

    _assert_png(out_path)
