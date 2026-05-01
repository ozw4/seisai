from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from seisai_engine.pipelines.fbpick.coarse.eval import (
    CoarseCoverageEvalConfig,
    build_gap_neighborhood_mask,
    compute_confidence_bin_metrics,
    compute_summary_metrics,
    compute_trace_errors,
    load_coarse_prediction_npz,
    run_coarse_coverage_eval,
    run_eval_from_config,
)
from seisai_engine.pipelines.fbpick.coarse.trace_anchor import TraceSegment


def _write_coarse_npz(
    path: Path,
    *,
    coarse_pick_i: np.ndarray,
    dt_sec: float = 0.004,
    n_samples_orig: int = 512,
    coarse_pmax: np.ndarray | None = None,
    **extra,
) -> Path:
    pick = np.asarray(coarse_pick_i, dtype=np.int32)
    n_traces = int(pick.shape[0])
    if coarse_pmax is None:
        coarse_pmax = np.linspace(0.2, 0.9, n_traces, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        dt_sec=np.asarray(dt_sec, dtype=np.float32),
        n_samples_orig=np.asarray(n_samples_orig, dtype=np.int32),
        n_traces=np.asarray(n_traces, dtype=np.int32),
        ffid_values=np.full((n_traces,), 7, dtype=np.int32),
        chno_values=np.arange(1, n_traces + 1, dtype=np.int32),
        offsets_m=np.arange(n_traces, dtype=np.float32) * 10.0,
        trace_indices=np.arange(n_traces, dtype=np.int64),
        coarse_pick_i=pick,
        coarse_pick_t_sec=pick.astype(np.float32) * np.float32(dt_sec),
        coarse_pmax=np.asarray(coarse_pmax, dtype=np.float32),
        coarse_prob_summary=np.asarray(coarse_pmax, dtype=np.float32),
        lineage=np.asarray('{"stage":"coarse-test"}'),
        **extra,
    )
    return path


def test_perfect_prediction_has_full_coverage() -> None:
    errors = compute_trace_errors(
        coarse_pick_i=np.array([10, 20, 30], dtype=np.int32),
        fb_i=np.array([10, 20, 30], dtype=np.int32),
        dt_sec=0.004,
        n_samples_orig=128,
    )
    metrics = compute_summary_metrics(
        errors,
        coverage_thresholds_samples=(0,),
        coverage_thresholds_ms=(0.0,),
        fine_window_half_samples=0,
    )

    assert metrics['mae_samples'] == 0.0
    assert metrics['coverage_fine_window'] == 1.0
    assert metrics['coverage_0'] == 1.0


def test_coverage_threshold_works() -> None:
    errors = compute_trace_errors(
        coarse_pick_i=np.array([10, 20, 100], dtype=np.int32),
        fb_i=np.array([12, 25, 200], dtype=np.int32),
        dt_sec=0.004,
        n_samples_orig=256,
    )
    metrics = compute_summary_metrics(
        errors,
        coverage_thresholds_samples=(5,),
        coverage_thresholds_ms=(20.0,),
        fine_window_half_samples=5,
    )

    assert metrics['coverage_5'] == pytest.approx(2.0 / 3.0)
    assert metrics['coverage_fine_window'] == pytest.approx(2.0 / 3.0)


def test_ms_metrics_use_dt_sec() -> None:
    errors = compute_trace_errors(
        coarse_pick_i=np.array([20, 40], dtype=np.int32),
        fb_i=np.array([10, 20], dtype=np.int32),
        dt_sec=0.004,
        n_samples_orig=128,
    )

    assert errors.abs_error_ms.tolist() == [40.0, 80.0]
    metrics = compute_summary_metrics(errors, fine_window_half_samples=20)
    assert metrics['mae_ms'] == 60.0
    assert metrics['bias_ms'] == 60.0


def test_invalid_fb_labels_are_excluded() -> None:
    errors = compute_trace_errors(
        coarse_pick_i=np.array([10, 999, 40], dtype=np.int32),
        fb_i=np.array([10, -1, 30], dtype=np.float32),
        dt_sec=0.004,
        n_samples_orig=1000,
    )
    metrics = compute_summary_metrics(errors, fine_window_half_samples=10)

    assert errors.n_valid == 2
    assert errors.n_invalid == 1
    assert metrics['mae_samples'] == 5.0
    assert metrics['max_abs_samples'] == 10.0


def test_shape_mismatch_fails() -> None:
    with pytest.raises(ValueError, match='fb_i shape'):
        compute_trace_errors(
            coarse_pick_i=np.arange(10, dtype=np.int32),
            fb_i=np.arange(9, dtype=np.int32),
            dt_sec=0.004,
            n_samples_orig=128,
        )


def test_coarse_pick_t_sec_consistency_check(tmp_path: Path) -> None:
    path = _write_coarse_npz(
        tmp_path / 'bad.coarse.npz',
        coarse_pick_i=np.array([10], dtype=np.int32),
    )
    with np.load(path, allow_pickle=False) as z:
        payload = {key: z[key] for key in z.files}
    payload['coarse_pick_t_sec'] = np.array([123.0], dtype=np.float32)
    np.savez_compressed(path, **payload)

    with pytest.raises(ValueError, match='coarse_pick_t_sec'):
        load_coarse_prediction_npz(path)


def test_confidence_bin_metrics() -> None:
    errors = compute_trace_errors(
        coarse_pick_i=np.array([100, 40, 12], dtype=np.int32),
        fb_i=np.array([0, 20, 10], dtype=np.int32),
        dt_sec=0.004,
        n_samples_orig=256,
        coarse_pmax=np.array([0.1, 0.3, 0.9], dtype=np.float32),
    )
    rows = compute_confidence_bin_metrics(
        errors,
        confidence_bins=(0.0, 0.5, 1.0),
        fine_window_half_samples=20,
    )

    assert rows[0]['n_valid'] == 2
    assert rows[1]['n_valid'] == 1
    assert rows[0]['mae_samples'] > rows[1]['mae_samples']


def test_gap_neighborhood_mask_includes_boundary_adjacent_traces() -> None:
    mask = build_gap_neighborhood_mask(
        n_traces=20,
        segments=(
            TraceSegment(segment_id=0, start_pos=0, stop_pos=10, n_traces=10),
            TraceSegment(segment_id=1, start_pos=10, stop_pos=20, n_traces=10),
        ),
        gap_neighborhood_traces=2,
    )

    assert np.flatnonzero(mask).tolist() == [7, 8, 9, 10, 11, 12]


def test_run_coarse_coverage_eval_writes_reports_and_figures(tmp_path: Path) -> None:
    coarse_path = _write_coarse_npz(
        tmp_path / 'coarse' / 'gather.coarse.npz',
        coarse_pick_i=np.array([10, 20, 100], dtype=np.int32),
        coarse_pmax=np.array([0.9, 0.4, 0.2], dtype=np.float32),
        segment_ids=np.array([0, 1], dtype=np.int32),
        segment_start_pos=np.array([0, 2], dtype=np.int32),
        segment_stop_pos=np.array([2, 3], dtype=np.int32),
    )
    fb_path = tmp_path / 'fb.npy'
    np.save(fb_path, np.array([12, 25, 200], dtype=np.int32))

    report_paths = run_coarse_coverage_eval(
        coarse_files=[str(coarse_path)],
        fb_files=[str(fb_path)],
        out_dir=tmp_path / 'eval',
        eval_config=CoarseCoverageEvalConfig(
            fine_window_half_samples=5,
            coverage_thresholds_samples=(5, 100),
            coverage_thresholds_ms=(20.0,),
            gap_neighborhood_traces=1,
            confidence_bins=(0.0, 0.5, 1.0),
            make_figures=True,
        ),
    )

    assert report_paths['per_gather_csv'].is_file()
    assert report_paths['summary_csv'].is_file()
    assert report_paths['summary_json'].is_file()
    assert report_paths['confidence_bins_csv'].is_file()
    assert report_paths['gap_neighborhood_csv'].is_file()
    assert report_paths['per_segment_csv'].is_file()
    assert (tmp_path / 'eval' / 'figures' / 'error_hist_samples.png').is_file()
    assert (tmp_path / 'eval' / 'figures' / 'coverage_by_threshold.png').is_file()

    summary = json.loads(report_paths['summary_json'].read_text(encoding='utf-8'))
    assert summary['coverage_fine_window'] == pytest.approx(2.0 / 3.0)
    assert summary['n_valid_total'] == 3

    with report_paths['per_gather_csv'].open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert float(rows[0]['coverage_5']) == pytest.approx(2.0 / 3.0)


def test_run_eval_from_config_expands_relative_listfiles(tmp_path: Path) -> None:
    coarse_path = _write_coarse_npz(
        tmp_path / 'coarse' / 'gather.coarse.npz',
        coarse_pick_i=np.array([10, 20], dtype=np.int32),
    )
    fb_path = tmp_path / 'labels' / 'fb.npy'
    fb_path.parent.mkdir()
    np.save(fb_path, np.array([10, 30], dtype=np.int32))

    list_dir = tmp_path / 'lists'
    list_dir.mkdir()
    coarse_list = list_dir / 'coarse.txt'
    fb_list = list_dir / 'fb.txt'
    coarse_list.write_text(f'../coarse/{coarse_path.name}\n', encoding='utf-8')
    fb_list.write_text('../labels/fb.npy\n', encoding='utf-8')
    cfg_path = tmp_path / 'config_eval.yaml'
    cfg_path.write_text(
        '\n'.join(
            [
                'paths:',
                '  coarse_files: lists/coarse.txt',
                '  fb_files: lists/fb.txt',
                '  out_dir: eval_out',
                'eval:',
                '  fine_window_half_samples: 10',
                '  coverage_thresholds_samples: [10]',
                '  coverage_thresholds_ms: [40]',
                '  make_figures: false',
            ]
        )
        + '\n',
        encoding='utf-8',
    )

    report_paths = run_eval_from_config(cfg_path)

    assert report_paths['summary_json'] == (tmp_path / 'eval_out' / 'summary.json')
    summary = json.loads(report_paths['summary_json'].read_text(encoding='utf-8'))
    assert summary['coverage_fine_window'] == 1.0
