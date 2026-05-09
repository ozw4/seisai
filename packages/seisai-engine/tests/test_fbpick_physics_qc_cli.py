from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import cli.run_fbpick_physics_qc as physics_qc_cli


def _ffid_info(key_to_indices: dict[int, list[int] | np.ndarray]) -> dict[str, object]:
    max_index = max(
        int(np.max(np.asarray(indices, dtype=np.int64)))
        for indices in key_to_indices.values()
        if int(np.asarray(indices).size) > 0
    )
    return {
        'ffid_key_to_indices': {
            key: np.asarray(indices, dtype=np.int64)
            for key, indices in key_to_indices.items()
        },
        'chno_values': np.arange(max_index + 1, dtype=np.int64),
    }


def _qc_common_payload(n_traces: int = 3) -> dict[str, np.ndarray]:
    return {
        'dt_sec': np.asarray(0.004, dtype=np.float32),
        'n_samples_orig': np.asarray(512, dtype=np.int32),
        'n_traces': np.asarray(n_traces, dtype=np.int32),
        'ffid_values': np.full(n_traces, 10, dtype=np.int32),
        'chno_values': np.arange(1, n_traces + 1, dtype=np.int32),
        'offsets_m': np.arange(n_traces, dtype=np.float32) * np.float32(100.0),
        'trace_indices': np.arange(n_traces, dtype=np.int64),
    }


def _qc_coarse_payload(n_traces: int = 3) -> dict[str, np.ndarray]:
    pick_i = np.arange(100, 100 + n_traces, dtype=np.int32)
    return {
        **_qc_common_payload(n_traces),
        'coarse_pick_i': pick_i,
        'coarse_pick_t_sec': pick_i.astype(np.float32) * np.float32(0.004),
        'coarse_pmax': np.linspace(0.7, 0.9, n_traces, dtype=np.float32),
        'coarse_prob_summary': np.linspace(0.7, 0.9, n_traces, dtype=np.float32),
    }


def _qc_robust_payload(n_traces: int = 3) -> dict[str, np.ndarray]:
    pick_i = np.arange(101, 101 + n_traces, dtype=np.int32)
    return {
        **_qc_common_payload(n_traces),
        'robust_pick_i': pick_i,
        'robust_pick_t_sec': pick_i.astype(np.float32) * np.float32(0.004),
        'robust_conf': np.linspace(0.6, 0.8, n_traces, dtype=np.float32),
        'robust_source': np.zeros(n_traces, dtype=np.uint8),
        'used_theoretical_mask': np.zeros(n_traces, dtype=np.bool_),
        'reason_mask': np.zeros(n_traces, dtype=np.uint8),
    }


def _qc_final_payload(n_traces: int = 3) -> dict[str, np.ndarray]:
    window_start_i = np.arange(50, 50 + n_traces, dtype=np.int32)
    final_pick_i = np.arange(111, 111 + n_traces, dtype=np.int32)
    return {
        **_qc_common_payload(n_traces),
        'window_start_i': window_start_i,
        'window_end_i': (window_start_i + 255).astype(np.int32),
        'final_pick_i': final_pick_i,
        'final_pick_f': final_pick_i.astype(np.float32),
        'final_conf': np.linspace(0.5, 0.7, n_traces, dtype=np.float32),
    }


def _qc_info(n_traces: int = 3) -> dict[str, object]:
    common = _qc_common_payload(n_traces)
    return {
        'n_traces': n_traces,
        'n_samples': 512,
        'dt_sec': 0.004,
        'ffid_values': common['ffid_values'],
        'chno_values': common['chno_values'],
        'offsets': common['offsets_m'],
    }


def _physics_qc_runtime(
    *,
    cfg: dict[str, object],
    base_dir: Path,
    coarse: dict[str, np.ndarray] | None = None,
    robust: dict[str, np.ndarray] | None = None,
    final: dict[str, np.ndarray] | None = None,
    final_error: Exception | None = None,
    loaded_final_paths: list[Path] | None = None,
) -> SimpleNamespace:
    def _load_final(path: str | Path) -> dict[str, np.ndarray]:
        if loaded_final_paths is not None:
            loaded_final_paths.append(Path(path))
        if final_error is not None:
            raise final_error
        if final is None:
            raise AssertionError('final npz should not be loaded')
        return final

    return SimpleNamespace(
        load_cfg_with_base_dir=lambda path: (cfg, base_dir),
        expand_cfg_listfiles=lambda cfg, *, keys: cfg,
        resolve_cfg_paths=lambda cfg, base_dir, *, keys: cfg,
        build_file_info=lambda *args, **kwargs: _qc_info(),
        segyio=SimpleNamespace(
            TraceField=SimpleNamespace(FieldRecord=1, TraceNumber=2, CDP=3)
        ),
        load_coarse_npz=lambda path: coarse or _qc_coarse_payload(),
        load_robust_npz=lambda path: robust or _qc_robust_payload(),
        load_fbpick_final_npz=_load_final,
        save_fbpick_physics_qc_cdf_png=lambda *args, **kwargs: Path(args[0]),
        save_fbpick_physics_qc_gather_png=lambda *args, **kwargs: Path(args[0]),
    )


def _write_fb(path: Path, n_traces: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.arange(100, 100 + n_traces, dtype=np.int64))


def test_load_vis_cfg_parses_skip_keys_and_max_traces() -> None:
    vis_cfg = physics_qc_cli._load_vis_cfg(
        {
            'vis': {
                'skip_gather_keys': {'ffid': [0], 'cmp': [-1, 2]},
                'max_traces_per_gather': 10000,
                'waveform_norm': 'per_trace',
                'clip_percentile': 99.0,
            }
        }
    )

    assert vis_cfg['skip_gather_keys'] == {'ffid': {0}, 'cmp': {-1, 2}}
    assert vis_cfg['max_traces_per_gather'] == 10000
    assert vis_cfg['waveform_norm'] == 'per_trace'
    assert vis_cfg['clip_percentile'] == 99.0


def test_load_vis_cfg_allows_null_max_traces() -> None:
    vis_cfg = physics_qc_cli._load_vis_cfg(
        {'vis': {'max_traces_per_gather': None}}
    )

    assert vis_cfg['max_traces_per_gather'] is None


@pytest.mark.parametrize(
    'vis',
    [
        {'skip_gather_keys': None},
        {'skip_gather_keys': {'ffid': [False]}},
        {'skip_gather_keys': {'ffid': (0,)}},
        {'skip_gather_keys': {1: [0]}},
        {'max_traces_per_gather': 0},
        {'max_traces_per_gather': True},
        {'waveform_norm': 'invalid'},
        {'waveform_norm': 1},
        {'clip_percentile': 0.0},
        {'clip_percentile': 100.1},
        {'clip_percentile': True},
        {'clip_percentile': '99.0'},
    ],
)
def test_load_vis_cfg_rejects_invalid_new_fields(vis: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        physics_qc_cli._load_vis_cfg({'vis': vis})


def test_iter_vis_gathers_skips_configured_key() -> None:
    info = _ffid_info({0: [0, 1], 1: [2], 2: [3]})

    yielded = list(
        physics_qc_cli._iter_vis_gathers(
            info,
            primary_keys=['ffid'],
            max_gathers=2,
            skip_gather_keys={'ffid': {0}},
            max_traces_per_gather=None,
            segy_path='line.sgy',
        )
    )

    assert [gather_key for _, gather_key, _ in yielded] == [1, 2]


def test_iter_vis_gathers_counts_accepted_gathers_not_skipped() -> None:
    info = _ffid_info({0: [0], 1: [1], 2: [2], 3: [3]})

    yielded = list(
        physics_qc_cli._iter_vis_gathers(
            info,
            primary_keys=['ffid'],
            max_gathers=2,
            skip_gather_keys={'ffid': {0}},
            max_traces_per_gather=None,
            segy_path='line.sgy',
        )
    )

    assert [gather_key for _, gather_key, _ in yielded] == [1, 2]


def test_iter_vis_gathers_skips_oversized_gather(
    capsys: pytest.CaptureFixture[str],
) -> None:
    info = _ffid_info({1: np.arange(5), 2: np.arange(5, 7)})

    yielded = list(
        physics_qc_cli._iter_vis_gathers(
            info,
            primary_keys=['ffid'],
            max_gathers=1,
            skip_gather_keys={},
            max_traces_per_gather=3,
            segy_path='/data/line.sgy',
        )
    )

    assert [gather_key for _, gather_key, _ in yielded] == [2]
    captured = capsys.readouterr()
    assert (
        'skip oversized gather: file=/data/line.sgy '
        'primary=ffid key=1 traces=5 limit=3'
    ) in captured.out


def test_save_vis_pngs_skips_oversized_before_mmap_access(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FailingMmap:
        def __getitem__(self, index: int) -> np.ndarray:
            raise AssertionError(f'mmap should not be read for skipped gather {index}')

    def _unexpected_save(*args, **kwargs) -> Path:
        raise AssertionError('PNG writer should not be called')

    segy_path = tmp_path / 'line' / 'huge.sgy'
    segy_path.parent.mkdir()
    info = _ffid_info({1: np.arange(5)})
    info['mmap'] = FailingMmap()
    runtime = SimpleNamespace(save_fbpick_physics_qc_gather_png=_unexpected_save)

    out_paths = physics_qc_cli._save_vis_pngs(
        info=info,
        segy_path=str(segy_path),
        out_dir=tmp_path / 'out',
        gt_pick_i=np.zeros(5, dtype=np.int64),
        coarse_pick_i=np.zeros(5, dtype=np.int64),
        robust_pick_i=np.zeros(5, dtype=np.int64),
        dataset_cfg={'primary_keys': ['ffid']},
        vis_cfg={
            'max_gathers_per_file': 1,
            'skip_gather_keys': {},
            'max_traces_per_gather': 3,
        },
        runtime=runtime,
    )

    assert out_paths == []
    captured = capsys.readouterr()
    assert 'skip oversized gather:' in captured.out
    assert f'No gather PNGs written for {segy_path}: all candidates were skipped' in (
        captured.out
    )


def test_save_vis_pngs_passes_waveform_display_config(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _save(out_png: Path, **kwargs: object) -> Path:
        captured.update(kwargs)
        return out_png

    segy_path = tmp_path / 'line' / 'small.sgy'
    segy_path.parent.mkdir()
    info = _ffid_info({1: [0, 1]})
    info['mmap'] = [
        np.asarray([0.0, 1.0, -1.0], dtype=np.float32),
        np.asarray([0.0, 1000.0, -1000.0], dtype=np.float32),
    ]
    runtime = SimpleNamespace(save_fbpick_physics_qc_gather_png=_save)

    out_paths = physics_qc_cli._save_vis_pngs(
        info=info,
        segy_path=str(segy_path),
        out_dir=tmp_path / 'out',
        gt_pick_i=np.asarray([1, 1], dtype=np.int64),
        coarse_pick_i=np.asarray([1, 1], dtype=np.int64),
        robust_pick_i=np.asarray([1, 1], dtype=np.int64),
        dataset_cfg={'primary_keys': ['ffid']},
        vis_cfg={
            'max_gathers_per_file': 1,
            'skip_gather_keys': {},
            'max_traces_per_gather': None,
            'waveform_norm': 'per_trace',
            'clip_percentile': 98.5,
        },
        runtime=runtime,
    )

    assert len(out_paths) == 1
    assert captured['waveform_norm'] == 'per_trace'
    assert captured['clip_percentile'] == 98.5


def test_save_vis_pngs_passes_optional_physical_overlay_arrays(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _save(out_png: Path, **kwargs: object) -> Path:
        captured.update(kwargs)
        return out_png

    segy_path = tmp_path / 'line' / 'physical.sgy'
    segy_path.parent.mkdir()
    info = _ffid_info({1: [2, 0]})
    info['mmap'] = [
        np.asarray([0.0, 1.0, -1.0], dtype=np.float32),
        np.asarray([0.0, 2.0, -2.0], dtype=np.float32),
        np.asarray([0.0, 3.0, -3.0], dtype=np.float32),
    ]
    runtime = SimpleNamespace(save_fbpick_physics_qc_gather_png=_save)

    out_paths = physics_qc_cli._save_vis_pngs(
        info=info,
        segy_path=str(segy_path),
        out_dir=tmp_path / 'out',
        gt_pick_i=np.asarray([1, 1, 1], dtype=np.int64),
        coarse_pick_i=np.asarray([10, 20, 30], dtype=np.int64),
        robust_pick_i=np.asarray([11, 21, 31], dtype=np.int64),
        coarse_pmax=np.asarray([0.2, 0.4, 0.6], dtype=np.float32),
        trend_center_i=np.asarray([12, 22, 32], dtype=np.int32),
        physical_center_i=np.asarray([13, 23, 33], dtype=np.int32),
        fine_center_i=np.asarray([14, 24, 34], dtype=np.int32),
        physical_model_status=np.asarray([0, 1, 2], dtype=np.uint8),
        dataset_cfg={'primary_keys': ['ffid']},
        vis_cfg={
            'max_gathers_per_file': 1,
            'skip_gather_keys': {},
            'max_traces_per_gather': None,
            'waveform_norm': 'global',
            'clip_percentile': 99.0,
        },
        runtime=runtime,
    )

    assert len(out_paths) == 1
    np.testing.assert_array_equal(
        captured['coarse_pmax'], np.asarray([0.2, 0.6], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        captured['trend_center_i'], np.asarray([12, 32], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        captured['physical_center_i'],
        np.asarray([13, 33], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        captured['fine_center_i'], np.asarray([14, 34], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        captured['physical_model_status'],
        np.asarray([0, 2], dtype=np.uint8),
    )


def test_run_pipeline_without_final_artifact_passes_none_to_vis(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fb_path = tmp_path / 'fb.npy'
    _write_fb(fb_path)
    cfg = {
        'paths': {
            'segy_files': [str(tmp_path / 'line' / 'survey.sgy')],
            'fb_files': [str(fb_path)],
            'coarse_npz_dir': str(tmp_path / 'coarse'),
            'robust_npz_dir': str(tmp_path / 'robust'),
            'out_dir': str(tmp_path / 'out'),
        },
        'dataset': {'primary_keys': ['ffid']},
        'vis': {'save_summary_csv': False, 'max_gathers_per_file': 0},
    }
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        physics_qc_cli,
        '_load_runtime',
        lambda: _physics_qc_runtime(cfg=cfg, base_dir=tmp_path),
    )
    monkeypatch.setattr(
        physics_qc_cli,
        '_save_vis_pngs',
        lambda **kwargs: captured.update(kwargs) or [],
    )

    physics_qc_cli.run_pipeline(tmp_path / 'config.yaml')

    assert 'window_start_i' in captured
    assert captured['window_start_i'] is None
    assert captured['window_end_i'] is None
    assert captured['final_pick_i'] is None


def test_run_pipeline_final_npz_dir_passes_actual_arrays_to_vis(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fb_path = tmp_path / 'fb.npy'
    segy_path = tmp_path / 'line' / 'survey.sgy'
    _write_fb(fb_path)
    final = _qc_final_payload()
    loaded_final_paths: list[Path] = []
    cfg = {
        'paths': {
            'segy_files': [str(segy_path)],
            'fb_files': [str(fb_path)],
            'coarse_npz_dir': str(tmp_path / 'coarse'),
            'robust_npz_dir': str(tmp_path / 'robust'),
            'final_npz_dir': str(tmp_path / 'final'),
            'out_dir': str(tmp_path / 'out'),
        },
        'dataset': {'primary_keys': ['ffid']},
        'vis': {'save_summary_csv': False, 'max_gathers_per_file': 0},
    }
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        physics_qc_cli,
        '_load_runtime',
        lambda: _physics_qc_runtime(
            cfg=cfg,
            base_dir=tmp_path,
            final=final,
            loaded_final_paths=loaded_final_paths,
        ),
    )
    monkeypatch.setattr(
        physics_qc_cli,
        '_save_vis_pngs',
        lambda **kwargs: captured.update(kwargs) or [],
    )

    physics_qc_cli.run_pipeline(tmp_path / 'config.yaml')

    assert loaded_final_paths == [
        tmp_path / 'final' / 'line__survey.fbpick_final.npz'
    ]
    np.testing.assert_array_equal(captured['window_start_i'], final['window_start_i'])
    np.testing.assert_array_equal(captured['window_end_i'], final['window_end_i'])
    np.testing.assert_array_equal(captured['final_pick_i'], final['final_pick_i'])


def test_run_pipeline_summary_csv_includes_fine_physical_and_final_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fb_path = tmp_path / 'fb.npy'
    segy_path = tmp_path / 'line' / 'survey.sgy'
    _write_fb(fb_path)
    robust = _qc_robust_payload()
    robust['fine_center_i'] = np.asarray([100, 101, 102], dtype=np.int32)
    robust['physical_center_i'] = np.asarray([100, 101, 102], dtype=np.int32)
    final = _qc_final_payload()
    cfg = {
        'paths': {
            'segy_files': [str(segy_path)],
            'fb_files': [str(fb_path)],
            'coarse_npz_dir': str(tmp_path / 'coarse'),
            'robust_npz_dir': str(tmp_path / 'robust'),
            'final_npz_files': [str(tmp_path / 'final.npz')],
            'out_dir': str(tmp_path / 'out'),
        },
        'dataset': {'primary_keys': ['ffid']},
        'vis': {'save_summary_csv': True, 'max_gathers_per_file': 0},
    }

    monkeypatch.setattr(
        physics_qc_cli,
        '_load_runtime',
        lambda: _physics_qc_runtime(
            cfg=cfg,
            base_dir=tmp_path,
            robust=robust,
            final=final,
        ),
    )

    physics_qc_cli.run_pipeline(tmp_path / 'config.yaml')

    with (tmp_path / 'out' / 'summary_per_file.csv').open(
        newline='',
        encoding='utf-8',
    ) as f:
        rows = list(csv.DictReader(f))

    assert rows[0]['fine_center_R32'] == '1'
    assert rows[0]['fine_center_delta_p90_vs_robust'] == '-1'
    assert rows[0]['physical_center_R127'] == '1'
    assert rows[0]['gt_in_actual_window_rate'] == '1'
    assert rows[0]['final_pick_R127'] == '1'


def test_run_pipeline_final_npz_files_take_priority_over_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fb_path = tmp_path / 'fb.npy'
    explicit_final_path = tmp_path / 'explicit' / 'survey.fbpick_final.npz'
    _write_fb(fb_path)
    loaded_final_paths: list[Path] = []
    cfg = {
        'paths': {
            'segy_files': [str(tmp_path / 'line' / 'survey.sgy')],
            'fb_files': [str(fb_path)],
            'coarse_npz_dir': str(tmp_path / 'coarse'),
            'robust_npz_dir': str(tmp_path / 'robust'),
            'final_npz_dir': str(tmp_path / 'final'),
            'final_npz_files': [str(explicit_final_path)],
            'out_dir': str(tmp_path / 'out'),
        },
        'dataset': {'primary_keys': ['ffid']},
        'vis': {'save_summary_csv': False, 'max_gathers_per_file': 0},
    }

    monkeypatch.setattr(
        physics_qc_cli,
        '_load_runtime',
        lambda: _physics_qc_runtime(
            cfg=cfg,
            base_dir=tmp_path,
            final=_qc_final_payload(),
            loaded_final_paths=loaded_final_paths,
        ),
    )
    monkeypatch.setattr(physics_qc_cli, '_save_vis_pngs', lambda **kwargs: [])

    physics_qc_cli.run_pipeline(tmp_path / 'config.yaml')

    assert loaded_final_paths == [explicit_final_path]


def test_run_pipeline_final_alignment_mismatch_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fb_path = tmp_path / 'fb.npy'
    _write_fb(fb_path)
    final = _qc_final_payload()
    final['trace_indices'] = np.asarray([0, 2, 1], dtype=np.int64)
    cfg = {
        'paths': {
            'segy_files': [str(tmp_path / 'line' / 'survey.sgy')],
            'fb_files': [str(fb_path)],
            'coarse_npz_dir': str(tmp_path / 'coarse'),
            'robust_npz_dir': str(tmp_path / 'robust'),
            'final_npz_files': [str(tmp_path / 'final.npz')],
            'out_dir': str(tmp_path / 'out'),
        },
        'dataset': {'primary_keys': ['ffid']},
        'vis': {'save_summary_csv': False, 'max_gathers_per_file': 0},
    }

    monkeypatch.setattr(
        physics_qc_cli,
        '_load_runtime',
        lambda: _physics_qc_runtime(cfg=cfg, base_dir=tmp_path, final=final),
    )
    monkeypatch.setattr(physics_qc_cli, '_save_vis_pngs', lambda **kwargs: [])

    with pytest.raises(ValueError, match='final trace_indices'):
        physics_qc_cli.run_pipeline(tmp_path / 'config.yaml')


def test_run_pipeline_final_path_specified_missing_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fb_path = tmp_path / 'fb.npy'
    missing_final_path = tmp_path / 'missing.fbpick_final.npz'
    _write_fb(fb_path)
    cfg = {
        'paths': {
            'segy_files': [str(tmp_path / 'line' / 'survey.sgy')],
            'fb_files': [str(fb_path)],
            'coarse_npz_dir': str(tmp_path / 'coarse'),
            'robust_npz_dir': str(tmp_path / 'robust'),
            'final_npz_files': [str(missing_final_path)],
            'out_dir': str(tmp_path / 'out'),
        },
        'dataset': {'primary_keys': ['ffid']},
        'vis': {'save_summary_csv': False, 'max_gathers_per_file': 0},
    }

    monkeypatch.setattr(
        physics_qc_cli,
        '_load_runtime',
        lambda: _physics_qc_runtime(
            cfg=cfg,
            base_dir=tmp_path,
            final_error=FileNotFoundError(missing_final_path),
        ),
    )
    monkeypatch.setattr(physics_qc_cli, '_save_vis_pngs', lambda **kwargs: [])

    with pytest.raises(FileNotFoundError):
        physics_qc_cli.run_pipeline(tmp_path / 'config.yaml')


def test_qc_summary_columns_include_physical_failure_reason_counts() -> None:
    assert 'physical_model_status_counts' in physics_qc_cli.PER_FILE_COLUMNS
    assert 'physical_model_failure_reason_counts' in physics_qc_cli.PER_FILE_COLUMNS
    assert 'physical_model_status_counts' in physics_qc_cli.GLOBAL_COLUMNS
    assert 'physical_model_failure_reason_counts' in physics_qc_cli.GLOBAL_COLUMNS


def test_format_uint8_counts_uses_physical_failure_labels() -> None:
    summary = physics_qc_cli._format_uint8_counts(
        np.asarray([0, 2, 2, 5], dtype=np.uint8),
        labels=physics_qc_cli.PHYSICAL_MODEL_FAILURE_LABELS,
    )

    assert summary == 'none=1; geometry_invalid=2; prediction_invalid=1'


def test_summarize_errors_reports_fine_center_improvement() -> None:
    gt_pick_i = np.asarray([100, 200, 300, 400], dtype=np.int64)

    metrics, *_ = physics_qc_cli._summarize_errors(
        coarse_pick_i=gt_pick_i + 40,
        robust_pick_i=gt_pick_i + 20,
        fine_center_i=gt_pick_i + 2,
        gt_pick_i=gt_pick_i,
        n_traces=4,
        n_samples_orig=1000,
    )

    assert metrics['fine_center_delta_p90_vs_robust'] < 0.0
    assert metrics['fine_center_delta_p95_vs_robust'] < 0.0


def test_summarize_errors_reports_fine_center_regression() -> None:
    gt_pick_i = np.asarray([100, 200, 300, 400], dtype=np.int64)

    metrics, *_ = physics_qc_cli._summarize_errors(
        coarse_pick_i=gt_pick_i + 40,
        robust_pick_i=gt_pick_i + 2,
        fine_center_i=gt_pick_i + 20,
        gt_pick_i=gt_pick_i,
        n_traces=4,
        n_samples_orig=1000,
    )

    assert metrics['fine_center_delta_p90_vs_robust'] > 0.0
    assert metrics['fine_center_delta_p95_vs_robust'] > 0.0


def test_summarize_errors_reports_physical_and_final_artifact_metrics() -> None:
    gt_pick_i = np.asarray([100, 200, 0, 300], dtype=np.int64)

    metrics, *_ = physics_qc_cli._summarize_errors(
        coarse_pick_i=np.asarray([150, 250, 50, 350], dtype=np.int64),
        robust_pick_i=np.asarray([110, 210, 20, 310], dtype=np.int64),
        physical_center_i=np.asarray([100, 500, 20, 600], dtype=np.int32),
        fine_center_i=np.asarray([100, 200, 20, 300], dtype=np.int32),
        window_start_i=np.asarray([90, 250, 0, 280], dtype=np.int32),
        window_end_i=np.asarray([110, 350, 10, 290], dtype=np.int32),
        final_pick_i=np.asarray([100, 80, 20, 500], dtype=np.int32),
        gt_pick_i=gt_pick_i,
        n_traces=4,
        n_samples_orig=1000,
    )

    assert metrics['gt_in_actual_window_rate'] == pytest.approx(1.0 / 3.0)
    assert metrics['final_pick_R127'] == pytest.approx(2.0 / 3.0)
    assert metrics['physical_center_R127'] == pytest.approx(1.0 / 3.0)


def test_summarize_errors_counts_invalid_final_picks_as_misses() -> None:
    gt_pick_i = np.asarray([100, 500], dtype=np.int64)

    metrics, *_ = physics_qc_cli._summarize_errors(
        coarse_pick_i=gt_pick_i,
        robust_pick_i=gt_pick_i,
        window_start_i=np.asarray([0, 400], dtype=np.int32),
        window_end_i=np.asarray([255, 655], dtype=np.int32),
        final_pick_i=np.asarray([100, 0], dtype=np.int32),
        gt_pick_i=gt_pick_i,
        n_traces=2,
        n_samples_orig=1000,
    )

    assert metrics['final_pick_valid_rate'] == pytest.approx(0.5)
    assert metrics['final_pick_R127'] == pytest.approx(0.5)
    assert metrics['final_pick_abs_err_median'] == pytest.approx(0.0)


def test_summarize_errors_all_invalid_final_picks_keeps_stable_metrics() -> None:
    gt_pick_i = np.asarray([100, 500], dtype=np.int64)

    metrics, *_ = physics_qc_cli._summarize_errors(
        coarse_pick_i=gt_pick_i,
        robust_pick_i=gt_pick_i,
        window_start_i=np.asarray([0, 400], dtype=np.int32),
        window_end_i=np.asarray([255, 655], dtype=np.int32),
        final_pick_i=np.asarray([0, 1000], dtype=np.int32),
        gt_pick_i=gt_pick_i,
        n_traces=2,
        n_samples_orig=1000,
    )

    assert metrics['final_pick_valid_rate'] == pytest.approx(0.0)
    assert metrics['final_pick_R127'] == pytest.approx(0.0)
    assert np.isnan(metrics['final_pick_abs_err_median'])


def test_summarize_errors_absent_final_artifact_keeps_stable_columns() -> None:
    gt_pick_i = np.asarray([100, 200, 300], dtype=np.int64)

    metrics, *_ = physics_qc_cli._summarize_errors(
        coarse_pick_i=gt_pick_i + 10,
        robust_pick_i=gt_pick_i,
        gt_pick_i=gt_pick_i,
        n_traces=3,
        n_samples_orig=1000,
    )

    for column in (
        'gt_in_actual_window_rate',
        'final_pick_valid_rate',
        'final_pick_R32',
        'final_pick_R64',
        'final_pick_R127',
        'final_pick_abs_err_median',
        'final_pick_abs_err_p90',
        'final_pick_abs_err_p95',
    ):
        assert column in metrics
        assert np.isnan(metrics[column])
        assert column in physics_qc_cli.PER_FILE_COLUMNS
        assert column in physics_qc_cli.GLOBAL_COLUMNS


@pytest.mark.parametrize(
    'kwargs, match',
    [
        (
            {'fine_center_i': np.asarray([100, 200], dtype=np.int32)},
            'fine_center_i',
        ),
        (
            {'physical_center_i': np.asarray([100, 200], dtype=np.int32)},
            'physical_center_i',
        ),
        (
            {
                'window_start_i': np.asarray([0, 0], dtype=np.int32),
                'window_end_i': np.asarray([255, 255, 255], dtype=np.int32),
                'final_pick_i': np.asarray([100, 200, 300], dtype=np.int32),
            },
            'window_start_i',
        ),
    ],
)
def test_summarize_errors_optional_shape_mismatch_fails(
    kwargs: dict[str, np.ndarray],
    match: str,
) -> None:
    gt_pick_i = np.asarray([100, 200, 300], dtype=np.int64)

    with pytest.raises(ValueError, match=match):
        physics_qc_cli._summarize_errors(
            coarse_pick_i=gt_pick_i,
            robust_pick_i=gt_pick_i,
            gt_pick_i=gt_pick_i,
            n_traces=3,
            n_samples_orig=1000,
            **kwargs,
        )
