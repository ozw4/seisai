from __future__ import annotations

# ruff: noqa: ANN001, D100, D103, INP001, S101
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from cli import run_fbpick_physics_runtime_benchmark as benchmark


def _write_robust(
    path: Path,
    *,
    physical_center_i: list[int],
    fine_center_i: list[int] | None = None,
    status: list[int] | None = None,
    failure: list[int] | None = None,
) -> Path:
    values = np.asarray(physical_center_i, dtype=np.int32)
    if fine_center_i is None:
        fine_center_i = physical_center_i
    if status is None:
        status = [0] * int(values.size)
    if failure is None:
        failure = [0] * int(values.size)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        physical_center_i=values,
        fine_center_i=np.asarray(fine_center_i, dtype=np.int32),
        physical_model_status=np.asarray(status, dtype=np.int16),
        physical_model_failure_reason=np.asarray(failure, dtype=np.int16),
        physical_model_break_offset_m=np.asarray([100.0, 200.0], dtype=np.float32),
        physical_model_slope_near_s_per_m=np.asarray(
            [0.001, 0.002],
            dtype=np.float32,
        ),
    )
    return path


def _write_runtime(path: Path, *, physics_total_sec: float) -> None:
    path.write_text(
        json.dumps(
            {
                'physics_total_sec': physics_total_sec,
                'physical_center_total_sec': physics_total_sec * 0.9,
                'ransac_fit_total_sec': physics_total_sec * 0.4,
                'non_ransac_total_sec': physics_total_sec * 0.5,
            }
        ),
        encoding='utf-8',
    )


def test_benchmark_manifest_parser_resolves_tagged_paths(tmp_path: Path) -> None:
    manifest_path = tmp_path / 'manifest.yaml'
    manifest_path.write_text(
        yaml.safe_dump(
            {
                'baseline': {
                    'name': 'A0',
                    'robust_npz': 'runs/A0/robust/${TAG}.robust.npz',
                },
                'candidates': [
                    {
                        'name': 'B1',
                        'config': 'configs/B1.yaml',
                        'gates': {
                            'exact_match_required': ['physical_center_i'],
                        },
                    }
                ],
                'checks': {
                    'exact_keys': ['physical_model_status'],
                    'runtime_keys': ['physics_total_sec'],
                },
            },
            sort_keys=False,
        ),
        encoding='utf-8',
    )

    manifest = benchmark.load_manifest(manifest_path, tag='Line01', repo_root=tmp_path)

    assert manifest.baseline.robust_npz == (
        tmp_path / 'runs' / 'A0' / 'robust' / 'Line01.robust.npz'
    )
    assert manifest.candidates[0].config == tmp_path / 'configs' / 'B1.yaml'
    assert manifest.candidates[0].gates == {
        'exact_match_required': ['physical_center_i']
    }
    assert manifest.exact_keys == ('physical_model_status',)
    assert manifest.runtime_keys == ('physics_total_sec',)


def test_benchmark_artifacts_only_passes_exact_gate_and_reports_missing_runtime_key(
    tmp_path: Path,
) -> None:
    baseline = _write_robust(
        tmp_path / 'runs' / 'A0' / 'robust' / 'T.robust.npz',
        physical_center_i=[10, 20],
    )
    candidate = _write_robust(
        tmp_path / 'runs' / 'B1' / 'robust' / 'T.robust.npz',
        physical_center_i=[10, 20],
    )
    _write_runtime(
        tmp_path / 'runs' / 'A0' / 'robust' / 'T.physics_runtime_summary.json',
        physics_total_sec=10.0,
    )
    _write_runtime(
        tmp_path / 'runs' / 'B1' / 'robust' / 'T.physics_runtime_summary.json',
        physics_total_sec=5.0,
    )
    manifest_path = tmp_path / 'manifest.yaml'
    manifest_path.write_text(
        yaml.safe_dump(
            {
                'baseline': {'name': 'A0', 'robust_npz': str(baseline)},
                'candidates': [{'name': 'B1', 'robust_npz': str(candidate)}],
                'checks': {
                    'exact_keys': ['fine_center_i'],
                    'diff_keys': ['physical_center_i'],
                    'runtime_keys': ['physics_total_sec', 'neighbor_plan_sec'],
                },
                'gates': {
                    'exact_match_required': ['physical_model_status'],
                    'max_abs_diff_samples': {'physical_center_i': 0},
                    'min_speedup_physics_total': 1.0,
                },
            },
            sort_keys=False,
        ),
        encoding='utf-8',
    )

    summary = benchmark.run_benchmark(
        manifest_path=manifest_path,
        tag='T',
        out_dir=tmp_path / 'out',
        artifacts_only=True,
        repo_root=tmp_path,
    )

    assert summary['passed'] is True
    candidate_summary = summary['candidates'][0]
    assert candidate_summary['gates']['passed'] is True
    missing_row = next(
        row for row in candidate_summary['runtime'] if row['key'] == 'neighbor_plan_sec'
    )
    assert missing_row['missing_candidate'] is True
    assert (tmp_path / 'out' / 'summary.json').is_file()
    assert (tmp_path / 'out' / 'summary.csv').is_file()
    md_text = (tmp_path / 'out' / 'summary.md').read_text(encoding='utf-8')
    assert 'neighbor_plan_sec' in md_text
    assert 'missing' in md_text
    assert '## Exact Match Summary' in md_text
    assert '| B1 | fine_center_i | pass | pass | pass | no | no |' in md_text


def test_benchmark_tolerance_gate_fails(tmp_path: Path) -> None:
    baseline = _write_robust(
        tmp_path / 'runs' / 'A0' / 'robust' / 'T.robust.npz',
        physical_center_i=[10, 20],
    )
    candidate = _write_robust(
        tmp_path / 'runs' / 'B1' / 'robust' / 'T.robust.npz',
        physical_center_i=[10, 40],
    )
    _write_runtime(
        tmp_path / 'runs' / 'A0' / 'robust' / 'T.physics_runtime_summary.json',
        physics_total_sec=10.0,
    )
    _write_runtime(
        tmp_path / 'runs' / 'B1' / 'robust' / 'T.physics_runtime_summary.json',
        physics_total_sec=8.0,
    )
    manifest_path = tmp_path / 'manifest.yaml'
    manifest_path.write_text(
        yaml.safe_dump(
            {
                'baseline': {'name': 'A0', 'robust_npz': str(baseline)},
                'candidates': [{'name': 'B1', 'robust_npz': str(candidate)}],
                'checks': {'diff_keys': ['physical_center_i']},
                'gates': {'max_p90_abs_diff_samples': {'physical_center_i': 8}},
            },
            sort_keys=False,
        ),
        encoding='utf-8',
    )

    summary = benchmark.run_benchmark(
        manifest_path=manifest_path,
        tag='T',
        out_dir=tmp_path / 'out',
        artifacts_only=True,
        repo_root=tmp_path,
    )

    assert summary['passed'] is False
    gate = summary['candidates'][0]['gates']['checks'][0]
    assert gate['gate'] == 'max_p90_abs_diff_samples'
    assert gate['passed'] is False


def test_benchmark_uses_candidate_specific_gates(tmp_path: Path) -> None:
    baseline = _write_robust(
        tmp_path / 'runs' / 'A0' / 'robust' / 'T.robust.npz',
        physical_center_i=[10, 20],
    )
    strict_candidate = _write_robust(
        tmp_path / 'runs' / 'A1' / 'robust' / 'T.robust.npz',
        physical_center_i=[10, 22],
    )
    loose_candidate = _write_robust(
        tmp_path / 'runs' / 'A3' / 'robust' / 'T.robust.npz',
        physical_center_i=[10, 22],
    )
    manifest_path = tmp_path / 'manifest.yaml'
    manifest_path.write_text(
        yaml.safe_dump(
            {
                'baseline': {'name': 'A0', 'robust_npz': str(baseline)},
                'candidates': [
                    {
                        'name': 'A1',
                        'robust_npz': str(strict_candidate),
                        'gates': {
                            'exact_match_required': ['physical_center_i'],
                        },
                    },
                    {
                        'name': 'A3',
                        'robust_npz': str(loose_candidate),
                        'gates': {
                            'max_p90_abs_diff_samples': {'physical_center_i': 8},
                        },
                    },
                ],
                'checks': {'diff_keys': ['physical_center_i']},
            },
            sort_keys=False,
        ),
        encoding='utf-8',
    )

    summary = benchmark.run_benchmark(
        manifest_path=manifest_path,
        tag='T',
        out_dir=tmp_path / 'out',
        artifacts_only=True,
        repo_root=tmp_path,
    )

    strict_summary, loose_summary = summary['candidates']
    assert summary['passed'] is False
    assert strict_summary['gates']['checks'][0]['gate'] == 'exact_match_required'
    assert strict_summary['gates']['checks'][0]['passed'] is False
    assert loose_summary['gates']['checks'][0]['gate'] == 'max_p90_abs_diff_samples'
    assert loose_summary['gates']['checks'][0]['passed'] is True


def test_benchmark_tolerance_gate_fails_on_one_sided_missing_pick(
    tmp_path: Path,
) -> None:
    baseline = _write_robust(
        tmp_path / 'runs' / 'A0' / 'robust' / 'T.robust.npz',
        physical_center_i=[10, 20, 30],
    )
    candidate = _write_robust(
        tmp_path / 'runs' / 'B1' / 'robust' / 'T.robust.npz',
        physical_center_i=[10, -1, 30],
    )
    manifest_path = tmp_path / 'manifest.yaml'
    manifest_path.write_text(
        yaml.safe_dump(
            {
                'baseline': {'name': 'A0', 'robust_npz': str(baseline)},
                'candidates': [{'name': 'B1', 'robust_npz': str(candidate)}],
                'checks': {'diff_keys': ['physical_center_i']},
                'gates': {
                    'max_abs_diff_samples': {'physical_center_i': 999},
                    'max_p90_abs_diff_samples': {'physical_center_i': 999},
                    'min_within_16_sample_rate': {'physical_center_i': 0.0},
                },
            },
            sort_keys=False,
        ),
        encoding='utf-8',
    )

    summary = benchmark.run_benchmark(
        manifest_path=manifest_path,
        tag='T',
        out_dir=tmp_path / 'out',
        artifacts_only=True,
        repo_root=tmp_path,
    )

    assert summary['passed'] is False
    diff = summary['candidates'][0]['diff_checks']['physical_center_i']
    assert diff['n_one_sided_missing'] == 1
    assert diff['within_16_sample_rate'] == 2 / 3
    for gate in summary['candidates'][0]['gates']['checks']:
        assert gate['passed'] is False
        assert gate['reason'] == 'one-sided missing values are present'


def test_benchmark_exact_gate_fails(tmp_path: Path) -> None:
    baseline = _write_robust(
        tmp_path / 'runs' / 'A0' / 'robust' / 'T.robust.npz',
        physical_center_i=[10, 20],
        status=[0, 0],
    )
    candidate = _write_robust(
        tmp_path / 'runs' / 'B1' / 'robust' / 'T.robust.npz',
        physical_center_i=[10, 20],
        status=[0, 1],
    )
    manifest_path = tmp_path / 'manifest.yaml'
    manifest_path.write_text(
        yaml.safe_dump(
            {
                'baseline': {'name': 'A0', 'robust_npz': str(baseline)},
                'candidates': [{'name': 'B1', 'robust_npz': str(candidate)}],
                'gates': {'exact_match_required': ['physical_model_status']},
            },
            sort_keys=False,
        ),
        encoding='utf-8',
    )

    summary = benchmark.run_benchmark(
        manifest_path=manifest_path,
        tag='T',
        out_dir=tmp_path / 'out',
        artifacts_only=True,
        repo_root=tmp_path,
    )

    assert summary['passed'] is False
    gate = summary['candidates'][0]['gates']['checks'][0]
    assert gate['gate'] == 'exact_match_required'
    assert gate['passed'] is False


def test_benchmark_reports_missing_runtime_summary_file(tmp_path: Path) -> None:
    baseline = _write_robust(
        tmp_path / 'runs' / 'A0' / 'robust' / 'T.robust.npz',
        physical_center_i=[10, 20],
    )
    candidate = _write_robust(
        tmp_path / 'runs' / 'B1' / 'robust' / 'T.robust.npz',
        physical_center_i=[10, 20],
    )
    _write_runtime(
        tmp_path / 'runs' / 'A0' / 'robust' / 'T.physics_runtime_summary.json',
        physics_total_sec=10.0,
    )
    manifest_path = tmp_path / 'manifest.yaml'
    manifest_path.write_text(
        yaml.safe_dump(
            {
                'baseline': {'name': 'A0', 'robust_npz': str(baseline)},
                'candidates': [{'name': 'B1', 'robust_npz': str(candidate)}],
                'checks': {'runtime_keys': ['physics_total_sec']},
                'gates': {'exact_match_required': ['physical_model_status']},
            },
            sort_keys=False,
        ),
        encoding='utf-8',
    )

    summary = benchmark.run_benchmark(
        manifest_path=manifest_path,
        tag='T',
        out_dir=tmp_path / 'out',
        artifacts_only=True,
        repo_root=tmp_path,
    )

    candidate_summary = summary['candidates'][0]
    runtime_row = candidate_summary['runtime'][0]
    assert summary['passed'] is True
    assert candidate_summary['artifacts']['runtime_json_exists'] is False
    assert runtime_row['baseline'] == 10.0
    assert runtime_row['candidate'] is None
    assert runtime_row['missing_candidate'] is True
    md_text = (tmp_path / 'out' / 'summary.md').read_text(encoding='utf-8')
    assert '| B1 | pass | missing | missing | missing | physics_total_sec |' in md_text


def test_benchmark_runs_candidate_config_when_not_artifacts_only(
    monkeypatch,
    tmp_path: Path,
) -> None:
    tag = 'T'
    baseline = _write_robust(
        tmp_path
        / 'proc'
        / 'arakawa'
        / 'outputs'
        / 'runtime_runs'
        / 'A0'
        / 'robust'
        / f'{tag}.robust.npz',
        physical_center_i=[10, 20],
    )
    _write_runtime(
        baseline.with_name(f'{tag}.physics_runtime_summary.json'),
        physics_total_sec=10.0,
    )
    config_dir = (
        tmp_path / 'proc' / 'arakawa' / 'experiments' / 'runtime_speedup' / 'configs'
    )
    config_dir.mkdir(parents=True)
    candidate_config = config_dir / 'B1.yaml'
    candidate_config.write_text('{}\n', encoding='utf-8')
    manifest_path = tmp_path / 'manifest.yaml'
    manifest_path.write_text(
        yaml.safe_dump(
            {
                'baseline': {'name': 'A0', 'robust_npz': str(baseline)},
                'candidates': [{'name': 'B1', 'config': str(candidate_config)}],
                'checks': {'diff_keys': ['physical_center_i']},
                'gates': {'max_abs_diff_samples': {'physical_center_i': 0}},
            },
            sort_keys=False,
        ),
        encoding='utf-8',
    )
    calls: list[Path] = []

    def _fake_run(config_path: Path) -> None:
        calls.append(Path(config_path))
        out = (
            tmp_path
            / 'proc'
            / 'arakawa'
            / 'outputs'
            / 'runtime_runs'
            / 'B1'
            / 'robust'
            / f'{tag}.robust.npz'
        )
        _write_robust(out, physical_center_i=[10, 20])
        _write_runtime(
            out.with_name(f'{tag}.physics_runtime_summary.json'),
            physics_total_sec=5.0,
        )

    monkeypatch.setattr(
        benchmark,
        '_load_runtime',
        lambda: SimpleNamespace(run_arakawa_fbpick_physical_export=_fake_run),
    )

    summary = benchmark.run_benchmark(
        manifest_path=manifest_path,
        tag=tag,
        out_dir=tmp_path / 'out',
        artifacts_only=False,
        repo_root=tmp_path,
    )

    assert calls == [candidate_config]
    assert summary['passed'] is True
    assert summary['candidates'][0]['artifacts']['robust_exists'] is True
    md_text = (tmp_path / 'out' / 'summary.md').read_text(encoding='utf-8')
    assert '## Candidate Artifacts' in md_text
    assert str(candidate_config) in md_text
    assert str(summary['candidates'][0]['artifacts']['robust_npz']) in md_text
    assert '## Status Count Diff' in md_text
    assert 'physical_model_status' in md_text
    assert 'two_piece_ok=2' in md_text
    assert 'none=2' in md_text


def test_benchmark_cli_smoke_artifacts_only_no_fail(tmp_path: Path) -> None:
    baseline = _write_robust(
        tmp_path / 'runs' / 'A0' / 'robust' / 'T.robust.npz',
        physical_center_i=[10, 20],
    )
    candidate = _write_robust(
        tmp_path / 'runs' / 'B1' / 'robust' / 'T.robust.npz',
        physical_center_i=[10, 40],
    )
    manifest_path = tmp_path / 'manifest.yaml'
    manifest_path.write_text(
        yaml.safe_dump(
            {
                'baseline': {'name': 'A0', 'robust_npz': str(baseline)},
                'candidates': [{'name': 'B1', 'robust_npz': str(candidate)}],
                'gates': {'max_abs_diff_samples': {'physical_center_i': 0}},
            },
            sort_keys=False,
        ),
        encoding='utf-8',
    )

    benchmark.main(
        [
            '--manifest',
            str(manifest_path),
            '--tag',
            'T',
            '--out-dir',
            str(tmp_path / 'out'),
            '--artifacts-only',
            '--no-fail-on-gate',
        ]
    )

    assert (tmp_path / 'out' / 'summary.md').is_file()
