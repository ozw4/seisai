"""Run and report fbpick physics runtime benchmark manifests."""

from __future__ import annotations

# ruff: noqa: ANN401, PERF401, PLR0911, PLR0913, PLR0915
import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml

from cli.compare_fbpick_physics_runtime import compare_paths, write_compare_csv

__all__ = [
    'BenchmarkManifest',
    'BenchmarkRunSpec',
    'load_manifest',
    'main',
    'run_benchmark',
]

DEFAULT_RUNTIME_KEYS = (
    'physics_total_sec',
    'physical_center_total_sec',
    'non_ransac_total_sec',
    'ransac_fit_total_sec',
    'neighbor_plan_sec',
    'side_segment_build_sec',
    'prediction_sec',
    'assignment_sec',
)
DEFAULT_SAMPLE_DIFF_KEYS = (
    'physical_center_i',
    'fine_center_i',
)
TAG_PLACEHOLDERS = ('${TAG}', '{tag}')


@dataclass(frozen=True)
class BenchmarkRunSpec:
    """One baseline or candidate entry from a benchmark manifest."""

    name: str
    config: Path | None = None
    robust_npz: Path | None = None
    export_npz: Path | None = None
    runtime_json: Path | None = None
    gates: dict[str, Any] | None = None


@dataclass(frozen=True)
class BenchmarkManifest:
    """Resolved benchmark manifest fields used by the harness."""

    manifest_path: Path
    baseline: BenchmarkRunSpec
    candidates: tuple[BenchmarkRunSpec, ...]
    exact_keys: tuple[str, ...]
    diff_keys: tuple[str, ...]
    runtime_keys: tuple[str, ...]
    gates: dict[str, Any]


@dataclass(frozen=True)
class ResolvedRun:
    name: str
    config: Path | None
    robust_npz: Path
    export_npz: Path | None
    runtime_json: Path | None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_runtime() -> SimpleNamespace:
    from cli.run_arakawa_fbpick_physical_export import run_pipeline  # noqa: PLC0415

    return SimpleNamespace(run_arakawa_fbpick_physical_export=run_pipeline)


def _as_dict(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        msg = f'{name} must be dict'
        raise TypeError(msg)
    return value


def _as_list(value: Any, *, name: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        msg = f'{name} must be list'
        raise TypeError(msg)
    return value


def _string_list(value: Any, *, name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if not isinstance(value, list):
        msg = f'{name} must be list[str]'
        raise TypeError(msg)
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            msg = f'{name} entries must be non-empty str'
            raise TypeError(msg)
        out.append(item)
    return tuple(out)


def _tagged(value: str, *, tag: str | None, field: str) -> str:
    if any(token in value for token in TAG_PLACEHOLDERS):
        if tag is None or not tag:
            msg = f'{field} uses a tag placeholder but --tag was not provided'
            raise ValueError(msg)
        for token in TAG_PLACEHOLDERS:
            value = value.replace(token, tag)
    return value


def _resolve_manifest_path(
    value: Any,
    *,
    field: str,
    repo_root: Path,
    tag: str | None,
) -> Path:
    if not isinstance(value, str) or not value:
        msg = f'{field} must be non-empty str'
        raise TypeError(msg)
    path = Path(_tagged(value, tag=tag, field=field)).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _optional_manifest_path(
    raw: dict[str, Any],
    *,
    keys: tuple[str, ...],
    field_prefix: str,
    repo_root: Path,
    tag: str | None,
) -> Path | None:
    for key in keys:
        value = raw.get(key)
        if value is None:
            continue
        return _resolve_manifest_path(
            value,
            field=f'{field_prefix}.{key}',
            repo_root=repo_root,
            tag=tag,
        )
    return None


def _run_spec_from_raw(
    raw: dict[str, Any],
    *,
    field_prefix: str,
    repo_root: Path,
    tag: str | None,
) -> BenchmarkRunSpec:
    name = raw.get('name')
    if not isinstance(name, str) or not name:
        msg = f'{field_prefix}.name must be non-empty str'
        raise TypeError(msg)

    config = _optional_manifest_path(
        raw,
        keys=('config',),
        field_prefix=field_prefix,
        repo_root=repo_root,
        tag=tag,
    )
    robust_npz = _optional_manifest_path(
        raw,
        keys=('robust_npz',),
        field_prefix=field_prefix,
        repo_root=repo_root,
        tag=tag,
    )
    export_npz = _optional_manifest_path(
        raw,
        keys=('export_npz',),
        field_prefix=field_prefix,
        repo_root=repo_root,
        tag=tag,
    )
    runtime_json = _optional_manifest_path(
        raw,
        keys=('runtime_json',),
        field_prefix=field_prefix,
        repo_root=repo_root,
        tag=tag,
    )
    if robust_npz is None and config is None:
        msg = f'{field_prefix} must set robust_npz or config'
        raise ValueError(msg)

    return BenchmarkRunSpec(
        name=name,
        config=config,
        robust_npz=robust_npz,
        export_npz=export_npz,
        runtime_json=runtime_json,
        gates=dict(_as_dict(raw.get('gates'), name=f'{field_prefix}.gates')),
    )


def load_manifest(
    manifest_path: str | Path,
    *,
    tag: str | None = None,
    repo_root: str | Path | None = None,
) -> BenchmarkManifest:
    """Load a runtime benchmark manifest and resolve repo-root paths."""
    root = Path(repo_root).resolve() if repo_root is not None else _repo_root()
    path = Path(manifest_path).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    payload = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        msg = f'benchmark manifest must be dict: {path}'
        raise TypeError(msg)

    baseline = _run_spec_from_raw(
        _as_dict(payload.get('baseline'), name='baseline'),
        field_prefix='baseline',
        repo_root=root,
        tag=tag,
    )
    candidates_raw = _as_list(payload.get('candidates'), name='candidates')
    if not candidates_raw:
        msg = 'candidates must contain at least one candidate'
        raise ValueError(msg)
    candidates = tuple(
        _run_spec_from_raw(
            _as_dict(item, name=f'candidates[{idx}]'),
            field_prefix=f'candidates[{idx}]',
            repo_root=root,
            tag=tag,
        )
        for idx, item in enumerate(candidates_raw)
    )

    checks = _as_dict(payload.get('checks'), name='checks')
    return BenchmarkManifest(
        manifest_path=path,
        baseline=baseline,
        candidates=candidates,
        exact_keys=_string_list(
            checks.get('exact_keys'),
            name='checks.exact_keys',
            default=(),
        ),
        diff_keys=_string_list(
            checks.get('diff_keys'),
            name='checks.diff_keys',
            default=DEFAULT_SAMPLE_DIFF_KEYS,
        ),
        runtime_keys=_string_list(
            checks.get('runtime_keys'),
            name='checks.runtime_keys',
            default=DEFAULT_RUNTIME_KEYS,
        ),
        gates=dict(_as_dict(payload.get('gates'), name='gates')),
    )


def _load_yaml(path: Path, *, name: str) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        msg = f'{name} must be dict: {path}'
        raise TypeError(msg)
    return payload


def _is_repo_root_relative_arakawa_path(path: Path) -> bool:
    parts = path.parts
    return len(parts) >= 2 and parts[0] == 'proc' and parts[1] == 'arakawa'


def _resolve_config_path(
    value: str | Path,
    *,
    base_dir: Path,
    repo_root: Path | None = None,
) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        if repo_root is not None and _is_repo_root_relative_arakawa_path(path):
            path = repo_root / path
        else:
            path = base_dir / path
    return path.resolve()


def _pick_label(pick_key: str) -> str:
    if pick_key.endswith('_i'):
        return pick_key[:-2]
    return pick_key


def _derive_runtime_summary_path(robust_npz: Path) -> Path:
    suffix = '.robust.npz'
    tag = (
        robust_npz.name[: -len(suffix)]
        if robust_npz.name.endswith(suffix)
        else robust_npz.stem
    )
    return robust_npz.with_name(f'{tag}.physics_runtime_summary.json')


def _runtime_experiment_run_name(
    *,
    config_path: Path,
    repo_root: Path,
) -> str | None:
    runtime_config_dir = (
        repo_root / 'proc' / 'arakawa' / 'experiments' / 'runtime_speedup' / 'configs'
    )
    if config_path.parent.resolve() != runtime_config_dir.resolve():
        return None
    return config_path.stem


def _derive_from_config(
    spec: BenchmarkRunSpec,
    *,
    tag: str | None,
    repo_root: Path,
) -> tuple[Path, Path | None]:
    if spec.robust_npz is not None:
        return spec.robust_npz, spec.export_npz
    if spec.config is None:
        msg = f'{spec.name} needs robust_npz when config is absent'
        raise ValueError(msg)
    if tag is None or not tag:
        msg = f'{spec.name} needs --tag to derive artifact paths from config'
        raise ValueError(msg)

    cfg = _load_yaml(spec.config, name=f'{spec.name} config')
    base_dir = spec.config.parent
    paths = _as_dict(cfg.get('paths'), name=f'{spec.name}.paths')
    arakawa_dir = repo_root / 'proc' / 'arakawa'
    default_work_dir = arakawa_dir / 'outputs'
    run_name = _runtime_experiment_run_name(
        config_path=spec.config,
        repo_root=repo_root,
    )
    if run_name is not None:
        default_work_dir = default_work_dir / 'runtime_runs' / run_name

    work_dir = _resolve_config_path(
        paths.get('work_dir', str(default_work_dir)),
        base_dir=base_dir,
        repo_root=repo_root,
    )
    robust_dir = _resolve_config_path(
        paths.get('robust_dir', str(work_dir / 'robust')),
        base_dir=base_dir,
        repo_root=repo_root,
    )
    grstat_dir = _resolve_config_path(
        paths.get('grstat_dir', str(work_dir / 'grstat')),
        base_dir=base_dir,
        repo_root=repo_root,
    )

    robust_npz = spec.robust_npz or robust_dir / f'{tag}.robust.npz'
    if spec.export_npz is not None:
        return robust_npz, spec.export_npz

    export_cfg = _as_dict(cfg.get('export'), name=f'{spec.name}.export')
    pick_key = export_cfg.get('pick_key', 'physical_center_i')
    phase_mode = export_cfg.get('phase_mode', 'peak')
    max_shift_samples = export_cfg.get('max_shift_samples', 2)
    if not isinstance(pick_key, str) or not pick_key:
        msg = f'{spec.name}.export.pick_key must be non-empty str'
        raise TypeError(msg)
    if not isinstance(phase_mode, str) or not phase_mode:
        msg = f'{spec.name}.export.phase_mode must be non-empty str'
        raise TypeError(msg)
    if not isinstance(max_shift_samples, int):
        msg = f'{spec.name}.export.max_shift_samples must be int'
        raise TypeError(msg)

    out_label = (
        f'{tag}.{_pick_label(pick_key)}.snap_{phase_mode}.ltcor{max_shift_samples}'
    )
    out_npz_value = paths.get('out_npz')
    export_npz = (
        _resolve_config_path(out_npz_value, base_dir=base_dir, repo_root=repo_root)
        if out_npz_value is not None
        else grstat_dir / f'{out_label}.npz'
    )
    return robust_npz, export_npz


def _resolve_run(
    spec: BenchmarkRunSpec,
    *,
    tag: str | None,
    repo_root: Path,
) -> ResolvedRun:
    robust_npz, export_npz = _derive_from_config(
        spec,
        tag=tag,
        repo_root=repo_root,
    )
    return ResolvedRun(
        name=spec.name,
        config=spec.config,
        robust_npz=robust_npz,
        export_npz=export_npz,
        runtime_json=spec.runtime_json or _derive_runtime_summary_path(robust_npz),
    )


def _json_scalar(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _json_scalar(value.item())
        return [_json_scalar(item) for item in value.tolist()]
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(key): _json_scalar(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_scalar(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_scalar(payload), ensure_ascii=True, indent=2, sort_keys=True),
        encoding='utf-8',
    )
    return path


def _read_runtime_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        msg = f'runtime summary must be a JSON object: {path}'
        raise TypeError(msg)
    return payload


def _runtime_speedup_from_values(baseline: Any, candidate: Any) -> float | None:
    if baseline is None or candidate is None:
        return None
    if isinstance(baseline, bool) or isinstance(candidate, bool):
        return None
    if not isinstance(baseline, int | float) or not isinstance(candidate, int | float):
        return None
    if not np.isfinite(float(baseline)) or not np.isfinite(float(candidate)):
        return None
    if float(candidate) <= 0.0:
        return None
    return float(baseline) / float(candidate)


def _runtime_value(
    *,
    summary: dict[str, Any] | None,
    comparison_runtime: dict[str, Any],
    summary_key: str,
    comparison_key: str,
) -> Any:
    if summary is not None and summary_key in summary:
        return summary[summary_key]
    return comparison_runtime.get(comparison_key)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _valid_numeric_mask(values: np.ndarray, *, negative_is_missing: bool) -> np.ndarray:
    if np.issubdtype(values.dtype, np.floating):
        valid = np.isfinite(values)
    else:
        valid = np.ones(values.shape, dtype=np.bool_)
    if negative_is_missing and np.issubdtype(values.dtype, np.number):
        valid &= values >= 0
    return valid


def _rate(mask: np.ndarray) -> float | None:
    if mask.size == 0:
        return None
    return float(np.mean(mask.astype(np.float64)))


def _percentile(values: np.ndarray, q: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.percentile(values.astype(np.float64, copy=False), q))


def _exact_check(
    *,
    key: str,
    baseline: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
) -> dict[str, Any]:
    if key not in baseline or key not in candidate:
        return {
            'key': key,
            'available': False,
            'missing_baseline_field': key not in baseline,
            'missing_candidate_field': key not in candidate,
            'arrays_match': False,
        }
    base = np.asarray(baseline[key])
    cand = np.asarray(candidate[key])
    return {
        'key': key,
        'available': True,
        'missing_baseline_field': False,
        'missing_candidate_field': False,
        'shape_match': base.shape == cand.shape,
        'arrays_match': bool(np.array_equal(base, cand)),
    }


def _diff_check(
    *,
    key: str,
    baseline: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
) -> dict[str, Any]:
    if key not in baseline or key not in candidate:
        return {
            'key': key,
            'available': False,
            'missing_baseline_field': key not in baseline,
            'missing_candidate_field': key not in candidate,
        }

    base = np.asarray(baseline[key]).reshape(-1)
    cand = np.asarray(candidate[key]).reshape(-1)
    if base.shape != cand.shape:
        msg = f'{key} shape mismatch: baseline {base.shape}, candidate {cand.shape}'
        raise ValueError(msg)
    if not np.issubdtype(base.dtype, np.number) or not np.issubdtype(
        cand.dtype,
        np.number,
    ):
        msg = f'{key} must be numeric for diff checks'
        raise TypeError(msg)

    negative_is_missing = key.endswith('_i')
    valid_base = _valid_numeric_mask(base, negative_is_missing=negative_is_missing)
    valid_cand = _valid_numeric_mask(cand, negative_is_missing=negative_is_missing)
    valid = valid_base & valid_cand
    missing_base = ~valid_base
    missing_cand = ~valid_cand
    missing_both = missing_base & missing_cand
    one_sided_missing = missing_base ^ missing_cand
    diff = cand[valid].astype(np.float64, copy=False) - base[valid].astype(
        np.float64,
        copy=False,
    )
    abs_diff = np.abs(diff)
    n_one_sided_missing = int(np.count_nonzero(one_sided_missing))

    def _within_sample_rate(threshold: int) -> float | None:
        within = abs_diff <= threshold
        if n_one_sided_missing > 0:
            missing_mismatches = np.zeros(n_one_sided_missing, dtype=np.bool_)
            within = np.concatenate((within, missing_mismatches))
        return _rate(within)

    return {
        'key': key,
        'available': True,
        'n': int(base.size),
        'n_valid_both': int(np.count_nonzero(valid)),
        'n_missing_baseline': int(np.count_nonzero(missing_base)),
        'n_missing_candidate': int(np.count_nonzero(missing_cand)),
        'n_missing_both': int(np.count_nonzero(missing_both)),
        'n_one_sided_missing': n_one_sided_missing,
        'bias_mean': None if diff.size == 0 else float(np.mean(diff)),
        'abs_diff_mean': None if abs_diff.size == 0 else float(np.mean(abs_diff)),
        'abs_diff_p50': _percentile(abs_diff, 50.0),
        'abs_diff_p90': _percentile(abs_diff, 90.0),
        'abs_diff_p95': _percentile(abs_diff, 95.0),
        'abs_diff_p99': _percentile(abs_diff, 99.0),
        'abs_diff_max': None if abs_diff.size == 0 else float(np.max(abs_diff)),
        'within_1_sample_rate': _within_sample_rate(1),
        'within_2_sample_rate': _within_sample_rate(2),
        'within_4_sample_rate': _within_sample_rate(4),
        'within_8_sample_rate': _within_sample_rate(8),
        'within_16_sample_rate': _within_sample_rate(16),
    }


def _ordered_unique(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return tuple(out)


def _numeric_gate_map(
    value: Any,
    *,
    gate_name: str,
    default_keys: tuple[str, ...],
) -> dict[str, float]:
    if value is None:
        return {}
    if isinstance(value, int | float):
        return {key: float(value) for key in default_keys}
    if not isinstance(value, dict):
        msg = f'gates.{gate_name} must be number or dict[str, number]'
        raise TypeError(msg)
    out: dict[str, float] = {}
    for key, raw_threshold in value.items():
        if not isinstance(key, str) or not key:
            msg = f'gates.{gate_name} keys must be non-empty str'
            raise TypeError(msg)
        if not isinstance(raw_threshold, int | float):
            msg = f'gates.{gate_name}.{key} must be numeric'
            raise TypeError(msg)
        out[key] = float(raw_threshold)
    return out


def _candidate_gates(
    *,
    manifest_gates: dict[str, Any],
    candidate: BenchmarkRunSpec,
) -> dict[str, Any]:
    merged = dict(manifest_gates)
    if not candidate.gates:
        return merged
    for key, value in candidate.gates.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = {**existing, **value}
        elif isinstance(existing, list) and isinstance(value, list):
            merged[key] = list(value)
        else:
            merged[key] = value
    return merged


def _manifest_gate_sets(manifest: BenchmarkManifest) -> tuple[dict[str, Any], ...]:
    gates: list[dict[str, Any]] = [manifest.gates]
    gates.extend(
        _candidate_gates(manifest_gates=manifest.gates, candidate=candidate)
        for candidate in manifest.candidates
    )
    return tuple(gates)


def _gate_row(
    *,
    gate: str,
    passed: bool,
    key: str | None = None,
    value: Any = None,
    threshold: Any = None,
    reason: str = '',
) -> dict[str, Any]:
    return {
        'gate': gate,
        'key': key,
        'value': value,
        'threshold': threshold,
        'passed': bool(passed),
        'reason': reason,
    }


def _one_sided_missing_count(check: dict[str, Any] | None) -> int:
    if check is None:
        return 0
    return int(check.get('n_one_sided_missing') or 0)


def _evaluate_gates(
    *,
    gates: dict[str, Any],
    exact_checks: dict[str, dict[str, Any]],
    diff_checks: dict[str, dict[str, Any]],
    comparison: dict[str, Any] | None,
    artifact_available: bool,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    if not artifact_available:
        checks.append(
            _gate_row(
                gate='artifacts_available',
                passed=False,
                reason='baseline or candidate robust artifact is missing',
            )
        )

    exact_required = _string_list(
        gates.get('exact_match_required'),
        name='gates.exact_match_required',
        default=(),
    )
    for key in exact_required:
        check = exact_checks.get(key)
        passed = bool(check and check.get('available') and check.get('arrays_match'))
        checks.append(
            _gate_row(
                gate='exact_match_required',
                key=key,
                value=check.get('arrays_match') if check else None,
                threshold=True,
                passed=passed,
                reason='' if passed else 'arrays differ or field is missing',
            )
        )

    sample_keys = tuple(key for key in diff_checks if key.endswith('_i')) or tuple(
        diff_checks
    )
    max_abs = _numeric_gate_map(
        gates.get('max_abs_diff_samples'),
        gate_name='max_abs_diff_samples',
        default_keys=sample_keys,
    )
    for key, threshold in max_abs.items():
        check = diff_checks.get(key)
        value = check.get('abs_diff_max') if check else None
        n_one_sided_missing = _one_sided_missing_count(check)
        passed = (
            value is not None
            and float(value) <= threshold
            and n_one_sided_missing == 0
        )
        checks.append(
            _gate_row(
                gate='max_abs_diff_samples',
                key=key,
                value=value,
                threshold=threshold,
                passed=passed,
                reason=(
                    ''
                    if passed
                    else (
                        'one-sided missing values are present'
                        if n_one_sided_missing > 0
                        else 'max absolute diff exceeds threshold'
                    )
                ),
            )
        )

    max_p90 = _numeric_gate_map(
        gates.get('max_p90_abs_diff_samples'),
        gate_name='max_p90_abs_diff_samples',
        default_keys=sample_keys,
    )
    for key, threshold in max_p90.items():
        check = diff_checks.get(key)
        value = check.get('abs_diff_p90') if check else None
        n_one_sided_missing = _one_sided_missing_count(check)
        passed = (
            value is not None
            and float(value) <= threshold
            and n_one_sided_missing == 0
        )
        checks.append(
            _gate_row(
                gate='max_p90_abs_diff_samples',
                key=key,
                value=value,
                threshold=threshold,
                passed=passed,
                reason=(
                    ''
                    if passed
                    else (
                        'one-sided missing values are present'
                        if n_one_sided_missing > 0
                        else 'p90 absolute diff exceeds threshold'
                    )
                ),
            )
        )

    min_within16 = _numeric_gate_map(
        gates.get('min_within_16_sample_rate'),
        gate_name='min_within_16_sample_rate',
        default_keys=sample_keys,
    )
    for key, threshold in min_within16.items():
        check = diff_checks.get(key)
        value = check.get('within_16_sample_rate') if check else None
        n_one_sided_missing = _one_sided_missing_count(check)
        passed = (
            value is not None
            and float(value) >= threshold
            and n_one_sided_missing == 0
        )
        checks.append(
            _gate_row(
                gate='min_within_16_sample_rate',
                key=key,
                value=value,
                threshold=threshold,
                passed=passed,
                reason=(
                    ''
                    if passed
                    else (
                        'one-sided missing values are present'
                        if n_one_sided_missing > 0
                        else 'within-16 sample rate is below threshold'
                    )
                ),
            )
        )

    if 'min_speedup_physics_total' in gates:
        threshold = gates['min_speedup_physics_total']
        if not isinstance(threshold, int | float):
            msg = 'gates.min_speedup_physics_total must be numeric'
            raise TypeError(msg)
        runtime = comparison.get('runtime') if comparison is not None else None
        value = (
            runtime.get('speedup_physics_total')
            if isinstance(runtime, dict)
            else None
        )
        passed = value is not None and float(value) >= float(threshold)
        checks.append(
            _gate_row(
                gate='min_speedup_physics_total',
                value=value,
                threshold=float(threshold),
                passed=passed,
                reason='' if passed else 'physics total speedup is below threshold',
            )
        )

    if gates.get('allow_status_count_change') is not True:
        status_counts = comparison.get('status_counts') if comparison else None
        for key in ('physical_model_status', 'physical_model_failure_reason'):
            group = status_counts.get(key) if isinstance(status_counts, dict) else None
            value = group.get('counts_match') if isinstance(group, dict) else None
            passed = value is True
            checks.append(
                _gate_row(
                    gate='status_counts_match',
                    key=key,
                    value=value,
                    threshold=True,
                    passed=passed,
                    reason='' if passed else 'status counts changed or are missing',
                )
            )

    return {
        'passed': all(bool(check['passed']) for check in checks),
        'checks': checks,
    }


def _runtime_rows(
    *,
    runtime: dict[str, Any] | None,
    runtime_keys: tuple[str, ...],
    baseline_runtime_summary: dict[str, Any] | None = None,
    candidate_runtime_summary: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    comparison_runtime = runtime or {}
    for key in runtime_keys:
        baseline = _runtime_value(
            summary=baseline_runtime_summary,
            comparison_runtime=comparison_runtime,
            summary_key=key,
            comparison_key=f'{key}_baseline',
        )
        candidate = _runtime_value(
            summary=candidate_runtime_summary,
            comparison_runtime=comparison_runtime,
            summary_key=key,
            comparison_key=f'{key}_candidate',
        )
        speedup = _runtime_speedup_from_values(baseline, candidate)
        if speedup is None:
            speedup = comparison_runtime.get(f'speedup_{key.removesuffix("_sec")}')
        rows.append(
            {
                'key': key,
                'baseline': baseline,
                'candidate': candidate,
                'speedup': speedup,
                'missing_baseline': baseline is None,
                'missing_candidate': candidate is None,
            }
        )
    return rows


def _format_value(value: Any) -> str:
    if value is None:
        return 'missing'
    if isinstance(value, bool):
        return 'pass' if value else 'fail'
    if isinstance(value, float):
        return f'{value:.6g}'
    return str(value)


def _format_path_cell(value: Any) -> str:
    if value is None or value == '':
        return ''
    return f'`{value}`'


def _format_count_map(value: Any) -> str:
    if not isinstance(value, dict):
        return 'missing'
    if not value:
        return ''
    return '; '.join(
        f'{label}={_format_value(count)}' for label, count in value.items()
    )


def _format_missing_flag(value: Any) -> str:
    if value is None:
        return 'missing'
    return 'yes' if bool(value) else 'no'


def _append_candidate_artifacts_section(
    lines: list[str],
    *,
    summary: dict[str, Any],
) -> None:
    lines.append('## Candidate Artifacts')
    lines.append('')
    lines.append(
        '| candidate | config | robust_npz | export_npz | runtime_summary | '
        'comparison_json | comparison_csv |'
    )
    lines.append('|---|---|---|---|---|---|---|')
    for candidate in summary['candidates']:
        artifacts = candidate['artifacts']
        lines.append(
            '| '
            + ' | '.join(
                (
                    candidate['name'],
                    _format_path_cell(candidate.get('config')),
                    _format_path_cell(artifacts.get('robust_npz')),
                    _format_path_cell(artifacts.get('export_npz')),
                    _format_path_cell(artifacts.get('runtime_json')),
                    _format_path_cell(artifacts.get('comparison_json')),
                    _format_path_cell(artifacts.get('comparison_csv')),
                )
            )
            + ' |'
        )
    lines.append('')


def _append_exact_match_summary_section(
    lines: list[str],
    *,
    summary: dict[str, Any],
) -> None:
    lines.append('## Exact Match Summary')
    lines.append('')
    lines.append(
        '| candidate | key | available | shape_match | arrays_match | '
        'missing_baseline | missing_candidate |'
    )
    lines.append('|---|---|---|---|---|---|---|')
    for candidate in summary['candidates']:
        exact_checks = candidate.get('exact_checks') or {}
        if not exact_checks:
            lines.append(
                f'| {candidate["name"]} | none | missing | missing | missing | '
                'missing | missing |'
            )
            continue
        for key, check in exact_checks.items():
            lines.append(
                '| '
                + ' | '.join(
                    (
                        candidate['name'],
                        key,
                        _format_value(check.get('available')),
                        _format_value(check.get('shape_match')),
                        _format_value(check.get('arrays_match')),
                        _format_missing_flag(check.get('missing_baseline_field')),
                        _format_missing_flag(check.get('missing_candidate_field')),
                    )
                )
                + ' |'
            )
    lines.append('')


def _append_status_count_diff_section(
    lines: list[str],
    *,
    summary: dict[str, Any],
) -> None:
    lines.append('## Status Count Diff')
    lines.append('')
    lines.append(
        '| candidate | key | available | counts_match | arrays_match | '
        'baseline_counts | candidate_counts |'
    )
    lines.append('|---|---|---|---|---|---|---|')
    for candidate in summary['candidates']:
        comparison = candidate.get('comparison') or {}
        status_counts = comparison.get('status_counts') or {}
        if not status_counts:
            lines.append(
                f'| {candidate["name"]} | missing | fail | fail | fail | missing | '
                'missing |'
            )
            continue
        for key, group in status_counts.items():
            lines.append(
                '| '
                + ' | '.join(
                    (
                        candidate['name'],
                        key,
                        _format_value(group.get('available')),
                        _format_value(group.get('counts_match')),
                        _format_value(group.get('arrays_match')),
                        _format_count_map(group.get('baseline')),
                        _format_count_map(group.get('candidate')),
                    )
                )
                + ' |'
            )
    lines.append('')


def _write_summary_csv(path: Path, summary: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(('candidate', 'section', 'key', 'value'))
        writer.writerow(
            (
                'baseline',
                'artifact',
                'robust_npz',
                summary['baseline']['robust_npz'],
            )
        )
        for candidate in summary['candidates']:
            name = candidate['name']
            for key, value in candidate['artifacts'].items():
                writer.writerow((name, 'artifact', key, value))
            for row in candidate['runtime']:
                for key in ('baseline', 'candidate', 'speedup'):
                    writer.writerow((name, 'runtime', f'{row["key"]}_{key}', row[key]))
            for key, check in candidate['exact_checks'].items():
                writer.writerow(
                    (
                        name,
                        'exact',
                        f'{key}_arrays_match',
                        check.get('arrays_match'),
                    )
                )
            for key, check in candidate['diff_checks'].items():
                for metric in (
                    'n_valid_both',
                    'n_one_sided_missing',
                    'abs_diff_p90',
                    'abs_diff_p99',
                    'abs_diff_max',
                    'within_16_sample_rate',
                ):
                    writer.writerow(
                        (name, 'diff', f'{key}_{metric}', check.get(metric))
                    )
            for check in candidate['gates']['checks']:
                suffix = f'.{check["key"]}' if check.get('key') else ''
                writer.writerow(
                    (name, 'gate', f'{check["gate"]}{suffix}', check['passed'])
                )
    return path


def _write_summary_markdown(path: Path, summary: dict[str, Any]) -> Path:
    lines: list[str] = []
    lines.append('# FBPick physics runtime benchmark')
    lines.append('')
    lines.append(f'- manifest: `{summary["manifest"]}`')
    lines.append(f'- tag: `{summary.get("tag") or ""}`')
    lines.append(f'- artifacts_only: `{summary["artifacts_only"]}`')
    lines.append('')
    lines.append('## Baseline')
    lines.append('')
    baseline = summary['baseline']
    lines.append('| name | config | robust_npz | export_npz | runtime_summary |')
    lines.append('|---|---|---|---|---|')
    lines.append(
        '| '
        + ' | '.join(
            (
                baseline['name'],
                _format_path_cell(baseline.get('config')),
                _format_path_cell(baseline['robust_npz']),
                _format_path_cell(baseline.get('export_npz')),
                _format_path_cell(baseline.get('runtime_json')),
            )
        )
        + ' |'
    )
    lines.append('')
    _append_candidate_artifacts_section(lines, summary=summary)
    lines.append('## Runtime Summary')
    lines.append('')
    lines.append(
        '| candidate | gates | physics_total_sec | speedup_physics_total | '
        'ransac_fit_total_sec | missing_runtime_keys |'
    )
    lines.append('|---|---|---:|---:|---:|---|')
    for candidate in summary['candidates']:
        runtime_by_key = {row['key']: row for row in candidate['runtime']}
        missing = [
            row['key']
            for row in candidate['runtime']
            if row['missing_baseline'] or row['missing_candidate']
        ]
        physics = runtime_by_key.get('physics_total_sec', {})
        ransac = runtime_by_key.get('ransac_fit_total_sec', {})
        lines.append(
            '| '
            + ' | '.join(
                (
                    candidate['name'],
                    'pass' if candidate['passed'] else 'fail',
                    _format_value(physics.get('candidate')),
                    _format_value(physics.get('speedup')),
                    _format_value(ransac.get('candidate')),
                    ', '.join(missing) if missing else '',
                )
            )
            + ' |'
        )
    lines.append('')
    lines.append('## Detailed Timing')
    lines.append('')
    header = ['candidate', *summary['runtime_keys']]
    lines.append('| ' + ' | '.join(header) + ' |')
    lines.append('|' + '|'.join('---' for _ in header) + '|')
    for candidate in summary['candidates']:
        values = {row['key']: row for row in candidate['runtime']}
        row = [candidate['name']]
        row.extend(
            _format_value(values.get(key, {}).get('candidate'))
            for key in summary['runtime_keys']
        )
        lines.append('| ' + ' | '.join(row) + ' |')
    lines.append('')
    _append_exact_match_summary_section(lines, summary=summary)
    lines.append('## Diff Summary')
    lines.append('')
    lines.append(
        '| candidate | key | valid_both | one_sided_missing | p90 | p99 | max | '
        'within_16 |'
    )
    lines.append('|---|---|---:|---:|---:|---:|---:|---:|')
    for candidate in summary['candidates']:
        for key, check in candidate['diff_checks'].items():
            lines.append(
                '| '
                + ' | '.join(
                    (
                        candidate['name'],
                        key,
                        _format_value(check.get('n_valid_both')),
                        _format_value(check.get('n_one_sided_missing')),
                        _format_value(check.get('abs_diff_p90')),
                        _format_value(check.get('abs_diff_p99')),
                        _format_value(check.get('abs_diff_max')),
                        _format_value(check.get('within_16_sample_rate')),
                    )
                )
                + ' |'
            )
    lines.append('')
    _append_status_count_diff_section(lines, summary=summary)
    lines.append('## Gate Summary')
    lines.append('')
    lines.append('| candidate | gate | key | value | threshold | result |')
    lines.append('|---|---|---|---:|---:|---|')
    for candidate in summary['candidates']:
        if not candidate['gates']['checks']:
            lines.append(f'| {candidate["name"]} | none |  |  |  | pass |')
            continue
        for check in candidate['gates']['checks']:
            lines.append(
                '| '
                + ' | '.join(
                    (
                        candidate['name'],
                        check['gate'],
                        check.get('key') or '',
                        _format_value(check.get('value')),
                        _format_value(check.get('threshold')),
                        'pass' if check['passed'] else 'fail',
                    )
                )
                + ' |'
            )
    lines.append('')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path


def _run_required_artifacts(
    *,
    baseline: ResolvedRun,
    candidates: tuple[ResolvedRun, ...],
    artifacts_only: bool,
) -> None:
    if artifacts_only:
        return
    runtime = _load_runtime()
    if baseline.config is not None and not baseline.robust_npz.is_file():
        print(f'[run] baseline: {baseline.config}')
        runtime.run_arakawa_fbpick_physical_export(baseline.config)
    for candidate in candidates:
        if candidate.config is None:
            continue
        print(f'[run] candidate {candidate.name}: {candidate.config}')
        runtime.run_arakawa_fbpick_physical_export(candidate.config)


def run_benchmark(
    *,
    manifest_path: str | Path,
    tag: str | None,
    out_dir: str | Path,
    artifacts_only: bool = False,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run a benchmark manifest and write combined reports."""
    root = Path(repo_root).resolve() if repo_root is not None else _repo_root()
    manifest = load_manifest(manifest_path, tag=tag, repo_root=root)
    baseline = _resolve_run(manifest.baseline, tag=tag, repo_root=root)
    candidates = tuple(
        _resolve_run(candidate, tag=tag, repo_root=root)
        for candidate in manifest.candidates
    )
    output_dir = Path(out_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir = output_dir.resolve()
    comparison_dir = output_dir / 'comparisons'

    _run_required_artifacts(
        baseline=baseline,
        candidates=candidates,
        artifacts_only=artifacts_only,
    )

    baseline_payload: dict[str, np.ndarray] | None = None
    baseline_exists = baseline.robust_npz.is_file()
    if baseline_exists:
        baseline_payload = _load_npz(baseline.robust_npz)
    baseline_runtime_summary = _read_runtime_json(baseline.runtime_json)

    gate_sets = _manifest_gate_sets(manifest)
    exact_gate_keys = tuple(
        key
        for gates in gate_sets
        for key in _string_list(
            gates.get('exact_match_required'),
            name='gates.exact_match_required',
            default=(),
        )
    )
    max_abs_keys = tuple(
        key
        for gates in gate_sets
        for key in _numeric_gate_map(
            gates.get('max_abs_diff_samples'),
            gate_name='max_abs_diff_samples',
            default_keys=manifest.diff_keys,
        )
    )
    max_p90_keys = tuple(
        key
        for gates in gate_sets
        for key in _numeric_gate_map(
            gates.get('max_p90_abs_diff_samples'),
            gate_name='max_p90_abs_diff_samples',
            default_keys=manifest.diff_keys,
        )
    )
    min_within16_keys = tuple(
        key
        for gates in gate_sets
        for key in _numeric_gate_map(
            gates.get('min_within_16_sample_rate'),
            gate_name='min_within_16_sample_rate',
            default_keys=manifest.diff_keys,
        )
    )
    exact_keys = _ordered_unique([*manifest.exact_keys, *exact_gate_keys])
    diff_keys = _ordered_unique(
        [
            *manifest.diff_keys,
            *max_abs_keys,
            *max_p90_keys,
            *min_within16_keys,
        ]
    )

    summary_candidates: list[dict[str, Any]] = []
    for candidate, candidate_spec in zip(candidates, manifest.candidates, strict=True):
        candidate_exists = candidate.robust_npz.is_file()
        candidate_runtime_summary = _read_runtime_json(candidate.runtime_json)
        artifact_available = baseline_exists and candidate_exists
        comparison: dict[str, Any] | None = None
        comparison_json: Path | None = None
        comparison_csv: Path | None = None
        exact_checks: dict[str, dict[str, Any]] = {}
        diff_checks: dict[str, dict[str, Any]] = {}

        if artifact_available and baseline_payload is not None:
            candidate_payload = _load_npz(candidate.robust_npz)
            compare_kwargs: dict[str, Any] = {
                'baseline_robust': baseline.robust_npz,
                'candidate_robust': candidate.robust_npz,
                'baseline_runtime_json': baseline.runtime_json,
                'candidate_runtime_json': candidate.runtime_json,
            }
            exports_available = (
                baseline.export_npz is not None
                and candidate.export_npz is not None
                and baseline.export_npz.is_file()
                and candidate.export_npz.is_file()
            )
            if exports_available:
                compare_kwargs['baseline_export'] = baseline.export_npz
                compare_kwargs['candidate_export'] = candidate.export_npz

            comparison = compare_paths(**compare_kwargs)
            comparison_json = (
                comparison_dir / f'{candidate.name}_vs_{baseline.name}.json'
            )
            comparison_csv = (
                comparison_dir / f'{candidate.name}_vs_{baseline.name}.csv'
            )
            _write_json(comparison_json, comparison)
            write_compare_csv(comparison_csv, comparison)

            exact_checks = {
                key: _exact_check(
                    key=key,
                    baseline=baseline_payload,
                    candidate=candidate_payload,
                )
                for key in exact_keys
            }
            diff_checks = {
                key: _diff_check(
                    key=key,
                    baseline=baseline_payload,
                    candidate=candidate_payload,
                )
                for key in diff_keys
            }

        gates = _evaluate_gates(
            gates=_candidate_gates(
                manifest_gates=manifest.gates,
                candidate=candidate_spec,
            ),
            exact_checks=exact_checks,
            diff_checks=diff_checks,
            comparison=comparison,
            artifact_available=artifact_available,
        )
        candidate_summary = {
            'name': candidate.name,
            'config': str(candidate.config) if candidate.config is not None else None,
            'artifacts': {
                'robust_npz': str(candidate.robust_npz),
                'robust_exists': candidate_exists,
                'export_npz': (
                    str(candidate.export_npz) if candidate.export_npz else None
                ),
                'export_exists': (
                    candidate.export_npz.is_file()
                    if candidate.export_npz is not None
                    else None
                ),
                'runtime_json': (
                    str(candidate.runtime_json) if candidate.runtime_json else None
                ),
                'runtime_json_exists': (
                    candidate.runtime_json.is_file()
                    if candidate.runtime_json is not None
                    else None
                ),
                'comparison_json': (
                    str(comparison_json) if comparison_json is not None else None
                ),
                'comparison_csv': (
                    str(comparison_csv) if comparison_csv is not None else None
                ),
            },
            'comparison': comparison,
            'runtime': _runtime_rows(
                runtime=comparison.get('runtime') if comparison is not None else None,
                runtime_keys=manifest.runtime_keys,
                baseline_runtime_summary=baseline_runtime_summary,
                candidate_runtime_summary=candidate_runtime_summary,
            ),
            'exact_checks': exact_checks,
            'diff_checks': diff_checks,
            'gates': gates,
            'passed': bool(artifact_available and gates['passed']),
        }
        summary_candidates.append(candidate_summary)

    summary: dict[str, Any] = {
        'manifest': str(manifest.manifest_path),
        'tag': tag,
        'artifacts_only': bool(artifacts_only),
        'runtime_keys': list(manifest.runtime_keys),
        'exact_keys': list(exact_keys),
        'diff_keys': list(diff_keys),
        'baseline': {
            'name': baseline.name,
            'config': str(baseline.config) if baseline.config is not None else None,
            'robust_npz': str(baseline.robust_npz),
            'robust_exists': baseline_exists,
            'export_npz': str(baseline.export_npz) if baseline.export_npz else None,
            'runtime_json': (
                str(baseline.runtime_json) if baseline.runtime_json else None
            ),
        },
        'candidates': summary_candidates,
        'passed': all(candidate['passed'] for candidate in summary_candidates),
    }

    summary_json = _write_json(output_dir / 'summary.json', summary)
    summary_csv = _write_summary_csv(output_dir / 'summary.csv', summary)
    summary_md = _write_summary_markdown(output_dir / 'summary.md', summary)
    summary['outputs'] = {
        'summary_json': str(summary_json),
        'summary_csv': str(summary_csv),
        'summary_md': str(summary_md),
    }
    _write_json(output_dir / 'summary.json', summary)
    return _json_scalar(summary)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--tag')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument(
        '--artifacts-only',
        action='store_true',
        help=(
            'Do not run baseline or candidate configs; compare existing '
            'artifacts only.'
        ),
    )
    parser.add_argument(
        '--no-fail-on-gate',
        action='store_true',
        help='Write reports but return success even when a quality gate fails.',
    )
    args = parser.parse_args(argv)

    summary = run_benchmark(
        manifest_path=args.manifest,
        tag=args.tag,
        out_dir=args.out_dir,
        artifacts_only=args.artifacts_only,
    )
    print(f'wrote_json: {summary["outputs"]["summary_json"]}')
    print(f'wrote_csv: {summary["outputs"]["summary_csv"]}')
    print(f'wrote_markdown: {summary["outputs"]["summary_md"]}')
    print(f'gates_passed: {summary["passed"]}')
    if not args.no_fail_on_gate and not summary['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
