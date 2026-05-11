"""Run Arakawa fbpick coarse -> physics -> physical-center grstat export."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml

__all__ = ['main', 'run_pipeline']

DEFAULT_ARAKAWA_SEGY_DIR = Path('/home/dcuser/data/ActiveSeisField/Arakawa2026')


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_runtime() -> SimpleNamespace:
    from cli.run_fbpick_coarse_infer import run_pipeline as run_coarse_infer
    from cli.run_fbpick_physics import run_pipeline as run_physics
    from cli.run_fbpick_physics_qc import run_pipeline as run_physics_qc

    from seisai_engine.pipelines.common import load_cfg_with_base_dir, resolve_relpath
    from seisai_engine.pipelines.fbpick.common.path_naming import build_fbpick_tag
    from seisai_engine.pipelines.fbpick.export.grstat import export_robust_pick_to_grstat

    return SimpleNamespace(
        build_fbpick_tag=build_fbpick_tag,
        export_robust_pick_to_grstat=export_robust_pick_to_grstat,
        load_cfg_with_base_dir=load_cfg_with_base_dir,
        resolve_relpath=resolve_relpath,
        run_coarse_infer=run_coarse_infer,
        run_physics=run_physics,
        run_physics_qc=run_physics_qc,
    )


def _as_dict(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        msg = f'{name} must be dict'
        raise TypeError(msg)
    return value


def _get_first_path_str(
    cfg: dict[str, Any],
    *,
    keys: tuple[str, ...],
    required: bool = False,
) -> str | None:
    paths = _as_dict(cfg.get('paths'), name='paths')
    for key in keys:
        value = paths.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            msg = f'paths.{key} must be non-empty str'
            raise TypeError(msg)
        return value
    if required:
        msg = 'set one of ' + ', '.join('paths.' + key for key in keys)
        raise KeyError(msg)
    return None


def _resolve_segy_path(
    cfg: dict[str, Any],
    *,
    base_dir: Path,
    runtime: SimpleNamespace,
) -> str:
    paths = _as_dict(cfg.get('paths'), name='paths')
    segy_value = _get_first_path_str(
        cfg,
        keys=('sgy_file', 'segy_file', 'sgy_path', 'segy_path', 'sgy', 'segy'),
    )
    if segy_value is None:
        top_sgy = cfg.get('sgy')
        if isinstance(top_sgy, str) and top_sgy.strip():
            segy_value = top_sgy
        else:
            msg = (
                'set paths.sgy_file in the config; aliases paths.segy_file, '
                'paths.segy_path, paths.sgy, paths.segy, and top-level sgy also work'
            )
            raise KeyError(msg)

    raw_path = Path(segy_value).expanduser()
    segy_dir_value = paths.get('sgy_dir', paths.get('segy_dir'))
    if (
        not raw_path.is_absolute()
        and raw_path.parent == Path('.')
        and isinstance(segy_dir_value, str)
        and segy_dir_value.strip()
    ):
        segy_dir = Path(runtime.resolve_relpath(base_dir, segy_dir_value))
        return str((segy_dir / raw_path.name).resolve())

    if not raw_path.is_absolute() and raw_path.parent == Path('.'):
        candidate = DEFAULT_ARAKAWA_SEGY_DIR / raw_path.name
        if candidate.is_file():
            return str(candidate.resolve())

    return runtime.resolve_relpath(base_dir, segy_value)


def _get_evaluation_per_trace_flag(cfg: dict[str, Any]) -> bool:
    evaluation_cfg = _as_dict(cfg.get('evaluation'), name='evaluation')
    value = evaluation_cfg.get(
        'write_per_trace_csv',
        evaluation_cfg.get('save_per_trace_csv', False),
    )
    if not isinstance(value, bool):
        msg = 'evaluation.write_per_trace_csv must be bool'
        raise TypeError(msg)
    return value


def _resolve_optional_path(
    value: str | None,
    *,
    base_dir: Path,
    runtime: SimpleNamespace,
) -> str | None:
    if value is None:
        return None
    return runtime.resolve_relpath(base_dir, value)


def _write_yaml(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding='utf-8',
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        msg = f'template config must be dict: {path}'
        raise TypeError(msg)
    return data


def _pick_label(pick_key: str) -> str:
    label = pick_key
    if label.endswith('_i'):
        label = label[:-2]
    return label


def _bool_cfg(cfg: dict[str, Any], section: str, key: str, default: bool) -> bool:
    sec = _as_dict(cfg.get(section), name=section)
    value = sec.get(key, default)
    if not isinstance(value, bool):
        msg = f'{section}.{key} must be bool'
        raise TypeError(msg)
    return value


def _int_cfg(cfg: dict[str, Any], section: str, key: str, default: int) -> int:
    sec = _as_dict(cfg.get(section), name=section)
    value = sec.get(key, default)
    if not isinstance(value, int):
        msg = f'{section}.{key} must be int'
        raise TypeError(msg)
    return value


def _str_cfg(cfg: dict[str, Any], section: str, key: str, default: str) -> str:
    sec = _as_dict(cfg.get(section), name=section)
    value = sec.get(key, default)
    if not isinstance(value, str) or not value:
        msg = f'{section}.{key} must be non-empty str'
        raise TypeError(msg)
    return value


def _optional_float_cfg(cfg: dict[str, Any], section: str, key: str) -> float | None:
    sec = _as_dict(cfg.get(section), name=section)
    value = sec.get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        msg = f'{section}.{key} must be float or null'
        raise TypeError(msg)
    return float(value)


def _prepare_coarse_config(
    *,
    template_path: Path,
    segy_path: str,
    coarse_dir: Path,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    out = deepcopy(_load_yaml(template_path))
    paths = _as_dict(out.get('paths'), name='coarse_template.paths')
    paths['segy_files'] = [segy_path]
    paths['out_dir'] = str(coarse_dir)
    out['paths'] = paths

    coord_scale = _optional_float_cfg(cfg, 'dataset', 'coord_unit_scale_to_m')
    if coord_scale is not None:
        dataset = _as_dict(out.get('dataset'), name='coarse_template.dataset')
        dataset['coord_unit_scale_to_m'] = coord_scale
        out['dataset'] = dataset
    return out


def _prepare_physics_config(
    *,
    template_path: Path,
    coarse_npz_path: Path,
    robust_npz_path: Path,
) -> dict[str, Any]:
    out = deepcopy(_load_yaml(template_path))
    paths = _as_dict(out.get('paths'), name='physics_template.paths')
    paths['coarse_npz_path'] = str(coarse_npz_path)
    paths['out_path'] = str(robust_npz_path)
    out['paths'] = paths
    return out


def _get_optional_section_path_str(
    cfg: dict[str, Any],
    *,
    section: str,
    keys: tuple[str, ...],
) -> str | None:
    sec = _as_dict(cfg.get(section), name=section)
    for key in keys:
        value = sec.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            msg = f'{section}.{key} must be non-empty str'
            raise TypeError(msg)
        return value
    return None


def _infer_n_traces_from_npz(path: Path) -> int | None:
    try:
        with np.load(path) as payload:
            if 'n_traces' in payload.files:
                return int(np.asarray(payload['n_traces']).item())
            if 'trace_indices' in payload.files:
                return int(np.asarray(payload['trace_indices']).shape[0])
            for key in ('robust_pick_i', 'coarse_pick_i', 'physical_center_i'):
                if key in payload.files:
                    return int(np.asarray(payload[key]).shape[0])
    except Exception:
        return None
    return None


def _make_dummy_fb_npy(
    *,
    robust_npz_path: Path,
    coarse_npz_path: Path,
    out_path: Path,
) -> Path:
    n_traces = _infer_n_traces_from_npz(robust_npz_path)
    if n_traces is None:
        n_traces = _infer_n_traces_from_npz(coarse_npz_path)
    if n_traces is None:
        msg = (
            'cannot infer n_traces from robust or coarse npz; tried '
            f'{robust_npz_path} and {coarse_npz_path}'
        )
        raise KeyError(msg)
    if n_traces <= 0:
        msg = f'n_traces must be positive: {n_traces}'
        raise ValueError(msg)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing_ok = False
    if out_path.is_file():
        try:
            existing = np.asarray(np.load(out_path), dtype=np.int64)
            existing_ok = bool(existing.ndim == 1 and int(existing.shape[0]) == n_traces)
        except Exception:
            existing_ok = False
    if not existing_ok:
        np.save(out_path, np.full((n_traces,), -1, dtype=np.int64))
    return out_path


def _prepare_visualization_config(
    *,
    cfg: dict[str, Any],
    base_dir: Path,
    runtime: SimpleNamespace,
    segy_path: str,
    tag: str,
    work_dir: Path,
    coarse_dir: Path,
    robust_dir: Path,
    robust_npz_path: Path,
    coarse_npz_path: Path,
) -> tuple[dict[str, Any] | None, Path | None, Path | None]:
    vis_runner_cfg = _as_dict(cfg.get('visualization'), name='visualization')
    enabled = vis_runner_cfg.get('enabled', False)
    if not isinstance(enabled, bool):
        msg = 'visualization.enabled must be bool'
        raise TypeError(msg)
    if not enabled:
        return None, None, None

    paths_cfg = _as_dict(cfg.get('paths'), name='paths')
    qc_out_value = vis_runner_cfg.get('out_dir', paths_cfg.get('qc_dir', str(work_dir / 'qc')))
    if not isinstance(qc_out_value, str) or not qc_out_value:
        msg = 'visualization.out_dir must be str when provided'
        raise TypeError(msg)
    qc_out_dir = Path(runtime.resolve_relpath(base_dir, qc_out_value))

    fb_value = _get_optional_section_path_str(
        cfg,
        section='paths',
        keys=('fb_file', 'fb_path', 'fb_npy', 'fb'),
    )
    if fb_value is None:
        fb_value = _get_optional_section_path_str(
            cfg,
            section='visualization',
            keys=('fb_file', 'fb_path', 'fb_npy'),
        )

    if fb_value is not None:
        fb_path = Path(runtime.resolve_relpath(base_dir, fb_value))
    else:
        allow_no_fb = vis_runner_cfg.get('allow_no_fb', True)
        if not isinstance(allow_no_fb, bool):
            msg = 'visualization.allow_no_fb must be bool'
            raise TypeError(msg)
        if not allow_no_fb:
            msg = (
                'visualization is enabled but no FB npy file was provided; set '
                'paths.fb_file or visualization.allow_no_fb=true'
            )
            raise ValueError(msg)
        fb_dir_value = vis_runner_cfg.get('fb_dummy_dir', str(work_dir / 'fb_dummy'))
        if not isinstance(fb_dir_value, str) or not fb_dir_value:
            msg = 'visualization.fb_dummy_dir must be str when provided'
            raise TypeError(msg)
        fb_dir = Path(runtime.resolve_relpath(base_dir, fb_dir_value))
        fb_path = _make_dummy_fb_npy(
            robust_npz_path=robust_npz_path,
            coarse_npz_path=coarse_npz_path,
            out_path=fb_dir / f'{tag}.fb_none.npy',
        )

    dataset_src = _as_dict(cfg.get('dataset'), name='dataset')
    dataset_cfg: dict[str, Any] = {
        'primary_keys': dataset_src.get('primary_keys', ['ffid']),
        'infer_endian': dataset_src.get('infer_endian', 'big'),
        'use_header_cache': dataset_src.get('use_header_cache', True),
    }
    dataset_override = _as_dict(vis_runner_cfg.get('dataset'), name='visualization.dataset')
    dataset_cfg.update(dataset_override)

    overlays = {
        'coarse_pmax': True,
        'trend_center': True,
        'physical_center': True,
        'fine_center': False,
        'window': False,
        'final_pick': False,
        'physical_model_status': True,
    }
    overlays.update(_as_dict(vis_runner_cfg.get('overlays'), name='visualization.overlays'))

    flatten = {
        'enabled': True,
        'reference_key': 'physical_center_i',
        'half_samples': 256,
    }
    flatten.update(
        _as_dict(
            vis_runner_cfg.get('first_panel_flatten'),
            name='visualization.first_panel_flatten',
        )
    )

    vis_cfg: dict[str, Any] = {
        'waveform_norm': vis_runner_cfg.get('waveform_norm', 'per_trace'),
        'clip_percentile': vis_runner_cfg.get('clip_percentile', 99.0),
        'max_gathers_per_file': vis_runner_cfg.get('max_gathers_per_file', 10),
        'gather_selection': vis_runner_cfg.get('gather_selection', 'even'),
        'first_panel_only': vis_runner_cfg.get('first_panel_only', True),
        'max_traces_per_gather': vis_runner_cfg.get('max_traces_per_gather', 10000),
        'save_cdf': vis_runner_cfg.get('save_cdf', False),
        'save_summary_csv': vis_runner_cfg.get('save_summary_csv', True),
        'skip_gather_keys': vis_runner_cfg.get('skip_gather_keys', {}),
        'overlays': overlays,
        'first_panel_flatten': flatten,
    }

    return (
        {
            'paths': {
                'segy_files': [segy_path],
                'fb_files': [str(fb_path)],
                'coarse_npz_dir': str(coarse_dir),
                'robust_npz_dir': str(robust_dir),
                'out_dir': str(qc_out_dir),
            },
            'dataset': dataset_cfg,
            'vis': vis_cfg,
        },
        qc_out_dir,
        fb_path,
    )


def run_pipeline(config_path: str | Path) -> Path:
    runtime = _load_runtime()
    cfg, base_dir = runtime.load_cfg_with_base_dir(Path(config_path))
    root = _repo_root()

    segy_path = _resolve_segy_path(cfg, base_dir=base_dir, runtime=runtime)
    if not Path(segy_path).is_file():
        raise FileNotFoundError(segy_path)

    paths = _as_dict(cfg.get('paths'), name='paths')
    work_dir_value = paths.get('work_dir', str(root / 'proc' / 'arakawa'))
    if not isinstance(work_dir_value, str) or not work_dir_value:
        msg = 'paths.work_dir must be str when provided'
        raise TypeError(msg)
    work_dir = Path(runtime.resolve_relpath(base_dir, work_dir_value))

    coarse_template_value = paths.get(
        'coarse_template', str(root / 'proc' / 'arakawa' / 'configs' / 'coarse_one.yaml')
    )
    physics_template_value = paths.get(
        'physics_template',
        str(root / 'proc' / 'arakawa' / 'configs' / 'physics_one.yaml'),
    )
    if not isinstance(coarse_template_value, str) or not isinstance(
        physics_template_value, str
    ):
        msg = 'paths.coarse_template and paths.physics_template must be str'
        raise TypeError(msg)
    coarse_template = Path(runtime.resolve_relpath(base_dir, coarse_template_value))
    physics_template = Path(runtime.resolve_relpath(base_dir, physics_template_value))

    tag = runtime.build_fbpick_tag(segy_path)
    coarse_dir = Path(
        runtime.resolve_relpath(base_dir, paths.get('coarse_dir', str(work_dir / 'coarse')))
    )
    robust_dir = Path(
        runtime.resolve_relpath(base_dir, paths.get('robust_dir', str(work_dir / 'robust')))
    )
    grstat_dir = Path(
        runtime.resolve_relpath(base_dir, paths.get('grstat_dir', str(work_dir / 'grstat')))
    )
    eval_dir = Path(
        runtime.resolve_relpath(base_dir, paths.get('eval_dir', str(work_dir / 'eval')))
    )
    generated_cfg_dir = Path(
        runtime.resolve_relpath(
            base_dir, paths.get('generated_config_dir', str(work_dir / 'generated_configs'))
        )
    )

    for d in (coarse_dir, robust_dir, grstat_dir, eval_dir, generated_cfg_dir):
        d.mkdir(parents=True, exist_ok=True)

    coarse_npz_path = coarse_dir / f'{tag}.coarse.npz'
    robust_npz_path = robust_dir / f'{tag}.robust.npz'

    pick_key = _str_cfg(cfg, 'export', 'pick_key', 'physical_center_i')
    phase_mode = _str_cfg(cfg, 'export', 'phase_mode', 'peak')
    max_shift_samples = _int_cfg(cfg, 'export', 'max_shift_samples', 2)
    out_label = f'{tag}.{_pick_label(pick_key)}.snap_{phase_mode}.ltcor{max_shift_samples}'
    out_crd_value = paths.get('out_crd', str(grstat_dir / f'{out_label}.crd'))
    out_npz_value = paths.get('out_npz', str(grstat_dir / f'{out_label}.npz'))
    if not isinstance(out_crd_value, str) or not isinstance(out_npz_value, str):
        msg = 'paths.out_crd and paths.out_npz must be str when provided'
        raise TypeError(msg)
    out_crd = Path(runtime.resolve_relpath(base_dir, out_crd_value))
    out_npz = Path(runtime.resolve_relpath(base_dir, out_npz_value))

    run_force = _bool_cfg(cfg, 'run', 'force', False)
    force_coarse = _bool_cfg(cfg, 'run', 'force_coarse', run_force)
    force_physics = _bool_cfg(cfg, 'run', 'force_physics', run_force)
    force_export = _bool_cfg(cfg, 'run', 'force_export', True if run_force else False)

    coarse_cfg_path = generated_cfg_dir / f'{tag}.coarse.yaml'
    physics_cfg_path = generated_cfg_dir / f'{tag}.physics.yaml'
    qc_cfg_path = generated_cfg_dir / f'{tag}.physics_qc.yaml'

    coarse_cfg = _prepare_coarse_config(
        template_path=coarse_template,
        segy_path=segy_path,
        coarse_dir=coarse_dir,
        cfg=cfg,
    )
    _write_yaml(coarse_cfg_path, coarse_cfg)

    if coarse_npz_path.is_file() and not force_coarse:
        print(f'[skip] coarse exists: {coarse_npz_path}')
    else:
        print(f'[run] coarse infer: {coarse_cfg_path}')
        result = runtime.run_coarse_infer(coarse_cfg_path)
        if Path(result) != coarse_npz_path:
            msg = f'coarse output mismatch: {result} != {coarse_npz_path}'
            raise RuntimeError(msg)

    physics_cfg = _prepare_physics_config(
        template_path=physics_template,
        coarse_npz_path=coarse_npz_path,
        robust_npz_path=robust_npz_path,
    )
    _write_yaml(physics_cfg_path, physics_cfg)

    if robust_npz_path.is_file() and not force_physics:
        print(f'[skip] physics exists: {robust_npz_path}')
    else:
        print(f'[run] physics: {physics_cfg_path}')
        result = runtime.run_physics(physics_cfg_path)
        if Path(result) != robust_npz_path:
            msg = f'physics output mismatch: {result} != {robust_npz_path}'
            raise RuntimeError(msg)

    reference_grstat = _resolve_optional_path(
        _get_first_path_str(
            cfg,
            keys=('reference_grstat_path', 'reference_grstat'),
        ),
        base_dir=base_dir,
        runtime=runtime,
    )
    evaluation_cfg = _as_dict(cfg.get('evaluation'), name='evaluation')
    eval_enabled_raw = evaluation_cfg.get('enabled', reference_grstat is not None)
    if not isinstance(eval_enabled_raw, bool):
        msg = 'evaluation.enabled must be bool'
        raise TypeError(msg)
    eval_enabled = eval_enabled_raw and reference_grstat is not None

    eval_summary_json = eval_dir / f'{out_label}.eval_summary.json'
    eval_summary_csv = eval_dir / f'{out_label}.eval_summary.csv'
    eval_per_trace_csv = eval_dir / f'{out_label}.eval_per_trace.csv'
    write_per_trace_csv = _get_evaluation_per_trace_flag(cfg)
    if 'summary_json' in evaluation_cfg:
        value = evaluation_cfg['summary_json']
        if not isinstance(value, str):
            msg = 'evaluation.summary_json must be str'
            raise TypeError(msg)
        eval_summary_json = Path(runtime.resolve_relpath(base_dir, value))
    if 'summary_csv' in evaluation_cfg:
        value = evaluation_cfg['summary_csv']
        if not isinstance(value, str):
            msg = 'evaluation.summary_csv must be str'
            raise TypeError(msg)
        eval_summary_csv = Path(runtime.resolve_relpath(base_dir, value))
    if 'per_trace_csv' in evaluation_cfg:
        value = evaluation_cfg['per_trace_csv']
        if not isinstance(value, str):
            msg = 'evaluation.per_trace_csv must be str'
            raise TypeError(msg)
        eval_per_trace_csv = Path(runtime.resolve_relpath(base_dir, value))

    skip_export = (
        out_crd.is_file()
        and out_npz.is_file()
        and not force_export
        and not eval_enabled
    )
    if skip_export:
        print(f'[skip] export exists: {out_crd}')
        summary = {'out_crd': str(out_crd), 'out_npz': str(out_npz), 'skipped': True}
    else:
        print(f'[run] export grstat: {out_crd}')
        summary = runtime.export_robust_pick_to_grstat(
            segy_path=segy_path,
            robust_npz_path=robust_npz_path,
            out_crd_path=out_crd,
            out_npz_path=out_npz,
            pick_key=pick_key,
            phase_mode=phase_mode,  # type: ignore[arg-type]
            max_shift_samples=max_shift_samples,
            endian=_str_cfg(cfg, 'export', 'endian', 'big'),  # type: ignore[arg-type]
            header_comment=_str_cfg(
                cfg, 'export', 'header_comment', 'Arakawa physical_center_i snap to phase'
            ),
            duplicate_policy=_str_cfg(cfg, 'export', 'duplicate_policy', 'error'),  # type: ignore[arg-type]
            dt_multiplier=_optional_float_cfg(cfg, 'export', 'dt_multiplier'),
            unbounded_zero_crossing=_bool_cfg(
                cfg, 'export', 'unbounded_zero_crossing', False
            ),
            strict_trace_count=_bool_cfg(cfg, 'export', 'strict_trace_count', True),
            strict_sample_count=_bool_cfg(cfg, 'export', 'strict_sample_count', True),
            reference_grstat_path=reference_grstat if eval_enabled else None,
            eval_summary_json_path=eval_summary_json if eval_enabled else None,
            eval_summary_csv_path=eval_summary_csv if eval_enabled else None,
            eval_per_trace_csv_path=(
                eval_per_trace_csv if eval_enabled and write_per_trace_csv else None
            ),
            eval_strict_gather_numbers=_bool_cfg(
                cfg, 'evaluation', 'strict_gather_numbers', True
            ),
            eval_strict_shape=_bool_cfg(cfg, 'evaluation', 'strict_shape', True),
        )

    visualization_cfg, visualization_out_dir, visualization_fb_path = (
        _prepare_visualization_config(
            cfg=cfg,
            base_dir=base_dir,
            runtime=runtime,
            segy_path=segy_path,
            tag=tag,
            work_dir=work_dir,
            coarse_dir=coarse_dir,
            robust_dir=robust_dir,
            robust_npz_path=robust_npz_path,
            coarse_npz_path=coarse_npz_path,
        )
    )
    visualization_cfg_path: Path | None = None
    visualization_summary_path: Path | None = None
    if visualization_cfg is not None:
        visualization_cfg_path = qc_cfg_path
        _write_yaml(visualization_cfg_path, visualization_cfg)
        print(f'[run] visualization QC: {visualization_cfg_path}')
        visualization_summary_path = Path(runtime.run_physics_qc(visualization_cfg_path))

    run_summary = {
        'tag': tag,
        'segy': segy_path,
        'coarse_npz': str(coarse_npz_path),
        'robust_npz': str(robust_npz_path),
        'out_crd': str(out_crd),
        'out_npz': str(out_npz),
        'coarse_config': str(coarse_cfg_path),
        'physics_config': str(physics_cfg_path),
        'visualization_enabled': visualization_cfg is not None,
        'visualization_config': (
            str(visualization_cfg_path) if visualization_cfg_path is not None else ''
        ),
        'visualization_out_dir': (
            str(visualization_out_dir) if visualization_out_dir is not None else ''
        ),
        'visualization_fb_file': (
            str(visualization_fb_path) if visualization_fb_path is not None else ''
        ),
        'visualization_summary_csv': (
            str(visualization_summary_path) if visualization_summary_path is not None else ''
        ),
        'reference_grstat': reference_grstat or '',
        'evaluation_enabled': eval_enabled,
        'evaluation_write_per_trace_csv': bool(write_per_trace_csv),
        'export_summary': summary,
    }
    summary_path = work_dir / f'{tag}.arakawa_physical_export_summary.json'
    summary_path.write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding='utf-8',
    )
    print(json.dumps(run_summary, ensure_ascii=False, indent=2, sort_keys=True))
    return out_crd


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args(argv)
    run_pipeline(args.config)


if __name__ == '__main__':
    main()
