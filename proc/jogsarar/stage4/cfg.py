from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, replace
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Stage4Cfg:
    in_raw_segy_root: Path = Path('/home/dcuser/data/ActiveSeisField/jogsarar')
    in_win512_segy_root: Path = Path(
        '/home/dcuser/data/ActiveSeisField/jogsarar_psn512'
    )
    out_pred_root: Path = Path('/home/dcuser/data/ActiveSeisField/jogsarar_psn512_pred')
    cfg_yaml: Path | None = Path('configs/config_convnext_prestage2_drop005.yaml')
    ckpt_path: Path | None = None
    standardize_eps: float = 1.0e-8
    segy_exts: tuple[str, ...] = ('.sgy', '.segy')
    device: str = 'cuda'
    p_class_index: int = 0
    up_factor: float = 2.0
    tile_h: int = 128
    tile_w: int = 512
    overlap_h: int = 96
    overlap_w: int = 0
    tiles_per_batch: int = 8
    use_amp: bool = True
    use_tqdm: bool = False
    rs_pre: int = 20
    rs_post: int = 20
    rs_max_lag: int = 4
    rs_k_neighbors: int = 5
    rs_n_iter: int = 1
    rs_mode: str = 'diff'
    rs_c_th: float = 0.8
    rs_smooth_method: str = 'wls'
    rs_lam: float = 5.0
    rs_subsample: bool = True
    rs_propagate_low_corr: bool = False
    rs_taper: str = 'hann'
    rs_taper_power: float = 1.0
    rs_lag_penalty: float = 0.10
    rs_lag_penalty_power: float = 1.0
    snap_mode: str = 'trough'
    snap_ltcor: int = 3
    log_gather_rs: bool = True
    post_trough_max_shift: int = 16
    post_trough_scan_ahead: int = 32
    post_trough_peak_search: str = 'after_pick'
    post_trough_smooth_win: int = 5
    post_trough_offs_abs_min_m: float | None = None
    post_trough_offs_abs_max_m: float | None = 1500
    post_trough_a_th: float = 0.03
    post_trough_outlier_radius: int = 4
    post_trough_outlier_min_support: int = 3
    post_trough_outlier_max_dev: int = 2
    post_trough_align_propagate_zero: bool = False
    post_trough_align_zero_pin_tol: int = 2
    post_trough_debug: bool = False
    post_trough_debug_max_examples: int = 5
    post_trough_debug_every_n_gathers: int = 10
    dt_tol_sec: float = 1e-9
    viz_every_n_shots: int = 20
    viz_dirname: str = 'viz'
    viz_plot_start: int = 0
    viz_plot_end: int = 1000
    viz_figsize: tuple[int, int] = (12, 9)
    viz_dpi: int = 200
    viz_gain: float = 2.0
    min_gather_h: int = 32
    edge_pick_max_gap_samples: int = 5


DEFAULT_STAGE4_CFG = Stage4Cfg()


def _parse_segy_exts(text: str) -> tuple[str, ...]:
    vals = [x.strip() for x in str(text).split(',')]
    out: list[str] = []
    for v in vals:
        if not v:
            continue
        e = v.lower()
        if not e.startswith('.'):
            e = '.' + e
        out.append(e)
    if len(out) == 0:
        msg = f'empty segy_exts: {text!r}'
        raise ValueError(msg)
    return tuple(out)


def _coerce_path_value(key: str, value: object, *, allow_none: bool) -> Path | None:
    if value is None:
        if allow_none:
            return None
        msg = f'config[{key}] must not be null'
        raise TypeError(msg)
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        return Path(value)
    msg = f'config[{key}] must be str or Path, got {type(value).__name__}'
    raise TypeError(msg)


def _coerce_segy_exts_value(value: object) -> tuple[str, ...]:
    if isinstance(value, tuple):
        return _parse_segy_exts(','.join(str(x) for x in value))
    if isinstance(value, list):
        if len(value) == 0:
            raise ValueError('config[segy_exts] must not be empty')
        for i, v in enumerate(value):
            if not isinstance(v, str):
                msg = f'config[segy_exts][{i}] must be str, got {type(v).__name__}'
                raise TypeError(msg)
        return _parse_segy_exts(','.join(value))
    if isinstance(value, str):
        return _parse_segy_exts(value)
    msg = f'config[segy_exts] must be str | list[str] | tuple[str,...], got {type(value).__name__}'
    raise TypeError(msg)


def _coerce_viz_figsize_value(value: object) -> tuple[int, int]:
    if isinstance(value, tuple) or isinstance(value, list):
        if len(value) != 2:
            raise ValueError(
                f'config[viz_figsize] must have length 2, got {len(value)}'
            )
        a, b = value[0], value[1]
        if (not isinstance(a, int)) or isinstance(a, bool):
            raise TypeError(
                f'config[viz_figsize][0] must be int, got {type(a).__name__}'
            )
        if (not isinstance(b, int)) or isinstance(b, bool):
            raise TypeError(
                f'config[viz_figsize][1] must be int, got {type(b).__name__}'
            )
        return (int(a), int(b))
    msg = f'config[viz_figsize] must be list[int,int] or tuple[int,int], got {type(value).__name__}'
    raise TypeError(msg)


def load_stage4_cfg_yaml(path: Path) -> Stage4Cfg:
    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.is_file():
        msg = f'--config not found: {cfg_path}'
        raise FileNotFoundError(msg)

    with cfg_path.open('r', encoding='utf-8') as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        msg = f'config top-level must be dict, got {type(loaded).__name__}'
        raise TypeError(msg)

    if 'stage4' in loaded:
        sub = loaded['stage4']
        if not isinstance(sub, dict):
            msg = f'config.stage4 must be dict, got {type(sub).__name__}'
            raise TypeError(msg)
        data = sub
    else:
        data = loaded

    allowed = {f.name for f in fields(Stage4Cfg)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        msg = f'unknown stage4 config keys: {unknown}'
        raise ValueError(msg)

    overrides: dict[str, object] = {}
    path_keys = {
        'in_raw_segy_root',
        'in_win512_segy_root',
        'out_pred_root',
        'cfg_yaml',
        'ckpt_path',
    }
    opt_path_keys = {'cfg_yaml', 'ckpt_path'}

    for key, value in data.items():
        if key in path_keys:
            overrides[key] = _coerce_path_value(
                key, value, allow_none=(key in opt_path_keys)
            )
            continue
        if key == 'segy_exts':
            overrides[key] = _coerce_segy_exts_value(value)
            continue
        if key == 'viz_figsize':
            overrides[key] = _coerce_viz_figsize_value(value)
            continue
        overrides[key] = value

    return replace(DEFAULT_STAGE4_CFG, **overrides)


def _build_stage4_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Stage4: PSN512 infer -> raw pick pipeline')
    p.add_argument(
        '--config',
        type=Path,
        default=None,
        help='Stage4 YAML config. CLI overrides YAML.',
    )
    p.add_argument('--in-raw-segy-root', type=Path, default=None)
    p.add_argument('--in-win512-segy-root', type=Path, default=None)
    p.add_argument('--out-pred-root', type=Path, default=None)
    p.add_argument(
        '--cfg-yaml',
        type=Path,
        default=None,
        help='PSN training config YAML (model definition)',
    )
    p.add_argument(
        '--ckpt-path',
        type=Path,
        default=None,
        help='Explicit checkpoint path (best.pt etc.)',
    )
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--segy-exts', type=str, default=None, help='e.g. ".sgy,.segy"')
    p.add_argument(
        '--post-trough-peak-search', choices=('after_pick', 'before_pick'), default=None
    )
    p.add_argument('--viz-every-n-shots', type=int, default=None)
    return p


def _load_stage4_cfg_from_yaml(path: Path) -> Stage4Cfg:
    return load_stage4_cfg_yaml(path)


__all__ = [
    'DEFAULT_STAGE4_CFG',
    'Stage4Cfg',
    '_build_stage4_parser',
    '_coerce_path_value',
    '_coerce_segy_exts_value',
    '_coerce_viz_figsize_value',
    '_load_stage4_cfg_from_yaml',
    '_parse_segy_exts',
    'load_stage4_cfg_yaml',
]
