from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

from config_io import (
    build_yaml_defaults,
    coerce_optional_bool,
    coerce_optional_float,
    coerce_optional_int,
    coerce_path,
    load_yaml_dict,
    normalize_segy_exts,
)


@dataclass(frozen=True)
class Stage1Cfg:
    in_segy_root: Path = Path('/home/dcuser/data/ActiveSeisField/jogsarar')
    out_infer_root: Path = Path('/home/dcuser/data/ActiveSeisField/jogsarar_out')
    weights_path: Path = Path('/home/dcuser/data/model_weight/fbseg_caformer_b36.pth')
    segy_exts: tuple[str, ...] = ('.sgy', '.segy')
    recursive: bool = False
    backbone: str = 'caformer_b36.sail_in22k_ft_in1k'
    device: str = 'cuda'
    use_tta: bool = True
    pmax_th: float = 0.0
    ltcor: int = 5
    segy_endian: str = 'big'
    waveform_mode: str = 'mmap'
    header_cache_dir: str | None = None
    viz_every_n_shots: int = 100
    viz_dirname: str = 'viz'
    vmin_mask: float = 100.0
    vmax_mask: float = 5000.0
    t0_lo_ms: float = -10.0
    t0_hi_ms: float = 100.0
    taper_ms: float = 10.0
    tile_h: int = 128
    tile_w: int = 6016
    overlap_h: int = 96
    tiles_per_batch: int = 8
    polarity_flip: bool = True
    lmo_vel_mps: float = 3200.0
    lmo_bulk_shift_samples: float = 50.0
    plot_start: int = 0
    plot_end: int = 350
    viz_score_components: bool = True
    viz_score_style: str = 'bar'
    viz_conf_prob_scale_enable: bool = True
    viz_conf_prob_pct_lo: float = 5.0
    viz_conf_prob_pct_hi: float = 99.0
    viz_conf_prob_pct_eps: float = 1e-12
    viz_ymax_conf_prob: float | None = 1.0
    viz_ymax_conf_trend: float | None = 1.0
    viz_ymax_conf_rs: float | None = 1.0
    viz_trend_line_enable: bool = True
    viz_trend_line_lw: float = 1.6
    viz_trend_line_alpha: float = 0.9
    viz_trend_line_label: str = 'trend'
    viz_trend_line_color: str = 'g'
    use_residual_statics: bool = True
    rs_base_pick: str = 'snap'
    rs_pre_snap_mode: str = 'trough'
    rs_pre_snap_ltcor: int = 3
    rs_pre_samples: int = 20
    rs_post_samples: int = 20
    rs_max_lag: int = 8
    rs_k_neighbors: int = 5
    rs_n_iter: int = 2
    rs_mode: str = 'diff'
    rs_c_th: float = 0.5
    rs_smooth_method: str = 'wls'
    rs_lam: float = 5.0
    rs_subsample: bool = True
    rs_propagate_low_corr: bool = False
    rs_taper: str = 'hann'
    rs_taper_power: float = 1.0
    rs_lag_penalty: float = 0.10
    rs_lag_penalty_power: float = 1.0
    use_final_snap: bool = True
    final_snap_mode: str = 'trough'
    final_snap_ltcor: int = 3
    conf_enable: bool = True
    conf_viz_enable: bool = True
    conf_viz_ffid: int = 2147
    conf_half_win: int = 20
    trend_local_enable: bool = True
    trend_local_use_abs_offset: bool = False
    trend_local_sort_offsets: bool = False
    trend_side_split_enable: bool = True
    trend_local_use_abs_offset_header: bool = True
    trend_local_section_len: int = 16
    trend_local_stride: int = 4
    trend_local_huber_c: float = 1.345
    trend_local_iters: int = 3
    trend_local_vmin_mps: float | None = 300.0
    trend_local_vmax_mps: float | None = 8000.0
    trend_local_weight_mode: str = 'uniform'
    trend_sigma_ms: float = 6.0
    trend_min_pts: int = 12
    trend_var_half_win_traces: int = 8
    trend_var_sigma_std_ms: float = 6.0
    trend_var_min_count: int = 3
    rs_cmax_th: float = 0.5
    rs_abs_lag_soft: float = 8.0
    save_trend_to_npz: bool = True
    trend_source_label: str = 'pick_final'
    trend_method_label: str = 'local_irls_split_sides'


DEFAULT_STAGE1_CFG = Stage1Cfg()

INPUT_DIR = DEFAULT_STAGE1_CFG.in_segy_root
OUT_DIR = DEFAULT_STAGE1_CFG.out_infer_root
WEIGHTS_PATH = DEFAULT_STAGE1_CFG.weights_path
VIZ_EVERY_N_SHOTS = DEFAULT_STAGE1_CFG.viz_every_n_shots
VIZ_DIRNAME = DEFAULT_STAGE1_CFG.viz_dirname

_STAGE1_CFG_KEYS = tuple(Stage1Cfg.__dataclass_fields__.keys())
_PATH_KEYS = {'in_segy_root', 'out_infer_root', 'weights_path'}
_OPTIONAL_FLOAT_KEYS = {
    'viz_ymax_conf_prob',
    'viz_ymax_conf_trend',
    'viz_ymax_conf_rs',
    'trend_local_vmin_mps',
    'trend_local_vmax_mps',
}
_OPTIONAL_STR_KEYS = {'header_cache_dir'}


def _coerce_required_int(key: str, value: object) -> int:
    out = coerce_optional_int(key, value)
    if out is None:
        msg = f'config[{key}] must not be null'
        raise TypeError(msg)
    return int(out)


def _coerce_required_bool(key: str, value: object) -> bool:
    out = coerce_optional_bool(key, value)
    if out is None:
        msg = f'config[{key}] must not be null'
        raise TypeError(msg)
    return bool(out)


def _coerce_required_float(key: str, value: object) -> float:
    out = coerce_optional_float(key, value)
    if out is None:
        msg = f'config[{key}] must not be null'
        raise TypeError(msg)
    return float(out)


def _coerce_optional_float_field(key: str, value: object) -> float | None:
    out = coerce_optional_float(key, value)
    if out is None:
        return None
    return float(out)


def _coerce_required_str(key: str, value: object) -> str:
    if not isinstance(value, str):
        msg = f'config[{key}] must be str, got {type(value).__name__}'
        raise TypeError(msg)
    if value == '':
        msg = f'config[{key}] must not be empty'
        raise ValueError(msg)
    return value


def _coerce_optional_str_field(key: str, value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        msg = f'config[{key}] must be str or null, got {type(value).__name__}'
        raise TypeError(msg)
    return value


def _build_stage1_yaml_coercers() -> dict[str, Callable[[object], object]]:
    coercers: dict[str, Callable[[object], object]] = {}
    for key in _STAGE1_CFG_KEYS:
        if key in _PATH_KEYS:
            coercers[key] = partial(coerce_path, key, allow_none=False)
            continue
        if key == 'segy_exts':
            coercers[key] = normalize_segy_exts
            continue
        if key in _OPTIONAL_FLOAT_KEYS:
            coercers[key] = partial(_coerce_optional_float_field, key)
            continue
        if key in _OPTIONAL_STR_KEYS:
            coercers[key] = partial(_coerce_optional_str_field, key)
            continue

        default_value = getattr(DEFAULT_STAGE1_CFG, key)
        if isinstance(default_value, bool):
            coercers[key] = partial(_coerce_required_bool, key)
            continue
        if isinstance(default_value, int):
            coercers[key] = partial(_coerce_required_int, key)
            continue
        if isinstance(default_value, float):
            coercers[key] = partial(_coerce_required_float, key)
            continue
        if isinstance(default_value, str):
            coercers[key] = partial(_coerce_required_str, key)
            continue

        msg = f'unsupported stage1 config key type: key={key}'
        raise RuntimeError(msg)
    return coercers


def _load_yaml_defaults(config_path: Path) -> dict[str, object]:
    loaded = load_yaml_dict(config_path)
    return build_yaml_defaults(
        loaded,
        allowed_keys=set(_STAGE1_CFG_KEYS),
        coercers=_build_stage1_yaml_coercers(),
    )


def _validate_stage1_cfg(cfg: Stage1Cfg) -> Stage1Cfg:
    segy_exts = normalize_segy_exts(list(cfg.segy_exts))
    cfg = replace(cfg, segy_exts=segy_exts)

    if cfg.device == '':
        raise ValueError('device must not be empty')
    if cfg.viz_dirname == '':
        raise ValueError('viz_dirname must not be empty')
    if cfg.segy_endian not in {'big', 'little'}:
        raise ValueError(
            f"segy_endian must be 'big' or 'little', got {cfg.segy_endian!r}"
        )
    if cfg.waveform_mode not in {'mmap', 'eager'}:
        raise ValueError(
            f"waveform_mode must be 'mmap' or 'eager', got {cfg.waveform_mode!r}"
        )
    if cfg.viz_score_style not in {'bar', 'line'}:
        raise ValueError(
            f"viz_score_style must be 'bar' or 'line', got {cfg.viz_score_style!r}"
        )
    if cfg.rs_base_pick not in {'pre', 'snap'}:
        raise ValueError(
            f"rs_base_pick must be 'pre' or 'snap', got {cfg.rs_base_pick!r}"
        )
    if cfg.rs_mode not in {'diff', 'raw'}:
        raise ValueError(f"rs_mode must be 'diff' or 'raw', got {cfg.rs_mode!r}")
    if cfg.trend_local_weight_mode not in {'uniform', 'pmax'}:
        raise ValueError(
            f"trend_local_weight_mode must be 'uniform' or 'pmax', got {cfg.trend_local_weight_mode!r}"
        )

    if cfg.tile_h <= 0:
        raise ValueError(f'tile_h must be > 0, got {cfg.tile_h}')
    if cfg.tile_w <= 0:
        raise ValueError(f'tile_w must be > 0, got {cfg.tile_w}')
    if cfg.overlap_h < 0:
        raise ValueError(f'overlap_h must be >= 0, got {cfg.overlap_h}')
    if cfg.tiles_per_batch <= 0:
        raise ValueError(f'tiles_per_batch must be > 0, got {cfg.tiles_per_batch}')
    if cfg.viz_every_n_shots < 0:
        raise ValueError(f'viz_every_n_shots must be >= 0, got {cfg.viz_every_n_shots}')
    if cfg.plot_end <= cfg.plot_start:
        raise ValueError(
            f'plot_end must be > plot_start, got start={cfg.plot_start}, end={cfg.plot_end}'
        )
    if cfg.vmax_mask <= 0.0:
        raise ValueError(f'vmax_mask must be > 0, got {cfg.vmax_mask}')
    if cfg.vmin_mask < 0.0:
        raise ValueError(f'vmin_mask must be >= 0, got {cfg.vmin_mask}')
    if cfg.vmax_mask < cfg.vmin_mask:
        raise ValueError(
            f'vmax_mask must be >= vmin_mask, got vmin={cfg.vmin_mask}, vmax={cfg.vmax_mask}'
        )
    if cfg.conf_half_win < 0:
        raise ValueError(f'conf_half_win must be >= 0, got {cfg.conf_half_win}')
    if cfg.trend_min_pts < 0:
        raise ValueError(f'trend_min_pts must be >= 0, got {cfg.trend_min_pts}')
    if cfg.rs_abs_lag_soft < 0.0:
        raise ValueError(f'rs_abs_lag_soft must be >= 0, got {cfg.rs_abs_lag_soft}')

    return cfg


def load_stage1_cfg_yaml(config_path: Path) -> Stage1Cfg:
    updates = _load_yaml_defaults(config_path)
    cfg = replace(DEFAULT_STAGE1_CFG, **updates)
    return _validate_stage1_cfg(cfg)


def _write_stage1_used_cfg_yaml(cfg: Stage1Cfg, *, out_root: Path) -> Path:
    serializable: dict[str, object] = {}
    for key in _STAGE1_CFG_KEYS:
        value = getattr(cfg, key)
        if isinstance(value, Path):
            serializable[key] = str(value)
            continue
        if isinstance(value, tuple):
            serializable[key] = [str(x) for x in value]
            continue
        serializable[key] = value

    target_root = Path(out_root).expanduser().resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    out_path = target_root / 'stage1_used.yaml'

    import yaml

    with out_path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(
            serializable,
            f,
            sort_keys=True,
        )
    return out_path


def _cfg_from_namespace(args: argparse.Namespace) -> Stage1Cfg:
    updates: dict[str, object] = {}
    for key in _STAGE1_CFG_KEYS:
        value = getattr(args, key)
        if value is None:
            continue
        if key == 'segy_exts':
            if isinstance(value, tuple):
                updates[key] = tuple(str(x) for x in value)
            else:
                updates[key] = normalize_segy_exts(value)
            continue
        if key in _PATH_KEYS:
            updates[key] = Path(value)
            continue
        updates[key] = value
    cfg = replace(DEFAULT_STAGE1_CFG, **updates)
    return _validate_stage1_cfg(cfg)


__all__ = [
    'DEFAULT_STAGE1_CFG',
    'INPUT_DIR',
    'OUT_DIR',
    'Stage1Cfg',
    'VIZ_DIRNAME',
    'VIZ_EVERY_N_SHOTS',
    'WEIGHTS_PATH',
    '_PATH_KEYS',
    '_STAGE1_CFG_KEYS',
    '_cfg_from_namespace',
    '_load_yaml_defaults',
    '_validate_stage1_cfg',
    '_write_stage1_used_cfg_yaml',
    'load_stage1_cfg_yaml',
]
