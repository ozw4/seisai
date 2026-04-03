from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from seisai_utils.config import optional_bool, optional_float, optional_int, optional_str

__all__ = [
    'DEFAULT_PHYSICS_LITE_CONFIG',
    'PhysicsFeasibleBandCfg',
    'PhysicsKeepRejectCfg',
    'PhysicsLiteConfig',
    'PhysicsResidualStaticsCfg',
    'PhysicsRobustCenterCfg',
    'PhysicsTrendCfg',
    'load_physics_lite_config',
    'physics_lite_config_to_dict',
]


@dataclass(frozen=True)
class PhysicsFeasibleBandCfg:
    vmin_mask: float = 100.0
    vmax_mask: float = 5000.0
    t0_lo_ms: float = -10.0
    t0_hi_ms: float = 100.0
    taper_ms: float = 10.0


@dataclass(frozen=True)
class PhysicsTrendCfg:
    trend_local_section_len: int = 16
    trend_local_stride: int = 4
    trend_local_huber_c: float = 1.345
    trend_local_iters: int = 3
    trend_local_vmin_mps: float = 300.0
    trend_local_vmax_mps: float = 8000.0
    trend_sigma_ms: float = 6.0
    trend_min_pts: int = 12
    trend_var_half_win_traces: int = 8
    trend_var_sigma_std_ms: float = 6.0
    trend_var_min_count: int = 3


@dataclass(frozen=True)
class PhysicsResidualStaticsCfg:
    use_residual_statics: bool = True
    rs_pre_snap_mode: str = 'trough'
    rs_pre_samples: int = 20
    rs_post_samples: int = 20
    rs_max_lag: int = 8
    rs_k_neighbors: int = 5
    rs_n_iter: int = 2
    rs_c_th: float = 0.5
    use_final_snap: bool = True
    final_snap_mode: str = 'trough'
    final_snap_ltcor: int = 3


@dataclass(frozen=True)
class PhysicsKeepRejectCfg:
    drop_low_frac: float = 0.05


@dataclass(frozen=True)
class PhysicsRobustCenterCfg:
    half_win: int = 128
    local_global_diff_th_samples: int = 128
    local_discard_radius_traces: int = 32
    local_inv_drop_th_samples: float = 10.0
    local_inv_min_consec_steps: int = 2
    global_vmin_m_s: float = 300.0
    global_vmax_m_s: float = 6000.0
    global_side_min_pts: int = 16


@dataclass(frozen=True)
class PhysicsLiteConfig:
    feasible_band: PhysicsFeasibleBandCfg = PhysicsFeasibleBandCfg()
    trend: PhysicsTrendCfg = PhysicsTrendCfg()
    residual_statics: PhysicsResidualStaticsCfg = PhysicsResidualStaticsCfg()
    keep_reject: PhysicsKeepRejectCfg = PhysicsKeepRejectCfg()
    robust_center: PhysicsRobustCenterCfg = PhysicsRobustCenterCfg()


DEFAULT_PHYSICS_LITE_CONFIG = PhysicsLiteConfig()


def _require_dict(value: object, *, key: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        msg = f'{key} must be dict'
        raise TypeError(msg)
    return value


def _validate_mode(name: str, value: str) -> str:
    if value not in {'trough', 'peak'}:
        msg = f"{name} must be 'trough' or 'peak', got {value!r}"
        raise ValueError(msg)
    return value


def _validate_positive_float(name: str, value: float) -> float:
    out = float(value)
    if out <= 0.0:
        msg = f'{name} must be > 0'
        raise ValueError(msg)
    return out


def _validate_nonnegative_float(name: str, value: float) -> float:
    out = float(value)
    if out < 0.0:
        msg = f'{name} must be >= 0'
        raise ValueError(msg)
    return out


def _validate_positive_int(name: str, value: int) -> int:
    out = int(value)
    if out <= 0:
        msg = f'{name} must be > 0'
        raise ValueError(msg)
    return out


def _validate_nonnegative_int(name: str, value: int) -> int:
    out = int(value)
    if out < 0:
        msg = f'{name} must be >= 0'
        raise ValueError(msg)
    return out


def _load_feasible_band_cfg(cfg: dict[str, Any]) -> PhysicsFeasibleBandCfg:
    return PhysicsFeasibleBandCfg(
        vmin_mask=float(optional_float(cfg, 'vmin_mask', 100.0)),
        vmax_mask=float(optional_float(cfg, 'vmax_mask', 5000.0)),
        t0_lo_ms=float(optional_float(cfg, 't0_lo_ms', -10.0)),
        t0_hi_ms=float(optional_float(cfg, 't0_hi_ms', 100.0)),
        taper_ms=float(optional_float(cfg, 'taper_ms', 10.0)),
    )


def _load_trend_cfg(cfg: dict[str, Any]) -> PhysicsTrendCfg:
    return PhysicsTrendCfg(
        trend_local_section_len=int(optional_int(cfg, 'trend_local_section_len', 16)),
        trend_local_stride=int(optional_int(cfg, 'trend_local_stride', 4)),
        trend_local_huber_c=float(optional_float(cfg, 'trend_local_huber_c', 1.345)),
        trend_local_iters=int(optional_int(cfg, 'trend_local_iters', 3)),
        trend_local_vmin_mps=float(optional_float(cfg, 'trend_local_vmin_mps', 300.0)),
        trend_local_vmax_mps=float(optional_float(cfg, 'trend_local_vmax_mps', 8000.0)),
        trend_sigma_ms=float(optional_float(cfg, 'trend_sigma_ms', 6.0)),
        trend_min_pts=int(optional_int(cfg, 'trend_min_pts', 12)),
        trend_var_half_win_traces=int(optional_int(cfg, 'trend_var_half_win_traces', 8)),
        trend_var_sigma_std_ms=float(
            optional_float(cfg, 'trend_var_sigma_std_ms', 6.0)
        ),
        trend_var_min_count=int(optional_int(cfg, 'trend_var_min_count', 3)),
    )


def _load_residual_statics_cfg(cfg: dict[str, Any]) -> PhysicsResidualStaticsCfg:
    rs_pre_snap_mode = optional_str(cfg, 'rs_pre_snap_mode', 'trough')
    final_snap_mode = optional_str(cfg, 'final_snap_mode', 'trough')
    if rs_pre_snap_mode is None or final_snap_mode is None:
        msg = 'snap mode fields must not be null'
        raise TypeError(msg)
    return PhysicsResidualStaticsCfg(
        use_residual_statics=bool(optional_bool(cfg, 'use_residual_statics', default=True)),
        rs_pre_snap_mode=str(rs_pre_snap_mode),
        rs_pre_samples=int(optional_int(cfg, 'rs_pre_samples', 20)),
        rs_post_samples=int(optional_int(cfg, 'rs_post_samples', 20)),
        rs_max_lag=int(optional_int(cfg, 'rs_max_lag', 8)),
        rs_k_neighbors=int(optional_int(cfg, 'rs_k_neighbors', 5)),
        rs_n_iter=int(optional_int(cfg, 'rs_n_iter', 2)),
        rs_c_th=float(optional_float(cfg, 'rs_c_th', 0.5)),
        use_final_snap=bool(optional_bool(cfg, 'use_final_snap', default=True)),
        final_snap_mode=str(final_snap_mode),
        final_snap_ltcor=int(optional_int(cfg, 'final_snap_ltcor', 3)),
    )


def _load_keep_reject_cfg(cfg: dict[str, Any]) -> PhysicsKeepRejectCfg:
    return PhysicsKeepRejectCfg(
        drop_low_frac=float(optional_float(cfg, 'drop_low_frac', 0.05)),
    )


def _load_robust_center_cfg(cfg: dict[str, Any]) -> PhysicsRobustCenterCfg:
    return PhysicsRobustCenterCfg(
        half_win=int(optional_int(cfg, 'half_win', 128)),
        local_global_diff_th_samples=int(
            optional_int(cfg, 'local_global_diff_th_samples', 128)
        ),
        local_discard_radius_traces=int(
            optional_int(cfg, 'local_discard_radius_traces', 32)
        ),
        local_inv_drop_th_samples=float(
            optional_float(cfg, 'local_inv_drop_th_samples', 10.0)
        ),
        local_inv_min_consec_steps=int(
            optional_int(cfg, 'local_inv_min_consec_steps', 2)
        ),
        global_vmin_m_s=float(optional_float(cfg, 'global_vmin_m_s', 300.0)),
        global_vmax_m_s=float(optional_float(cfg, 'global_vmax_m_s', 6000.0)),
        global_side_min_pts=int(optional_int(cfg, 'global_side_min_pts', 16)),
    )


def _validate_physics_lite_config(cfg: PhysicsLiteConfig) -> PhysicsLiteConfig:
    feasible = cfg.feasible_band
    _validate_positive_float('feasible_band.vmin_mask', feasible.vmin_mask)
    _validate_positive_float('feasible_band.vmax_mask', feasible.vmax_mask)
    if float(feasible.vmax_mask) < float(feasible.vmin_mask):
        msg = 'feasible_band.vmax_mask must be >= feasible_band.vmin_mask'
        raise ValueError(msg)
    _validate_nonnegative_float('feasible_band.taper_ms', feasible.taper_ms)

    trend = cfg.trend
    _validate_positive_int('trend.trend_local_section_len', trend.trend_local_section_len)
    _validate_positive_int('trend.trend_local_stride', trend.trend_local_stride)
    _validate_positive_float('trend.trend_local_huber_c', trend.trend_local_huber_c)
    _validate_positive_int('trend.trend_local_iters', trend.trend_local_iters)
    _validate_positive_float('trend.trend_local_vmin_mps', trend.trend_local_vmin_mps)
    _validate_positive_float('trend.trend_local_vmax_mps', trend.trend_local_vmax_mps)
    if float(trend.trend_local_vmax_mps) < float(trend.trend_local_vmin_mps):
        msg = 'trend.trend_local_vmax_mps must be >= trend.trend_local_vmin_mps'
        raise ValueError(msg)
    _validate_positive_float('trend.trend_sigma_ms', trend.trend_sigma_ms)
    _validate_positive_int('trend.trend_min_pts', trend.trend_min_pts)
    _validate_nonnegative_int(
        'trend.trend_var_half_win_traces',
        trend.trend_var_half_win_traces,
    )
    _validate_positive_float(
        'trend.trend_var_sigma_std_ms',
        trend.trend_var_sigma_std_ms,
    )
    _validate_positive_int('trend.trend_var_min_count', trend.trend_var_min_count)

    rs = cfg.residual_statics
    _validate_mode('residual_statics.rs_pre_snap_mode', rs.rs_pre_snap_mode)
    _validate_nonnegative_int('residual_statics.rs_pre_samples', rs.rs_pre_samples)
    _validate_nonnegative_int('residual_statics.rs_post_samples', rs.rs_post_samples)
    _validate_nonnegative_int('residual_statics.rs_max_lag', rs.rs_max_lag)
    _validate_positive_int('residual_statics.rs_k_neighbors', rs.rs_k_neighbors)
    _validate_nonnegative_int('residual_statics.rs_n_iter', rs.rs_n_iter)
    if not 0.0 <= float(rs.rs_c_th) <= 1.0:
        msg = 'residual_statics.rs_c_th must lie in [0, 1]'
        raise ValueError(msg)
    _validate_mode('residual_statics.final_snap_mode', rs.final_snap_mode)
    _validate_nonnegative_int('residual_statics.final_snap_ltcor', rs.final_snap_ltcor)

    keep = cfg.keep_reject
    if float(keep.drop_low_frac) < 0.0 or float(keep.drop_low_frac) >= 1.0:
        msg = 'keep_reject.drop_low_frac must lie in [0, 1)'
        raise ValueError(msg)

    robust = cfg.robust_center
    _validate_positive_int('robust_center.half_win', robust.half_win)
    _validate_nonnegative_int(
        'robust_center.local_global_diff_th_samples',
        robust.local_global_diff_th_samples,
    )
    _validate_nonnegative_int(
        'robust_center.local_discard_radius_traces',
        robust.local_discard_radius_traces,
    )
    _validate_nonnegative_float(
        'robust_center.local_inv_drop_th_samples',
        robust.local_inv_drop_th_samples,
    )
    _validate_positive_int(
        'robust_center.local_inv_min_consec_steps',
        robust.local_inv_min_consec_steps,
    )
    _validate_positive_float('robust_center.global_vmin_m_s', robust.global_vmin_m_s)
    _validate_positive_float('robust_center.global_vmax_m_s', robust.global_vmax_m_s)
    if float(robust.global_vmax_m_s) < float(robust.global_vmin_m_s):
        msg = 'robust_center.global_vmax_m_s must be >= robust_center.global_vmin_m_s'
        raise ValueError(msg)
    _validate_positive_int('robust_center.global_side_min_pts', robust.global_side_min_pts)
    return cfg


def load_physics_lite_config(cfg: dict[str, Any] | None) -> PhysicsLiteConfig:
    raw = {} if cfg is None else cfg
    if not isinstance(raw, dict):
        msg = 'physics config must be dict or null'
        raise TypeError(msg)
    typed = PhysicsLiteConfig(
        feasible_band=_load_feasible_band_cfg(
            _require_dict(raw.get('feasible_band'), key='feasible_band')
        ),
        trend=_load_trend_cfg(_require_dict(raw.get('trend'), key='trend')),
        residual_statics=_load_residual_statics_cfg(
            _require_dict(raw.get('residual_statics'), key='residual_statics')
        ),
        keep_reject=_load_keep_reject_cfg(
            _require_dict(raw.get('keep_reject'), key='keep_reject')
        ),
        robust_center=_load_robust_center_cfg(
            _require_dict(raw.get('robust_center'), key='robust_center')
        ),
    )
    return _validate_physics_lite_config(typed)


def physics_lite_config_to_dict(cfg: PhysicsLiteConfig) -> dict[str, Any]:
    return asdict(cfg)
