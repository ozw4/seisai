from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from seisai_utils.config import (
    optional_bool,
    optional_float,
    optional_int,
    optional_str,
)

__all__ = [
    'DEFAULT_PHYSICS_LITE_CONFIG',
    'NeighborContextCfg',
    'PhysicalAdaptiveRefitCfg',
    'PhysicalAnchorReuseCfg',
    'PhysicalAnchorSelectionCfg',
    'PhysicalFitExecutorCfg',
    'PhysicalObservationSamplingCfg',
    'PhysicalPrefilterCfg',
    'PhysicalProjectionCfg',
    'PhysicalRuntimeCfg',
    'PhysicalRuntimeDiagnosticsCfg',
    'PhysicalT0ShiftCfg',
    'PhysicalTrendCfg',
    'PhysicsFeasibleBandCfg',
    'PhysicsKeepRejectCfg',
    'PhysicsLiteConfig',
    'PhysicsResidualStaticsCfg',
    'PhysicsRobustCenterCfg',
    'PhysicsTrendCfg',
    'TwoPieceRansacCfg',
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
class PhysicalTrendCfg:
    enabled: bool = False
    fit_kind: str = 'two_piece_ransac_autobreak'
    use_geometry_offset: bool = True
    min_offset_spread_m: float = 1.0
    coord_group_tol_m: float = 1.0
    segment_by_offset_sign: bool = True
    split_by_offset_gap: bool = True
    gap_ratio: float = 5.0
    min_gap_m: float | None = None


@dataclass(frozen=True)
class NeighborContextCfg:
    enabled: bool = True
    mode: str = 'nearest_source_xy'
    k_neighbors: int = 5
    max_source_distance_m: float | None = None
    include_self: bool = True


@dataclass(frozen=True)
class PhysicalPrefilterCfg:
    enabled: bool = True
    vmin_m_s: float = 300.0
    vmax_m_s: float = 6000.0
    t0_lo_ms: float = -20.0
    t0_hi_ms: float = 200.0
    pmax_min: float = 0.0
    use_existing_feasible_mask: bool = False


@dataclass(frozen=True)
class TwoPieceRansacCfg:
    n_iter: int = 200
    inlier_th_ms: float = 40.0
    min_pts: int = 8
    n_break_cand: int = 64
    q_lo: float = 0.15
    q_hi: float = 0.85
    seed: int = 0
    slope_eps: float = 1.0e-6
    sort_offsets: bool = True


@dataclass(frozen=True)
class PhysicalProjectionCfg:
    mode: str = 'model'


@dataclass(frozen=True)
class PhysicalAnchorSelectionCfg:
    enabled: bool = False
    mode: str = 'source_xy_stride'
    anchor_stride_source_groups: int = 5
    anchor_spacing_m: float | None = None
    include_first: bool = True
    include_last: bool = True


@dataclass(frozen=True)
class PhysicalAnchorReuseCfg:
    enabled: bool = True
    non_anchor_mode: str = 'nearest_anchor'
    max_anchor_distance_m: float | None = None
    reuse_segment_policy: str = 'same_side_and_gap'
    fallback_if_no_compatible_segment: str = 'full_fit'


@dataclass(frozen=True)
class PhysicalT0ShiftCfg:
    enabled: bool = True
    estimator: str = 'median'
    min_valid_for_t0_shift: int = 8
    t0_shift_clip_ms: float = 60.0
    use_physical_prefilter_mask: bool = True
    use_pmax_min: bool = True


@dataclass(frozen=True)
class PhysicalAdaptiveRefitCfg:
    enabled: bool = False
    resid_p90_ms_gt: float = 50.0
    median_abs_shift_ms_gt: float = 40.0
    min_valid_for_resid_check: int = 8
    fallback_if_refit_fails: str = 'nearest_anchor_plus_t0_shift'


@dataclass(frozen=True)
class PhysicalObservationSamplingCfg:
    enabled: bool = False
    method: str = 'offset_bin'
    max_obs_per_fit: int = 256
    n_offset_bins: int = 64
    bin_pick: str = 'pmax_max'
    min_obs_per_fit_after_sampling: int = 8
    preserve_edge_bins: bool = True


@dataclass(frozen=True)
class PhysicalFitExecutorCfg:
    enabled: bool = False
    backend: str = 'process'
    max_workers: int | None = None
    torch_num_threads_per_worker: int = 1
    chunksize: int = 1


@dataclass(frozen=True)
class PhysicalRuntimeDiagnosticsCfg:
    enabled: bool = True
    detailed_timing: bool = False
    save_json: bool = True
    save_npz_scalars: bool = True
    save_per_trace_context: bool = False


@dataclass(frozen=True)
class PhysicalRuntimeCfg:
    fit_policy: str = 'full'
    diagnostics_enabled: bool = True
    write_runtime_summary: bool = True
    diagnostics: PhysicalRuntimeDiagnosticsCfg = PhysicalRuntimeDiagnosticsCfg()
    anchor_selection: PhysicalAnchorSelectionCfg = PhysicalAnchorSelectionCfg()
    anchor_reuse: PhysicalAnchorReuseCfg = PhysicalAnchorReuseCfg()
    t0_shift: PhysicalT0ShiftCfg = PhysicalT0ShiftCfg()
    adaptive_refit: PhysicalAdaptiveRefitCfg = PhysicalAdaptiveRefitCfg()
    observation_sampling: PhysicalObservationSamplingCfg = (
        PhysicalObservationSamplingCfg()
    )
    fit_executor: PhysicalFitExecutorCfg = PhysicalFitExecutorCfg()


@dataclass(frozen=True)
class PhysicsLiteConfig:
    feasible_band: PhysicsFeasibleBandCfg = PhysicsFeasibleBandCfg()
    trend: PhysicsTrendCfg = PhysicsTrendCfg()
    residual_statics: PhysicsResidualStaticsCfg = PhysicsResidualStaticsCfg()
    keep_reject: PhysicsKeepRejectCfg = PhysicsKeepRejectCfg()
    robust_center: PhysicsRobustCenterCfg = PhysicsRobustCenterCfg()
    physical_trend: PhysicalTrendCfg = PhysicalTrendCfg()
    neighbor_context: NeighborContextCfg = NeighborContextCfg()
    physical_prefilter: PhysicalPrefilterCfg = PhysicalPrefilterCfg()
    two_piece_ransac: TwoPieceRansacCfg = TwoPieceRansacCfg()
    physical_projection: PhysicalProjectionCfg = PhysicalProjectionCfg()
    physical_runtime: PhysicalRuntimeCfg = PhysicalRuntimeCfg()


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
    if (not math.isfinite(out)) or out <= 0.0:
        msg = f'{name} must be finite and > 0'
        raise ValueError(msg)
    return out


def _validate_nonnegative_float(name: str, value: float) -> float:
    out = float(value)
    if (not math.isfinite(out)) or out < 0.0:
        msg = f'{name} must be finite and >= 0'
        raise ValueError(msg)
    return out


def _validate_finite_float(name: str, value: float) -> float:
    out = float(value)
    if not math.isfinite(out):
        msg = f'{name} must be finite'
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


def _optional_float_or_none(
    cfg: dict[str, Any],
    key: str,
    default: float | None,
) -> float | None:
    if key not in cfg:
        return default
    value = cfg[key]
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        msg = f'config.{key} must be float or null'
        raise TypeError(msg)
    return float(value)


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
        trend_var_half_win_traces=int(
            optional_int(cfg, 'trend_var_half_win_traces', 8)
        ),
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
        use_residual_statics=bool(
            optional_bool(cfg, 'use_residual_statics', default=True)
        ),
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


def _load_physical_trend_cfg(cfg: dict[str, Any]) -> PhysicalTrendCfg:
    fit_kind = optional_str(cfg, 'fit_kind', 'two_piece_ransac_autobreak')
    if fit_kind is None:
        msg = 'physical_trend.fit_kind must not be null'
        raise TypeError(msg)
    return PhysicalTrendCfg(
        enabled=bool(optional_bool(cfg, 'enabled', default=False)),
        fit_kind=str(fit_kind),
        use_geometry_offset=bool(
            optional_bool(cfg, 'use_geometry_offset', default=True)
        ),
        min_offset_spread_m=float(optional_float(cfg, 'min_offset_spread_m', 1.0)),
        coord_group_tol_m=float(optional_float(cfg, 'coord_group_tol_m', 1.0)),
        segment_by_offset_sign=bool(
            optional_bool(cfg, 'segment_by_offset_sign', default=True)
        ),
        split_by_offset_gap=bool(
            optional_bool(cfg, 'split_by_offset_gap', default=True)
        ),
        gap_ratio=float(optional_float(cfg, 'gap_ratio', 5.0)),
        min_gap_m=_optional_float_or_none(cfg, 'min_gap_m', None),
    )


def _load_neighbor_context_cfg(cfg: dict[str, Any]) -> NeighborContextCfg:
    mode = optional_str(cfg, 'mode', 'nearest_source_xy')
    if mode is None:
        msg = 'neighbor_context.mode must not be null'
        raise TypeError(msg)
    return NeighborContextCfg(
        enabled=bool(optional_bool(cfg, 'enabled', default=True)),
        mode=str(mode),
        k_neighbors=int(optional_int(cfg, 'k_neighbors', 5)),
        max_source_distance_m=_optional_float_or_none(
            cfg,
            'max_source_distance_m',
            None,
        ),
        include_self=bool(optional_bool(cfg, 'include_self', default=True)),
    )


def _load_physical_prefilter_cfg(cfg: dict[str, Any]) -> PhysicalPrefilterCfg:
    return PhysicalPrefilterCfg(
        enabled=bool(optional_bool(cfg, 'enabled', default=True)),
        vmin_m_s=float(optional_float(cfg, 'vmin_m_s', 300.0)),
        vmax_m_s=float(optional_float(cfg, 'vmax_m_s', 6000.0)),
        t0_lo_ms=float(optional_float(cfg, 't0_lo_ms', -20.0)),
        t0_hi_ms=float(optional_float(cfg, 't0_hi_ms', 200.0)),
        pmax_min=float(optional_float(cfg, 'pmax_min', 0.0)),
        use_existing_feasible_mask=bool(
            optional_bool(cfg, 'use_existing_feasible_mask', default=False)
        ),
    )


def _load_two_piece_ransac_cfg(cfg: dict[str, Any]) -> TwoPieceRansacCfg:
    return TwoPieceRansacCfg(
        n_iter=int(optional_int(cfg, 'n_iter', 200)),
        inlier_th_ms=float(optional_float(cfg, 'inlier_th_ms', 40.0)),
        min_pts=int(optional_int(cfg, 'min_pts', 8)),
        n_break_cand=int(optional_int(cfg, 'n_break_cand', 64)),
        q_lo=float(optional_float(cfg, 'q_lo', 0.15)),
        q_hi=float(optional_float(cfg, 'q_hi', 0.85)),
        seed=int(optional_int(cfg, 'seed', 0)),
        slope_eps=float(optional_float(cfg, 'slope_eps', 1.0e-6)),
        sort_offsets=bool(optional_bool(cfg, 'sort_offsets', default=True)),
    )


def _load_physical_projection_cfg(cfg: dict[str, Any]) -> PhysicalProjectionCfg:
    mode = optional_str(cfg, 'mode', 'model')
    if mode is None:
        msg = 'physical_projection.mode must not be null'
        raise TypeError(msg)
    return PhysicalProjectionCfg(mode=str(mode))


def _load_physical_anchor_selection_cfg(
    cfg: dict[str, Any],
) -> PhysicalAnchorSelectionCfg:
    return PhysicalAnchorSelectionCfg(
        enabled=bool(optional_bool(cfg, 'enabled', default=False)),
        mode=optional_str(cfg, 'mode', 'source_xy_stride'),
        anchor_stride_source_groups=int(
            optional_int(cfg, 'anchor_stride_source_groups', 5)
        ),
        anchor_spacing_m=_optional_float_or_none(cfg, 'anchor_spacing_m', None),
        include_first=bool(optional_bool(cfg, 'include_first', default=True)),
        include_last=bool(optional_bool(cfg, 'include_last', default=True)),
    )


def _load_physical_anchor_reuse_cfg(
    cfg: dict[str, Any],
) -> PhysicalAnchorReuseCfg:
    return PhysicalAnchorReuseCfg(
        enabled=bool(optional_bool(cfg, 'enabled', default=True)),
        non_anchor_mode=optional_str(cfg, 'non_anchor_mode', 'nearest_anchor'),
        max_anchor_distance_m=_optional_float_or_none(
            cfg,
            'max_anchor_distance_m',
            None,
        ),
        reuse_segment_policy=optional_str(
            cfg,
            'reuse_segment_policy',
            'same_side_and_gap',
        ),
        fallback_if_no_compatible_segment=optional_str(
            cfg,
            'fallback_if_no_compatible_segment',
            'full_fit',
        ),
    )


def _load_physical_t0_shift_cfg(cfg: dict[str, Any]) -> PhysicalT0ShiftCfg:
    return PhysicalT0ShiftCfg(
        enabled=bool(optional_bool(cfg, 'enabled', default=True)),
        estimator=optional_str(cfg, 'estimator', 'median'),
        min_valid_for_t0_shift=int(optional_int(cfg, 'min_valid_for_t0_shift', 8)),
        t0_shift_clip_ms=float(optional_float(cfg, 't0_shift_clip_ms', 60.0)),
        use_physical_prefilter_mask=bool(
            optional_bool(cfg, 'use_physical_prefilter_mask', default=True)
        ),
        use_pmax_min=bool(optional_bool(cfg, 'use_pmax_min', default=True)),
    )


def _load_physical_adaptive_refit_cfg(
    cfg: dict[str, Any],
) -> PhysicalAdaptiveRefitCfg:
    return PhysicalAdaptiveRefitCfg(
        enabled=bool(optional_bool(cfg, 'enabled', default=False)),
        resid_p90_ms_gt=float(optional_float(cfg, 'resid_p90_ms_gt', 50.0)),
        median_abs_shift_ms_gt=float(
            optional_float(cfg, 'median_abs_shift_ms_gt', 40.0)
        ),
        min_valid_for_resid_check=int(
            optional_int(cfg, 'min_valid_for_resid_check', 8)
        ),
        fallback_if_refit_fails=optional_str(
            cfg,
            'fallback_if_refit_fails',
            'nearest_anchor_plus_t0_shift',
        ),
    )


def _load_physical_observation_sampling_cfg(
    cfg: dict[str, Any],
) -> PhysicalObservationSamplingCfg:
    method = optional_str(cfg, 'method', 'offset_bin')
    bin_pick = optional_str(cfg, 'bin_pick', 'pmax_max')
    if method is None:
        msg = 'physical_runtime.observation_sampling.method must not be null'
        raise TypeError(msg)
    if bin_pick is None:
        msg = 'physical_runtime.observation_sampling.bin_pick must not be null'
        raise TypeError(msg)
    return PhysicalObservationSamplingCfg(
        enabled=bool(optional_bool(cfg, 'enabled', default=False)),
        method=str(method),
        max_obs_per_fit=int(optional_int(cfg, 'max_obs_per_fit', 256)),
        n_offset_bins=int(optional_int(cfg, 'n_offset_bins', 64)),
        bin_pick=str(bin_pick),
        min_obs_per_fit_after_sampling=int(
            optional_int(cfg, 'min_obs_per_fit_after_sampling', 8)
        ),
        preserve_edge_bins=bool(
            optional_bool(cfg, 'preserve_edge_bins', default=True)
        ),
    )


def _load_physical_fit_executor_cfg(cfg: dict[str, Any]) -> PhysicalFitExecutorCfg:
    backend = optional_str(cfg, 'backend', 'process')
    if backend is None:
        msg = 'physical_runtime.fit_executor.backend must not be null'
        raise TypeError(msg)
    max_workers = cfg.get('max_workers')
    if max_workers is not None:
        max_workers = int(optional_int(cfg, 'max_workers', 0))
    return PhysicalFitExecutorCfg(
        enabled=bool(optional_bool(cfg, 'enabled', default=False)),
        backend=str(backend),
        max_workers=max_workers,
        torch_num_threads_per_worker=int(
            optional_int(cfg, 'torch_num_threads_per_worker', 1)
        ),
        chunksize=int(optional_int(cfg, 'chunksize', 1)),
    )


def _load_physical_runtime_cfg(cfg: dict[str, Any]) -> PhysicalRuntimeCfg:
    diagnostics_raw = _require_dict(
        cfg.get('diagnostics'),
        key='physical_runtime.diagnostics',
    )
    legacy_enabled = bool(optional_bool(cfg, 'diagnostics_enabled', default=True))
    legacy_save_json = bool(optional_bool(cfg, 'write_runtime_summary', default=True))
    diagnostics = PhysicalRuntimeDiagnosticsCfg(
        enabled=bool(
            optional_bool(diagnostics_raw, 'enabled', default=legacy_enabled)
        ),
        detailed_timing=bool(
            optional_bool(diagnostics_raw, 'detailed_timing', default=False)
        ),
        save_json=bool(
            optional_bool(diagnostics_raw, 'save_json', default=legacy_save_json)
        ),
        save_npz_scalars=bool(
            optional_bool(diagnostics_raw, 'save_npz_scalars', default=True)
        ),
        save_per_trace_context=bool(
            optional_bool(
                diagnostics_raw,
                'save_per_trace_context',
                default=False,
            )
        ),
    )
    return PhysicalRuntimeCfg(
        fit_policy=optional_str(cfg, 'fit_policy', 'full'),
        diagnostics_enabled=bool(diagnostics.enabled),
        write_runtime_summary=bool(diagnostics.save_json),
        diagnostics=diagnostics,
        anchor_selection=_load_physical_anchor_selection_cfg(
            _require_dict(
                cfg.get('anchor_selection'),
                key='physical_runtime.anchor_selection',
            )
        ),
        anchor_reuse=_load_physical_anchor_reuse_cfg(
            _require_dict(
                cfg.get('anchor_reuse'),
                key='physical_runtime.anchor_reuse',
            )
        ),
        t0_shift=_load_physical_t0_shift_cfg(
            _require_dict(
                cfg.get('t0_shift'),
                key='physical_runtime.t0_shift',
            )
        ),
        adaptive_refit=_load_physical_adaptive_refit_cfg(
            _require_dict(
                cfg.get('adaptive_refit'),
                key='physical_runtime.adaptive_refit',
            )
        ),
        observation_sampling=_load_physical_observation_sampling_cfg(
            _require_dict(
                cfg.get('observation_sampling'),
                key='physical_runtime.observation_sampling',
            )
        ),
        fit_executor=_load_physical_fit_executor_cfg(
            _require_dict(
                cfg.get('fit_executor'),
                key='physical_runtime.fit_executor',
            )
        ),
    )


def _validate_physical_trend_cfg(cfg: PhysicalTrendCfg) -> None:
    if cfg.fit_kind != 'two_piece_ransac_autobreak':
        msg = (
            "physical_trend.fit_kind must be 'two_piece_ransac_autobreak', "
            f'got {cfg.fit_kind!r}'
        )
        raise ValueError(msg)
    _validate_positive_float(
        'physical_trend.coord_group_tol_m',
        cfg.coord_group_tol_m,
    )
    _validate_nonnegative_float(
        'physical_trend.min_offset_spread_m',
        cfg.min_offset_spread_m,
    )
    gap_ratio = float(cfg.gap_ratio)
    if (not math.isfinite(gap_ratio)) or gap_ratio <= 1.0:
        msg = 'physical_trend.gap_ratio must be finite and > 1.0'
        raise ValueError(msg)
    if cfg.min_gap_m is not None:
        _validate_positive_float('physical_trend.min_gap_m', cfg.min_gap_m)


def _validate_neighbor_context_cfg(cfg: NeighborContextCfg) -> None:
    if cfg.mode != 'nearest_source_xy':
        msg = f"neighbor_context.mode must be 'nearest_source_xy', got {cfg.mode!r}"
        raise ValueError(msg)
    _validate_positive_int('neighbor_context.k_neighbors', cfg.k_neighbors)
    if cfg.max_source_distance_m is not None:
        _validate_nonnegative_float(
            'neighbor_context.max_source_distance_m',
            cfg.max_source_distance_m,
        )


def _validate_physical_prefilter_cfg(cfg: PhysicalPrefilterCfg) -> None:
    _validate_positive_float('physical_prefilter.vmin_m_s', cfg.vmin_m_s)
    _validate_positive_float('physical_prefilter.vmax_m_s', cfg.vmax_m_s)
    if float(cfg.vmax_m_s) < float(cfg.vmin_m_s):
        msg = 'physical_prefilter.vmax_m_s must be >= physical_prefilter.vmin_m_s'
        raise ValueError(msg)
    t0_lo_ms = _validate_finite_float(
        'physical_prefilter.t0_lo_ms',
        cfg.t0_lo_ms,
    )
    t0_hi_ms = _validate_finite_float(
        'physical_prefilter.t0_hi_ms',
        cfg.t0_hi_ms,
    )
    if t0_lo_ms > t0_hi_ms:
        msg = 'physical_prefilter.t0_lo_ms must be <= physical_prefilter.t0_hi_ms'
        raise ValueError(msg)
    pmax_min = _validate_finite_float(
        'physical_prefilter.pmax_min',
        cfg.pmax_min,
    )
    if not 0.0 <= pmax_min <= 1.0:
        msg = 'physical_prefilter.pmax_min must lie in [0, 1]'
        raise ValueError(msg)


def _validate_two_piece_ransac_cfg(cfg: TwoPieceRansacCfg) -> None:
    _validate_positive_int('two_piece_ransac.n_iter', cfg.n_iter)
    _validate_positive_float('two_piece_ransac.inlier_th_ms', cfg.inlier_th_ms)
    if int(cfg.min_pts) < 2:
        msg = 'two_piece_ransac.min_pts must be >= 2'
        raise ValueError(msg)
    _validate_positive_int('two_piece_ransac.n_break_cand', cfg.n_break_cand)
    q_lo = _validate_finite_float('two_piece_ransac.q_lo', cfg.q_lo)
    q_hi = _validate_finite_float('two_piece_ransac.q_hi', cfg.q_hi)
    if not 0.0 <= q_lo < q_hi <= 1.0:
        msg = 'two_piece_ransac requires 0 <= q_lo < q_hi <= 1'
        raise ValueError(msg)
    _validate_nonnegative_float('two_piece_ransac.slope_eps', cfg.slope_eps)


def _validate_physical_projection_cfg(cfg: PhysicalProjectionCfg) -> None:
    if cfg.mode != 'model':
        msg = f"physical_projection.mode must be 'model', got {cfg.mode!r}"
        raise ValueError(msg)


def _validate_physical_runtime_cfg(cfg: PhysicalRuntimeCfg) -> None:
    if cfg.fit_policy not in {'full', 'anchor_source_xy'}:
        msg = (
            "physical_runtime.fit_policy must be 'full' or 'anchor_source_xy', "
            f'got {cfg.fit_policy!r}'
        )
        raise ValueError(msg)
    anchor = cfg.anchor_selection
    if anchor.mode != 'source_xy_stride':
        msg = (
            "physical_runtime.anchor_selection.mode must be 'source_xy_stride', "
            f'got {anchor.mode!r}'
        )
        raise ValueError(msg)
    _validate_positive_int(
        'physical_runtime.anchor_selection.anchor_stride_source_groups',
        anchor.anchor_stride_source_groups,
    )
    if anchor.anchor_spacing_m is not None:
        msg = 'physical_runtime.anchor_selection.anchor_spacing_m must be null'
        raise ValueError(msg)
    reuse = cfg.anchor_reuse
    if reuse.non_anchor_mode not in {
        'nearest_anchor',
        'nearest_anchor_plus_t0_shift',
    }:
        msg = (
            "physical_runtime.anchor_reuse.non_anchor_mode must be "
            "'nearest_anchor' or 'nearest_anchor_plus_t0_shift', "
            f'got {reuse.non_anchor_mode!r}'
        )
        raise ValueError(msg)
    if reuse.max_anchor_distance_m is not None:
        _validate_nonnegative_float(
            'physical_runtime.anchor_reuse.max_anchor_distance_m',
            reuse.max_anchor_distance_m,
        )
    if reuse.reuse_segment_policy != 'same_side_and_gap':
        msg = (
            "physical_runtime.anchor_reuse.reuse_segment_policy must be "
            f"'same_side_and_gap', got {reuse.reuse_segment_policy!r}"
        )
        raise ValueError(msg)
    t0_shift = cfg.t0_shift
    if t0_shift.estimator != 'median':
        msg = (
            "physical_runtime.t0_shift.estimator must be 'median', "
            f'got {t0_shift.estimator!r}'
        )
        raise ValueError(msg)
    _validate_positive_int(
        'physical_runtime.t0_shift.min_valid_for_t0_shift',
        t0_shift.min_valid_for_t0_shift,
    )
    _validate_nonnegative_float(
        'physical_runtime.t0_shift.t0_shift_clip_ms',
        t0_shift.t0_shift_clip_ms,
    )
    adaptive = cfg.adaptive_refit
    _validate_nonnegative_float(
        'physical_runtime.adaptive_refit.resid_p90_ms_gt',
        adaptive.resid_p90_ms_gt,
    )
    _validate_nonnegative_float(
        'physical_runtime.adaptive_refit.median_abs_shift_ms_gt',
        adaptive.median_abs_shift_ms_gt,
    )
    _validate_positive_int(
        'physical_runtime.adaptive_refit.min_valid_for_resid_check',
        adaptive.min_valid_for_resid_check,
    )
    if adaptive.fallback_if_refit_fails not in {
        'nearest_anchor_plus_t0_shift',
        'nearest_anchor',
        'existing_trend',
        'robust',
    }:
        msg = (
            'physical_runtime.adaptive_refit.fallback_if_refit_fails must be one '
            "of 'nearest_anchor_plus_t0_shift', 'nearest_anchor', "
            "'existing_trend', or 'robust', "
            f'got {adaptive.fallback_if_refit_fails!r}'
        )
        raise ValueError(msg)
    if reuse.fallback_if_no_compatible_segment not in {
        'full_fit',
        'existing_trend',
        'robust',
    }:
        msg = (
            'physical_runtime.anchor_reuse.fallback_if_no_compatible_segment '
            "must be one of 'full_fit', 'existing_trend', or 'robust', "
            f'got {reuse.fallback_if_no_compatible_segment!r}'
        )
        raise ValueError(msg)
    sampling = cfg.observation_sampling
    if sampling.method != 'offset_bin':
        msg = (
            "physical_runtime.observation_sampling.method must be 'offset_bin', "
            f'got {sampling.method!r}'
        )
        raise ValueError(msg)
    _validate_positive_int(
        'physical_runtime.observation_sampling.max_obs_per_fit',
        sampling.max_obs_per_fit,
    )
    _validate_positive_int(
        'physical_runtime.observation_sampling.n_offset_bins',
        sampling.n_offset_bins,
    )
    _validate_positive_int(
        'physical_runtime.observation_sampling.min_obs_per_fit_after_sampling',
        sampling.min_obs_per_fit_after_sampling,
    )
    if int(sampling.min_obs_per_fit_after_sampling) > int(sampling.max_obs_per_fit):
        msg = (
            'physical_runtime.observation_sampling.'
            'min_obs_per_fit_after_sampling must be <= max_obs_per_fit'
        )
        raise ValueError(msg)
    if sampling.bin_pick not in {'pmax_max', 'median_time', 'random'}:
        msg = (
            'physical_runtime.observation_sampling.bin_pick must be one of '
            "'pmax_max', 'median_time', or 'random', "
            f'got {sampling.bin_pick!r}'
        )
        raise ValueError(msg)
    executor = cfg.fit_executor
    if executor.backend not in {'process', 'thread'}:
        msg = (
            "physical_runtime.fit_executor.backend must be 'process' or 'thread', "
            f'got {executor.backend!r}'
        )
        raise ValueError(msg)
    if executor.max_workers is not None:
        _validate_positive_int(
            'physical_runtime.fit_executor.max_workers',
            executor.max_workers,
        )
    _validate_positive_int(
        'physical_runtime.fit_executor.torch_num_threads_per_worker',
        executor.torch_num_threads_per_worker,
    )
    _validate_positive_int(
        'physical_runtime.fit_executor.chunksize',
        executor.chunksize,
    )


def _validate_robust_center_cfg(cfg: PhysicsRobustCenterCfg) -> None:
    _validate_positive_int('robust_center.half_win', cfg.half_win)
    _validate_nonnegative_int(
        'robust_center.local_global_diff_th_samples',
        cfg.local_global_diff_th_samples,
    )
    _validate_nonnegative_int(
        'robust_center.local_discard_radius_traces',
        cfg.local_discard_radius_traces,
    )
    _validate_nonnegative_float(
        'robust_center.local_inv_drop_th_samples',
        cfg.local_inv_drop_th_samples,
    )
    _validate_positive_int(
        'robust_center.local_inv_min_consec_steps',
        cfg.local_inv_min_consec_steps,
    )
    _validate_positive_float('robust_center.global_vmin_m_s', cfg.global_vmin_m_s)
    _validate_positive_float('robust_center.global_vmax_m_s', cfg.global_vmax_m_s)
    if float(cfg.global_vmax_m_s) < float(cfg.global_vmin_m_s):
        msg = 'robust_center.global_vmax_m_s must be >= robust_center.global_vmin_m_s'
        raise ValueError(msg)
    _validate_positive_int('robust_center.global_side_min_pts', cfg.global_side_min_pts)


def _validate_physical_cfg(cfg: PhysicsLiteConfig) -> None:
    _validate_physical_trend_cfg(cfg.physical_trend)
    _validate_neighbor_context_cfg(cfg.neighbor_context)
    _validate_physical_prefilter_cfg(cfg.physical_prefilter)
    _validate_two_piece_ransac_cfg(cfg.two_piece_ransac)
    _validate_physical_projection_cfg(cfg.physical_projection)
    _validate_physical_runtime_cfg(cfg.physical_runtime)


def _validate_physics_lite_config(cfg: PhysicsLiteConfig) -> PhysicsLiteConfig:
    feasible = cfg.feasible_band
    _validate_positive_float('feasible_band.vmin_mask', feasible.vmin_mask)
    _validate_positive_float('feasible_band.vmax_mask', feasible.vmax_mask)
    if float(feasible.vmax_mask) < float(feasible.vmin_mask):
        msg = 'feasible_band.vmax_mask must be >= feasible_band.vmin_mask'
        raise ValueError(msg)
    _validate_finite_float('feasible_band.t0_lo_ms', feasible.t0_lo_ms)
    _validate_finite_float('feasible_band.t0_hi_ms', feasible.t0_hi_ms)
    _validate_nonnegative_float('feasible_band.taper_ms', feasible.taper_ms)

    trend = cfg.trend
    _validate_positive_int(
        'trend.trend_local_section_len',
        trend.trend_local_section_len,
    )
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
    rs_c_th = _validate_finite_float('residual_statics.rs_c_th', rs.rs_c_th)
    if not 0.0 <= rs_c_th <= 1.0:
        msg = 'residual_statics.rs_c_th must lie in [0, 1]'
        raise ValueError(msg)
    _validate_mode('residual_statics.final_snap_mode', rs.final_snap_mode)
    _validate_nonnegative_int('residual_statics.final_snap_ltcor', rs.final_snap_ltcor)

    keep = cfg.keep_reject
    drop_low_frac = _validate_finite_float(
        'keep_reject.drop_low_frac',
        keep.drop_low_frac,
    )
    if drop_low_frac < 0.0 or drop_low_frac >= 1.0:
        msg = 'keep_reject.drop_low_frac must lie in [0, 1)'
        raise ValueError(msg)

    _validate_robust_center_cfg(cfg.robust_center)
    _validate_physical_cfg(cfg)
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
        physical_trend=_load_physical_trend_cfg(
            _require_dict(raw.get('physical_trend'), key='physical_trend')
        ),
        neighbor_context=_load_neighbor_context_cfg(
            _require_dict(raw.get('neighbor_context'), key='neighbor_context')
        ),
        physical_prefilter=_load_physical_prefilter_cfg(
            _require_dict(raw.get('physical_prefilter'), key='physical_prefilter')
        ),
        two_piece_ransac=_load_two_piece_ransac_cfg(
            _require_dict(raw.get('two_piece_ransac'), key='two_piece_ransac')
        ),
        physical_projection=_load_physical_projection_cfg(
            _require_dict(raw.get('physical_projection'), key='physical_projection')
        ),
        physical_runtime=_load_physical_runtime_cfg(
            _require_dict(raw.get('physical_runtime'), key='physical_runtime')
        ),
    )
    return _validate_physics_lite_config(typed)


def physics_lite_config_to_dict(cfg: PhysicsLiteConfig) -> dict[str, Any]:
    return asdict(cfg)
