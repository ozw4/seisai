from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Stage2Cfg:
    in_segy_root: Path = Path('/home/dcuser/data/ActiveSeisField/jogsarar')
    in_infer_root: Path = Path('/home/dcuser/data/ActiveSeisField/jogsarar_out')
    out_segy_root: Path = Path('/home/dcuser/data/ActiveSeisField/jogsarar_psn512')
    segy_exts: tuple[str, ...] = ('.sgy', '.segy')
    half_win: int = 128
    up_factor: int = 2
    drop_low_frac: float = 0.05
    score_keys_for_weight: tuple[str, ...] = (
        'conf_prob1',
        'conf_rs1',
    )
    score_keys_for_filter: tuple[str, ...] = (
        'conf_prob1',
        'conf_trend1',
        'conf_rs1',
    )
    pick_key: str = 'pick_final'
    thresh_mode: str = 'global'
    emit_training_artifacts: bool = True
    global_vmin_m_s: float = 300.0
    global_vmax_m_s: float = 6000.0
    global_slope_eps: float = 1e-6
    global_side_min_pts: int = 16
    use_stage1_local_trendline_baseline: bool = True
    local_global_diff_th_samples: int = 128
    local_discard_radius_traces: int = 32
    local_trend_t_sec_key: str = 'trend_t_sec'
    local_trend_covered_key: str = 'trend_covered'
    local_trend_dt_sec_key: str = 'dt_sec'
    local_inv_drop_th_samples: float = 10.0
    local_inv_min_consec_steps: int = 2
    conf_trend_sigma_ms: float = 6.0
    conf_trend_var_half_win_traces: int = 8
    conf_trend_var_sigma_std_ms: float = 6.0
    conf_trend_var_min_count: int = 3

    @property
    def out_ns(self) -> int:
        return int(2 * int(self.half_win) * int(self.up_factor))


DEFAULT_STAGE2_CFG = Stage2Cfg()


def _validate_stage2_threshold_cfg(*, cfg: Stage2Cfg = DEFAULT_STAGE2_CFG) -> None:
    if cfg.thresh_mode not in ('global', 'per_segy'):
        msg = f"thresh_mode must be 'global' or 'per_segy', got {cfg.thresh_mode!r}"
        raise ValueError(msg)
    if (not bool(cfg.emit_training_artifacts)) and cfg.thresh_mode == 'global':
        msg = (
            'emit_training_artifacts=False does not support thresh_mode=global. '
            "Set thresh_mode='per_segy'."
        )
        raise ValueError(msg)


def validate_stage2_cfg(*, cfg: Stage2Cfg = DEFAULT_STAGE2_CFG) -> None:
    _validate_stage2_threshold_cfg(cfg=cfg)


__all__ = [
    'DEFAULT_STAGE2_CFG',
    'Stage2Cfg',
    '_validate_stage2_threshold_cfg',
    'validate_stage2_cfg',
]
