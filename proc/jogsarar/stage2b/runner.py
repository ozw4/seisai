# %%
#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import segyio
from common.paths import stage4_pred_npz_path as _stage4_pred_npz_path
from common.segy_io import read_basic_segy_info
from jogsarar_shared import find_segy_files
from stage2.cfg import DEFAULT_STAGE2_CFG, Stage2Cfg, _validate_stage2_threshold_cfg
from stage2.runner import (
    _base_valid_mask,
    _build_phase_pick_csr_npz,
    _build_trend_result,
    _extract_256,
    _field_key_to_int,
    _percentile_threshold,
    _resolve_thresholds_arg_for_training,
    _upsample_256_to_512_linear,
    build_keep_mask,
    out_pick_csr_npz_path_for_out,
    out_segy_path_for_in,
    out_sidecar_npz_path_for_out,
)
from stage2b.io import load_stage4_seed_from_pred_npz
from stage2b.process_one import process_one_segy as _process_one_segy


def infer_npz_path_for_segy(
    segy_path: Path, *, cfg: Stage2Cfg = DEFAULT_STAGE2_CFG
) -> Path:
    return _stage4_pred_npz_path(
        segy_path,
        in_raw_root=cfg.in_segy_root,
        out_pred_root=cfg.in_infer_root,
    )


def _load_stage4_local_trend_disabled(
    *,
    z: np.lib.npyio.NpzFile,
    n_traces: int,
    dt_sec_in: float,
    cfg: Stage2Cfg = DEFAULT_STAGE2_CFG,
) -> tuple[np.ndarray, np.ndarray]:
    del z, dt_sec_in, cfg
    center_i = np.full(int(n_traces), np.nan, dtype=np.float32)
    ok = np.zeros(int(n_traces), dtype=bool)
    return center_i, ok


def _load_minimal_inputs_for_thresholds(
    segy_path: Path,
    *,
    cfg: Stage2Cfg = DEFAULT_STAGE2_CFG,
) -> tuple[
    int,
    int,
    float,
    np.ndarray,
    dict[str, np.ndarray],
    object,
]:
    infer_npz = infer_npz_path_for_segy(segy_path, cfg=cfg)
    if not infer_npz.exists():
        msg = f'infer npz not found for segy: {segy_path} expected={infer_npz}'
        raise FileNotFoundError(msg)

    with segyio.open(str(segy_path), 'r', ignore_geometry=True) as src:
        n_traces, ns_in, _dt_us_in, dt_sec_in = read_basic_segy_info(
            src,
            path=segy_path,
            name='',
        )

        seed = load_stage4_seed_from_pred_npz(
            pred_npz=infer_npz,
            n_traces=n_traces,
            dt_sec_in=dt_sec_in,
            cfg=cfg,
        )
        trend_res = _build_trend_result(
            src=src,
            n_traces=n_traces,
            n_samples_in=ns_in,
            dt_sec_in=dt_sec_in,
            pick_final_i=seed.pick_final,
            scores=seed.scores_weight,
            trend_center_i_local_in=seed.trend_center_i_local,
            local_trend_ok_in=seed.local_trend_ok,
            cfg=cfg,
        )
        scores_filter: dict[str, np.ndarray] = {
            'conf_prob1': seed.scores_weight['conf_prob1'],
            'conf_rs1': seed.scores_weight['conf_rs1'],
            'conf_trend1': trend_res.conf_trend1,
        }

        return (
            n_traces,
            ns_in,
            dt_sec_in,
            seed.pick_final,
            scores_filter,
            trend_res,
        )


def compute_global_thresholds(
    segys: list[Path], *, cfg: Stage2Cfg = DEFAULT_STAGE2_CFG
) -> dict[str, float]:
    vals: dict[str, list[np.ndarray]] = {k: [] for k in cfg.score_keys_for_filter}

    n_files_used = 0
    n_base_total = 0

    for p in segys:
        infer_npz = infer_npz_path_for_segy(p, cfg=cfg)
        if not infer_npz.exists():
            continue

        (
            _n_traces,
            ns_in,
            _dt_sec_in,
            pick_final,
            scores,
            trend_res,
        ) = _load_minimal_inputs_for_thresholds(p, cfg=cfg)

        base_valid, _reason = _base_valid_mask(
            pick_final_i=pick_final,
            trend_center_i=trend_res.trend_center_i_used,
            n_samples_in=ns_in,
            cfg=cfg,
        )

        n_b = int(np.count_nonzero(base_valid))
        if n_b <= 0:
            continue

        for k in cfg.score_keys_for_filter:
            s = np.asarray(scores[k], dtype=np.float32)
            vals[k].append(s[base_valid])

        n_files_used += 1
        n_base_total += n_b

    if n_files_used == 0:
        msg = 'no segy files with infer npz found for global thresholds'
        raise RuntimeError(msg)
    if n_base_total == 0:
        msg = 'no base_valid traces across all files (cannot compute global thresholds)'
        raise RuntimeError(msg)

    thresholds: dict[str, float] = {}
    for k in cfg.score_keys_for_filter:
        if not vals[k]:
            msg = f'no values accumulated for score={k}'
            raise RuntimeError(msg)
        v = np.concatenate(vals[k]).astype(np.float32, copy=False)
        th = _percentile_threshold(v, frac=cfg.drop_low_frac)
        if np.isnan(th):
            msg = f'global threshold became NaN for score={k}'
            raise RuntimeError(msg)
        thresholds[k] = th

    print(
        f'[GLOBAL_TH] files_used={n_files_used} base_valid_total={n_base_total} '
        f'p10 prob={thresholds["conf_prob1"]:.6g} trend={thresholds["conf_trend1"]:.6g} rs={thresholds["conf_rs1"]:.6g}'
    )
    return thresholds


def process_one_segy(
    segy_path: Path,
    *,
    global_thresholds: dict[str, float] | None,
    cfg: Stage2Cfg = DEFAULT_STAGE2_CFG,
) -> None:
    _process_one_segy(
        segy_path,
        global_thresholds=global_thresholds,
        cfg=cfg,
        validate_stage2_threshold_cfg_fn=_validate_stage2_threshold_cfg,
        infer_npz_path_for_segy_fn=infer_npz_path_for_segy,
        out_segy_path_for_in_fn=out_segy_path_for_in,
        out_sidecar_npz_path_for_out_fn=out_sidecar_npz_path_for_out,
        out_pick_csr_npz_path_for_out_fn=out_pick_csr_npz_path_for_out,
        load_stage1_local_trend_center_i_fn=_load_stage4_local_trend_disabled,
        build_trend_result_fn=_build_trend_result,
        resolve_thresholds_arg_for_training_fn=_resolve_thresholds_arg_for_training,
        build_keep_mask_fn=build_keep_mask,
        field_key_to_int_fn=_field_key_to_int,
        extract_256_fn=_extract_256,
        upsample_256_to_512_linear_fn=_upsample_256_to_512_linear,
        build_phase_pick_csr_npz_fn=_build_phase_pick_csr_npz,
    )


def run_stage2(
    *,
    cfg: Stage2Cfg = DEFAULT_STAGE2_CFG,
    segy_paths: list[Path] | None = None,
) -> None:
    _validate_stage2_threshold_cfg(cfg=cfg)

    if segy_paths is None:
        segys = find_segy_files(cfg.in_segy_root, exts=cfg.segy_exts, recursive=True)
    else:
        segys = list(segy_paths)
    print(f'[RUN] found {len(segys)} segy files under {cfg.in_segy_root}')

    segys2: list[Path] = []
    for p in segys:
        infer_npz = infer_npz_path_for_segy(p, cfg=cfg)
        if not infer_npz.exists():
            print(f'[SKIP] infer npz missing: {p}  expected={infer_npz}')
            continue
        segys2.append(p)

    global_thresholds = None
    if bool(cfg.emit_training_artifacts) and cfg.thresh_mode == 'global':
        global_thresholds = compute_global_thresholds(segys2, cfg=cfg)

    for p in segys2:
        process_one_segy(p, global_thresholds=global_thresholds, cfg=cfg)


__all__ = [
    '_percentile_threshold',
    'build_keep_mask',
    'compute_global_thresholds',
    'infer_npz_path_for_segy',
    'out_pick_csr_npz_path_for_out',
    'out_segy_path_for_in',
    'out_sidecar_npz_path_for_out',
    'process_one_segy',
    'run_stage2',
]
