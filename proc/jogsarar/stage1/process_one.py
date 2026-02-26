from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import segyio
import torch
from jogsarar_shared import (
    compute_conf_rs_from_residual_statics,
    compute_conf_trend_gaussian_var,
    compute_residual_statics_metrics,
    valid_pick_mask,
)
from seisai_dataset.config import LoaderConfig
from seisai_dataset.file_info import build_file_info_dataclass
from seisai_dataset.trace_subset_preproc import TraceSubsetLoader
from seisai_pick.pickio.io_grstat import numpy2fbcrd
from seisai_pick.score.confidence_from_prob import (
    trace_confidence_from_prob_local_window,
)
from seisai_pick.snap_picks_to_phase import snap_picks_to_phase
from jogsarar_viz.noop import save_stage1_gather_viz_noop

BuildFileInfoFn = Callable[..., object]
SnapPicksFn = Callable[..., object]
Numpy2FbCrdFn = Callable[..., object]


def process_one_segy(
    *,
    segy_path: Path,
    out_dir: Path,
    model: torch.nn.Module,
    cfg,
    build_file_info_dataclass_fn: BuildFileInfoFn = build_file_info_dataclass,
    TraceSubsetLoaderCls: type[TraceSubsetLoader] = TraceSubsetLoader,
    LoaderConfigCls: type[LoaderConfig] = LoaderConfig,
    snap_picks_to_phase_fn: SnapPicksFn = snap_picks_to_phase,
    numpy2fbcrd_fn: Numpy2FbCrdFn = numpy2fbcrd,
    viz_every_n_shots: int = 0,
    viz_dirname: str = 'viz',
    infer_gather_prob_fn: Callable[..., tuple[np.ndarray, int]],
    fit_local_trend_split_sides_sec_fn: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray, int]],
    fit_local_trend_sec_fn: Callable[..., tuple[np.ndarray, np.ndarray]],
    save_conf_scatter_fn: Callable[..., None],
    scale01_by_percentile_fn: Callable[..., tuple[np.ndarray, tuple[float, float]]],
    save_gather_viz_fn: Callable[..., None] = save_stage1_gather_viz_noop,
) -> None:
    HEADER_CACHE_DIR = cfg.header_cache_dir
    WAVEFORM_MODE = str(cfg.waveform_mode)
    SEGY_ENDIAN = str(cfg.segy_endian)
    TILE_W = int(cfg.tile_w)
    USE_RESIDUAL_STATICS = bool(cfg.use_residual_statics)
    RS_BASE_PICK = str(cfg.rs_base_pick)
    RS_PRE_SNAP_MODE = str(cfg.rs_pre_snap_mode)
    RS_PRE_SNAP_LTCOR = int(cfg.rs_pre_snap_ltcor)
    RS_PRE_SAMPLES = int(cfg.rs_pre_samples)
    RS_POST_SAMPLES = int(cfg.rs_post_samples)
    RS_MAX_LAG = int(cfg.rs_max_lag)
    RS_K_NEIGHBORS = int(cfg.rs_k_neighbors)
    RS_N_ITER = int(cfg.rs_n_iter)
    RS_MODE = str(cfg.rs_mode)
    RS_C_TH = float(cfg.rs_c_th)
    RS_SMOOTH_METHOD = str(cfg.rs_smooth_method)
    RS_LAM = float(cfg.rs_lam)
    RS_SUBSAMPLE = bool(cfg.rs_subsample)
    RS_PROPAGATE_LOW_CORR = bool(cfg.rs_propagate_low_corr)
    RS_TAPER = str(cfg.rs_taper)
    RS_TAPER_POWER = float(cfg.rs_taper_power)
    RS_LAG_PENALTY = float(cfg.rs_lag_penalty)
    RS_LAG_PENALTY_POWER = float(cfg.rs_lag_penalty_power)
    USE_FINAL_SNAP = bool(cfg.use_final_snap)
    FINAL_SNAP_MODE = str(cfg.final_snap_mode)
    FINAL_SNAP_LTCOR = int(cfg.final_snap_ltcor)
    PMAX_TH = float(cfg.pmax_th)
    CONF_ENABLE = bool(cfg.conf_enable)
    TREND_LOCAL_ENABLE = bool(cfg.trend_local_enable)
    TREND_MIN_PTS = int(cfg.trend_min_pts)
    TREND_LOCAL_USE_ABS_OFFSET_HEADER = bool(cfg.trend_local_use_abs_offset_header)
    TREND_SIDE_SPLIT_ENABLE = bool(cfg.trend_side_split_enable)
    CONF_HALF_WIN = int(cfg.conf_half_win)
    TREND_SIGMA_MS = float(cfg.trend_sigma_ms)
    TREND_VAR_HALF_WIN_TRACES = int(cfg.trend_var_half_win_traces)
    TREND_VAR_SIGMA_STD_MS = float(cfg.trend_var_sigma_std_ms)
    TREND_VAR_MIN_COUNT = int(cfg.trend_var_min_count)
    RS_CMAX_TH = float(cfg.rs_cmax_th)
    RS_ABS_LAG_SOFT = float(cfg.rs_abs_lag_soft)
    CONF_VIZ_ENABLE = bool(cfg.conf_viz_enable)
    CONF_VIZ_FFID = int(cfg.conf_viz_ffid)
    VIZ_CONF_PROB_SCALE_ENABLE = bool(cfg.viz_conf_prob_scale_enable)
    VIZ_CONF_PROB_PCT_LO = float(cfg.viz_conf_prob_pct_lo)
    VIZ_CONF_PROB_PCT_HI = float(cfg.viz_conf_prob_pct_hi)
    VIZ_CONF_PROB_PCT_EPS = float(cfg.viz_conf_prob_pct_eps)
    SAVE_TREND_TO_NPZ = bool(cfg.save_trend_to_npz)
    LMO_VEL_MPS = float(cfg.lmo_vel_mps)
    LMO_BULK_SHIFT_SAMPLES = float(cfg.lmo_bulk_shift_samples)
    PLOT_START = int(cfg.plot_start)
    PLOT_END = int(cfg.plot_end)
    VIZ_SCORE_COMPONENTS = bool(cfg.viz_score_components)
    VIZ_TREND_LINE_ENABLE = bool(cfg.viz_trend_line_enable)
    VIZ_TREND_LINE_LW = float(cfg.viz_trend_line_lw)
    VIZ_TREND_LINE_ALPHA = float(cfg.viz_trend_line_alpha)
    VIZ_TREND_LINE_COLOR = str(cfg.viz_trend_line_color)
    VIZ_TREND_LINE_LABEL = str(cfg.viz_trend_line_label)
    VIZ_YMAX_CONF_PROB = cfg.viz_ymax_conf_prob
    VIZ_YMAX_CONF_TREND = cfg.viz_ymax_conf_trend
    VIZ_YMAX_CONF_RS = cfg.viz_ymax_conf_rs
    VIZ_SCORE_STYLE = str(cfg.viz_score_style)
    TREND_SOURCE_LABEL = str(cfg.trend_source_label)
    TREND_METHOD_LABEL = str(cfg.trend_method_label)
    TREND_LOCAL_SECTION_LEN = int(cfg.trend_local_section_len)
    TREND_LOCAL_STRIDE = int(cfg.trend_local_stride)
    TREND_LOCAL_HUBER_C = float(cfg.trend_local_huber_c)
    TREND_LOCAL_ITERS = int(cfg.trend_local_iters)
    TREND_LOCAL_VMIN_MPS = cfg.trend_local_vmin_mps
    TREND_LOCAL_VMAX_MPS = cfg.trend_local_vmax_mps
    TREND_LOCAL_SORT_OFFSETS = bool(cfg.trend_local_sort_offsets)

    info = build_file_info_dataclass_fn(
        str(segy_path),
        ffid_byte=segyio.TraceField.FieldRecord,
        chno_byte=segyio.TraceField.TraceNumber,
        cmp_byte=None,
        header_cache_dir=HEADER_CACHE_DIR,
        use_header_cache=True,
        include_centroids=False,
        waveform_mode=WAVEFORM_MODE,
        segy_endian=SEGY_ENDIAN,
    )

    try:
        n_traces = int(info.n_traces)
        n_samples = int(info.n_samples)
        dt_sec = float(info.dt_sec)

        ffid_values = np.asarray(info.ffid_values, dtype=np.int32)
        chno_values = np.asarray(info.chno_values, dtype=np.int32)
        offsets = np.asarray(info.offsets, dtype=np.float32)

        if (
            ffid_values.shape != (n_traces,)
            or chno_values.shape != (n_traces,)
            or offsets.shape != (n_traces,)
        ):
            msg = 'header arrays must be length n_traces'
            raise ValueError(msg)

        if info.ffid_key_to_indices is None:
            msg = 'ffid_key_to_indices is None (cannot group by ffid)'
            raise ValueError(msg)

        prob_all = np.zeros((n_traces, TILE_W), dtype=np.float16)
        trace_indices_all = np.arange(n_traces, dtype=np.int64)

        pick0_all = np.zeros(n_traces, dtype=np.int32)
        pick_pre_snap_all = np.zeros(n_traces, dtype=np.int32)
        delta_pick_all = np.zeros(n_traces, dtype=np.float32)
        pick_ref_all = np.zeros(n_traces, dtype=np.float32)
        pick_ref_i_all = np.zeros(n_traces, dtype=np.int32)
        pick_final_all = np.zeros(n_traces, dtype=np.int32)
        cmax_all = np.zeros(n_traces, dtype=np.float32)
        score_all = np.zeros(n_traces, dtype=np.float32)
        rs_valid_mask_all = np.zeros(n_traces, dtype=bool)

        conf_prob0_all = np.zeros(n_traces, dtype=np.float32)
        conf_prob1_all = np.zeros(n_traces, dtype=np.float32)
        conf_trend0_all = np.zeros(n_traces, dtype=np.float32)
        conf_trend1_all = np.zeros(n_traces, dtype=np.float32)
        conf_rs1_all = np.ones(n_traces, dtype=np.float32)

        # ---- trend 保存（npz） ----
        trend_t_sec_all = np.full(n_traces, np.nan, dtype=np.float32)
        trend_covered_all = np.zeros(n_traces, dtype=bool)
        trend_offset_signed_proxy_all = np.full(n_traces, np.nan, dtype=np.float32)
        trend_split_index_all = np.full(n_traces, -1, dtype=np.int32)

        max_chno = int(chno_values.max(initial=0))
        ffids_sorted = sorted(int(x) for x in info.ffid_key_to_indices)
        ffid_to_row = {ff: i for i, ff in enumerate(ffids_sorted)}
        fb_mat = np.zeros((len(ffids_sorted), max_chno), dtype=np.int32)

        viz_ffids = (
            set(ffids_sorted[::viz_every_n_shots]) if viz_every_n_shots > 0 else set()
        )

        loader = TraceSubsetLoaderCls(LoaderConfigCls(pad_traces_to=1))

        if USE_RESIDUAL_STATICS:
            print(
                '[CFG][RS] '
                f'base_pick={RS_BASE_PICK} pre_snap=({RS_PRE_SNAP_MODE},{RS_PRE_SNAP_LTCOR}) '
                f'window=({RS_PRE_SAMPLES},{RS_POST_SAMPLES}) '
                f'max_lag={RS_MAX_LAG} k_neighbors={RS_K_NEIGHBORS} n_iter={RS_N_ITER} '
                f'mode={RS_MODE} c_th={RS_C_TH:.2f} smooth={RS_SMOOTH_METHOD} '
                f'lam={RS_LAM:.2f} subsample={RS_SUBSAMPLE} '
                f'propagate_low_corr={RS_PROPAGATE_LOW_CORR} '
                f'taper={RS_TAPER} taper_power={RS_TAPER_POWER:.2f} '
                f'lag_penalty={RS_LAG_PENALTY:.3f} lag_penalty_power={RS_LAG_PENALTY_POWER:.2f} '
                f'final_snap={USE_FINAL_SNAP} final_mode={FINAL_SNAP_MODE} '
                f'final_ltcor={FINAL_SNAP_LTCOR}'
            )
        else:
            print('[CFG][RS] disabled (NO snap, output pick0)')

        for ffid in ffids_sorted:
            idx0 = np.asarray(info.ffid_key_to_indices[int(ffid)], dtype=np.int64)
            if idx0.size == 0:
                continue

            ch = chno_values[idx0].astype(np.int64, copy=False)
            order = np.argsort(ch, kind='mergesort')
            idx = idx0[order]

            wave_hw = loader.load_traces(info.mmap, idx)  # (H,W0)
            offs_m = offsets[idx].astype(np.float32, copy=False)
            chno_g = chno_values[idx].astype(np.int32, copy=False)

            wave_max = (
                np.max(np.abs(wave_hw), axis=1)
                if wave_hw.size
                else np.array([], dtype=np.float32)
            )
            invalid = (offs_m == 0.0) | (wave_max == 0.0)

            prob_hw, n_samples_orig = infer_gather_prob_fn(
                model=model,
                wave_hw=wave_hw,
                offsets_m=offs_m,
                dt_sec=dt_sec,
                cfg=cfg,
            )

            if invalid.any():
                prob_hw[invalid, :] = 0.0

            pick_argmax = np.argmax(prob_hw, axis=1).astype(np.int32, copy=False)
            pmax = np.max(prob_hw, axis=1).astype(np.float32, copy=False)

            nopick = (pmax < float(PMAX_TH)) | invalid
            pick0 = pick_argmax.copy()
            pick0[nopick] = 0

            wave_pad = np.zeros((wave_hw.shape[0], TILE_W), dtype=np.float32)
            wave_pad[:, :n_samples_orig] = wave_hw.astype(np.float32, copy=False)

            # ========
            # RS 無効時は、snapも一切しない
            # ========
            if not USE_RESIDUAL_STATICS:
                rs_label = 'pick0'
                pick_pre_snap = pick0.copy()
                delta = np.zeros(pick0.shape[0], dtype=np.float32)
                cmax_rs = np.zeros(pick0.shape[0], dtype=np.float32)
                score_rs = np.zeros(pick0.shape[0], dtype=np.float32)
                valid_rs = np.zeros(pick0.shape[0], dtype=bool)

                pick_ref = pick0.astype(np.float32, copy=False)
                pick_ref_i = pick0.astype(np.int32, copy=False)
                pick_out_i = pick0.astype(np.int32, copy=False)

            else:
                # RS 有効時のみ、必要なら pre-snap を作る
                if RS_BASE_PICK == 'snap':
                    pick_pre_snap = snap_picks_to_phase_fn(
                        pick0.copy(),
                        wave_pad,
                        mode=str(RS_PRE_SNAP_MODE),
                        ltcor=int(RS_PRE_SNAP_LTCOR),
                    ).astype(np.int32, copy=False)
                    pick_pre_snap[(pick0 == 0) | invalid] = 0
                    too_late_pre = (pick_pre_snap < 0) | (
                        pick_pre_snap >= int(n_samples_orig)
                    )
                    if np.any(too_late_pre):
                        pick_pre_snap = pick_pre_snap.copy()
                        pick_pre_snap[too_late_pre] = 0
                    pick_base = pick_pre_snap
                    base_label = 'snap'
                elif RS_BASE_PICK == 'pre':
                    pick_pre_snap = pick0.copy()
                    pick_base = pick0
                    base_label = 'pre'
                else:
                    msg = f"RS_BASE_PICK must be 'pre' or 'snap', got {RS_BASE_PICK!r}"
                    raise ValueError(msg)

                rs_metrics = compute_residual_statics_metrics(
                    wave_hw=wave_pad[:, :n_samples_orig],
                    picks=pick_base,
                    pre=int(RS_PRE_SAMPLES),
                    post=int(RS_POST_SAMPLES),
                    fill=0.0,
                    max_lag=int(RS_MAX_LAG),
                    k_neighbors=int(RS_K_NEIGHBORS),
                    n_iter=int(RS_N_ITER),
                    mode=str(RS_MODE),
                    c_th=float(RS_C_TH),
                    smooth_method=str(RS_SMOOTH_METHOD),
                    lam=float(RS_LAM),
                    subsample=bool(RS_SUBSAMPLE),
                    propagate_low_corr=bool(RS_PROPAGATE_LOW_CORR),
                    taper=RS_TAPER,
                    taper_power=float(RS_TAPER_POWER),
                    lag_penalty=float(RS_LAG_PENALTY),
                    lag_penalty_power=float(RS_LAG_PENALTY_POWER),
                )

                delta = rs_metrics.delta_pick
                cmax_rs = rs_metrics.cmax
                score_rs = rs_metrics.score
                valid_rs = rs_metrics.valid_mask

                pick_ref = pick_base.astype(np.float32, copy=False) + delta
                pick_ref[pick_base == 0] = 0.0
                pick_ref[invalid] = 0.0
                pick_ref = np.clip(pick_ref, 0.0, float(n_samples_orig - 1))

                pick_ref_i = np.rint(pick_ref).astype(np.int32, copy=False)
                pick_ref_i[pick_base == 0] = 0
                pick_ref_i[invalid] = 0

                if USE_FINAL_SNAP:
                    pick_final = snap_picks_to_phase_fn(
                        pick_ref_i.copy(),
                        wave_pad,
                        mode=str(FINAL_SNAP_MODE),
                        ltcor=int(FINAL_SNAP_LTCOR),
                    ).astype(np.int32, copy=False)
                    pick_final[pick_ref_i == 0] = 0
                    pick_final[invalid] = 0
                    too_late_final = (pick_final < 0) | (
                        pick_final >= int(n_samples_orig)
                    )
                    if np.any(too_late_final):
                        pick_final = pick_final.copy()
                        pick_final[too_late_final] = 0
                    pick_out_i = pick_final
                    rs_label = f'refine({base_label})+final_snap'
                else:
                    pick_out_i = pick_ref_i
                    rs_label = f'refine({base_label})'

                hist_last = rs_metrics.history[-1] if rs_metrics.history else {}
                mean_cmax = float(hist_last.get('mean_cmax', 0.0))
                mean_abs_update = float(hist_last.get('mean_abs_update', 0.0))
                n_valid_hist = int(hist_last.get('n_valid', 0))
                n_weighted_hist = int(hist_last.get('n_weighted', 0))
                print(
                    f'[RS] ffid={int(ffid)} '
                    f'n_valid={n_valid_hist} n_weighted={n_weighted_hist} '
                    f'mean_cmax={mean_cmax:.3f} mean_abs_update={mean_abs_update:.3f}'
                )

            # -----------------
            # Confidence + Trend
            # -----------------
            if CONF_ENABLE:
                dt_ms = float(dt_sec) * 1000.0

                # ---- trend: pick_out_i から作成（RS+snap後） ----
                trend_fit_mask = valid_pick_mask(
                    pick_out_i, n_samples=n_samples_orig
                ) & (~invalid)
                t_trend_sec: np.ndarray | None = None
                trend_covered = np.zeros(idx.shape[0], dtype=bool)
                trend_offset_signed_proxy = np.full(
                    idx.shape[0], np.nan, dtype=np.float32
                )
                trend_split_index = -1

                if TREND_LOCAL_ENABLE and int(np.count_nonzero(trend_fit_mask)) >= int(
                    TREND_MIN_PTS
                ):
                    t_pick_fit_sec = pick_out_i.astype(np.float32, copy=False) * float(
                        dt_sec
                    )

                    x_abs = (
                        np.abs(offs_m).astype(np.float32, copy=False)
                        if TREND_LOCAL_USE_ABS_OFFSET_HEADER
                        else offs_m.astype(np.float32, copy=False)
                    )

                    if TREND_SIDE_SPLIT_ENABLE:
                        t_trend_fit, cov_fit, off_proxy, split = (
                            fit_local_trend_split_sides_sec_fn(
                                offsets_abs_m=np.abs(x_abs).astype(
                                    np.float32, copy=False
                                ),
                                t_pick_sec=t_pick_fit_sec,
                                valid_mask=trend_fit_mask,
                                pmax=pmax,
                                invalid=invalid,
                                cfg=cfg,
                            )
                        )
                        t_trend_sec = t_trend_fit
                        trend_covered = cov_fit
                        trend_offset_signed_proxy = off_proxy
                        trend_split_index = int(split)
                    else:
                        t_trend_fit, cov_fit = fit_local_trend_sec_fn(
                            offsets_m=x_abs,
                            t_pick_sec=t_pick_fit_sec,
                            valid_mask=trend_fit_mask,
                            pmax=pmax,
                            cfg=cfg,
                        )
                        t_trend_sec = t_trend_fit
                        trend_covered = cov_fit
                        trend_offset_signed_proxy = x_abs.astype(np.float32, copy=False)
                        trend_split_index = -1

                # ---- conf_prob: pick近傍の局所形状（保存は生） ----
                conf_prob0 = trace_confidence_from_prob_local_window(
                    prob_hw,
                    pick_pre_snap,
                    half_win=int(CONF_HALF_WIN),
                )
                conf_prob1 = trace_confidence_from_prob_local_window(
                    prob_hw,
                    pick_out_i,
                    half_win=int(CONF_HALF_WIN),
                )

                # ---- conf_trend: 予測-トレンド（gauss）×（トレース方向局所分散） ----
                if t_trend_sec is None:
                    conf_trend0 = np.zeros(idx.shape[0], dtype=np.float32)
                    conf_trend1 = np.zeros(idx.shape[0], dtype=np.float32)
                else:
                    t0_sec = pick_pre_snap.astype(np.float32, copy=False) * float(
                        dt_sec
                    )
                    t1_sec = pick_out_i.astype(np.float32, copy=False) * float(dt_sec)

                    trend_ok = (~invalid) & np.isfinite(t_trend_sec)
                    valid0 = (
                        valid_pick_mask(pick_pre_snap, n_samples=n_samples_orig)
                        & trend_ok
                    )
                    valid1 = (
                        valid_pick_mask(pick_out_i, n_samples=n_samples_orig) & trend_ok
                    )

                    conf_trend0 = compute_conf_trend_gaussian_var(
                        t_pick_sec=t0_sec,
                        t_trend_sec=t_trend_sec,
                        valid_mask=valid0,
                        sigma_ms=float(TREND_SIGMA_MS),
                        half_win_traces=int(TREND_VAR_HALF_WIN_TRACES),
                        sigma_std_ms=float(TREND_VAR_SIGMA_STD_MS),
                        min_count=int(TREND_VAR_MIN_COUNT),
                        zero_invalid=False,
                    )
                    conf_trend1 = compute_conf_trend_gaussian_var(
                        t_pick_sec=t1_sec,
                        t_trend_sec=t_trend_sec,
                        valid_mask=valid1,
                        sigma_ms=float(TREND_SIGMA_MS),
                        half_win_traces=int(TREND_VAR_HALF_WIN_TRACES),
                        sigma_std_ms=float(TREND_VAR_SIGMA_STD_MS),
                        min_count=int(TREND_VAR_MIN_COUNT),
                        zero_invalid=False,
                    )

                # ---- conf_rs: residual statics の信頼度 ----
                if USE_RESIDUAL_STATICS:
                    conf_rs1 = compute_conf_rs_from_residual_statics(
                        delta_pick=delta,
                        cmax=cmax_rs,
                        valid_mask=valid_rs,
                        c_th=float(RS_CMAX_TH),
                        max_lag=float(RS_ABS_LAG_SOFT),
                    )
                else:
                    conf_rs1 = np.ones(idx.shape[0], dtype=np.float32)

                # ---- conf_viz scatter（conf_probのみ percentile scaling で 0..1） ----
                if CONF_VIZ_ENABLE and int(ffid) == int(CONF_VIZ_FFID):
                    conf_viz_dir = out_dir / 'conf_viz'
                    conf_viz_dir.mkdir(parents=True, exist_ok=True)

                    trend_hat_ms = None
                    if t_trend_sec is not None:
                        trend_hat_ms = (
                            np.asarray(t_trend_sec, dtype=np.float32) * 1000.0
                        ).astype(np.float32, copy=False)

                    conf_prob0_viz01 = np.asarray(conf_prob0, dtype=np.float32)
                    conf_prob1_viz01 = np.asarray(conf_prob1, dtype=np.float32)
                    if VIZ_CONF_PROB_SCALE_ENABLE:
                        conf_prob0_viz01, (plo0, phi0) = scale01_by_percentile_fn(
                            conf_prob0_viz01,
                            pct_lo=float(VIZ_CONF_PROB_PCT_LO),
                            pct_hi=float(VIZ_CONF_PROB_PCT_HI),
                            eps=float(VIZ_CONF_PROB_PCT_EPS),
                        )
                        conf_prob1_viz01, (plo1, phi1) = scale01_by_percentile_fn(
                            conf_prob1_viz01,
                            pct_lo=float(VIZ_CONF_PROB_PCT_LO),
                            pct_hi=float(VIZ_CONF_PROB_PCT_HI),
                            eps=float(VIZ_CONF_PROB_PCT_EPS),
                        )
                        title0 = (
                            f'{segy_path.stem} ffid={int(ffid)} p0 (pick_pre_snap) '
                            f'prob_pct[{VIZ_CONF_PROB_PCT_LO:.0f},{VIZ_CONF_PROB_PCT_HI:.0f}]={plo0:.3g},{phi0:.3g}'
                        )
                        title1 = (
                            f'{segy_path.stem} ffid={int(ffid)} p1 (pick_final) '
                            f'prob_pct[{VIZ_CONF_PROB_PCT_LO:.0f},{VIZ_CONF_PROB_PCT_HI:.0f}]={plo1:.3g},{phi1:.3g}'
                        )
                    else:
                        title0 = f'{segy_path.stem} ffid={int(ffid)} p0 (pick_pre_snap)'
                        title1 = f'{segy_path.stem} ffid={int(ffid)} p1 (pick_final)'

                    x_abs = np.abs(offs_m).astype(np.float32, copy=False)

                    p0_png = (
                        conf_viz_dir / f'{segy_path.stem}.ffid{int(ffid)}.p0.conf.png'
                    )
                    p1_png = (
                        conf_viz_dir / f'{segy_path.stem}.ffid{int(ffid)}.p1.conf.png'
                    )

                    save_conf_scatter_fn(
                        out_png=p0_png,
                        x_abs=x_abs,
                        picks_i=pick_pre_snap,
                        dt_ms=dt_ms,
                        conf_prob_viz01=conf_prob0_viz01,
                        conf_rs=np.ones(idx.shape[0], dtype=np.float32),
                        conf_trend=np.asarray(conf_trend0, dtype=np.float32),
                        title=title0,
                        trend_hat_ms=trend_hat_ms,
                    )
                    save_conf_scatter_fn(
                        out_png=p1_png,
                        x_abs=x_abs,
                        picks_i=pick_out_i,
                        dt_ms=dt_ms,
                        conf_prob_viz01=conf_prob1_viz01,
                        conf_rs=np.asarray(conf_rs1, dtype=np.float32),
                        conf_trend=np.asarray(conf_trend1, dtype=np.float32),
                        title=title1,
                        trend_hat_ms=trend_hat_ms,
                    )
                    print(f'[CONF_VIZ] saved {p0_png}')
                    print(f'[CONF_VIZ] saved {p1_png}')
            else:
                conf_prob0 = np.zeros(idx.shape[0], dtype=np.float32)
                conf_prob1 = np.zeros(idx.shape[0], dtype=np.float32)
                conf_trend0 = np.zeros(idx.shape[0], dtype=np.float32)
                conf_trend1 = np.zeros(idx.shape[0], dtype=np.float32)
                conf_rs1 = np.ones(idx.shape[0], dtype=np.float32)

                t_trend_sec = None
                trend_covered = np.zeros(idx.shape[0], dtype=bool)
                trend_offset_signed_proxy = np.full(
                    idx.shape[0], np.nan, dtype=np.float32
                )
                trend_split_index = -1

            # -----------------
            # store per-trace outputs
            # -----------------
            prob_all[idx, :] = prob_hw.astype(np.float16, copy=False)
            pick0_all[idx] = pick0.astype(np.int32, copy=False)
            pick_pre_snap_all[idx] = pick_pre_snap.astype(np.int32, copy=False)
            delta_pick_all[idx] = delta
            pick_ref_all[idx] = pick_ref.astype(np.float32, copy=False)
            pick_ref_i_all[idx] = pick_ref_i.astype(np.int32, copy=False)
            pick_final_all[idx] = pick_out_i.astype(np.int32, copy=False)
            cmax_all[idx] = cmax_rs
            score_all[idx] = score_rs
            rs_valid_mask_all[idx] = valid_rs

            conf_prob0_all[idx] = np.asarray(conf_prob0, dtype=np.float32)
            conf_prob1_all[idx] = np.asarray(conf_prob1, dtype=np.float32)
            conf_trend0_all[idx] = np.asarray(conf_trend0, dtype=np.float32)
            conf_trend1_all[idx] = np.asarray(conf_trend1, dtype=np.float32)
            conf_rs1_all[idx] = np.asarray(conf_rs1, dtype=np.float32)

            # ---- trend 保存（per-trace） ----
            if SAVE_TREND_TO_NPZ:
                if 't_trend_sec' in locals() and t_trend_sec is not None:
                    trend_t_sec_all[idx] = np.asarray(t_trend_sec, dtype=np.float32)
                else:
                    trend_t_sec_all[idx] = np.nan
                trend_covered_all[idx] = np.asarray(trend_covered, dtype=bool)
                trend_offset_signed_proxy_all[idx] = np.asarray(
                    trend_offset_signed_proxy, dtype=np.float32
                )
                trend_split_index_all[idx] = int(trend_split_index)

            # -----------------
            # fb_mat (output: pick_out_i)
            # -----------------
            row = ffid_to_row[int(ffid)]
            for j in range(pick_out_i.shape[0]):
                cno = int(chno_g[j])
                if 1 <= cno <= max_chno:
                    fb_mat[row, cno - 1] = int(pick_out_i[j])

            # -----------------
            # Visualization (LMO display only)
            # -----------------
            if viz_ffids and int(ffid) in viz_ffids:
                viz_dir = out_dir / viz_dirname
                png_path = viz_dir / f'{segy_path.stem}.ffid{int(ffid)}.png'
                save_gather_viz_fn(
                    out_png=png_path,
                    wave_pad=wave_pad,
                    n_samples_orig=int(n_samples_orig),
                    offsets_m=offs_m,
                    dt_sec=float(dt_sec),
                    pick_argmax=pick_argmax,
                    nopick_mask=nopick,
                    pick_out_i=pick_out_i,
                    invalid_trace_mask=invalid,
                    rs_label=rs_label,
                    plot_start=int(PLOT_START),
                    plot_end=int(PLOT_END),
                    lmo_vel_mps=float(LMO_VEL_MPS),
                    lmo_bulk_shift_samples=float(LMO_BULK_SHIFT_SAMPLES),
                    viz_score_components=bool(VIZ_SCORE_COMPONENTS),
                    conf_prob1=conf_prob1,
                    conf_trend1=conf_trend1,
                    conf_rs1=conf_rs1,
                    viz_conf_prob_scale_enable=bool(VIZ_CONF_PROB_SCALE_ENABLE),
                    viz_conf_prob_pct_lo=float(VIZ_CONF_PROB_PCT_LO),
                    viz_conf_prob_pct_hi=float(VIZ_CONF_PROB_PCT_HI),
                    viz_conf_prob_pct_eps=float(VIZ_CONF_PROB_PCT_EPS),
                    viz_score_style=str(VIZ_SCORE_STYLE),
                    viz_ymax_conf_prob=VIZ_YMAX_CONF_PROB,
                    viz_ymax_conf_trend=VIZ_YMAX_CONF_TREND,
                    viz_ymax_conf_rs=VIZ_YMAX_CONF_RS,
                    viz_trend_line_enable=bool(VIZ_TREND_LINE_ENABLE),
                    t_trend_sec=t_trend_sec,
                    viz_trend_line_lw=float(VIZ_TREND_LINE_LW),
                    viz_trend_line_alpha=float(VIZ_TREND_LINE_ALPHA),
                    viz_trend_line_color=str(VIZ_TREND_LINE_COLOR),
                    viz_trend_line_label=str(VIZ_TREND_LINE_LABEL),
                    title=f'{segy_path.stem} ffid={int(ffid)} (LMO v={LMO_VEL_MPS:.1f} m/s)',
                    scale01_by_percentile_fn=scale01_by_percentile_fn,
                )

        out_dir.mkdir(parents=True, exist_ok=True)
        stem = segy_path.stem

        npz_path = out_dir / f'{stem}.prob.npz'
        np.savez_compressed(
            npz_path,
            prob=prob_all,
            dt_sec=np.float32(dt_sec),
            n_samples_orig=np.int32(n_samples),
            ffid_values=ffid_values,
            chno_values=chno_values,
            offsets=offsets,
            trace_indices=trace_indices_all,
            pick0=pick0_all,
            pick_pre_snap=pick_pre_snap_all,
            delta_pick=delta_pick_all,
            pick_ref=pick_ref_all,
            pick_ref_i=pick_ref_i_all,
            pick_final=pick_final_all,
            cmax=cmax_all,
            score=score_all,
            rs_valid_mask=rs_valid_mask_all,
            # 保存は “生”
            conf_prob0=conf_prob0_all,
            conf_prob1=conf_prob1_all,
            conf_trend0=conf_trend0_all,
            conf_trend1=conf_trend1_all,
            conf_rs1=conf_rs1_all,
            # trend 保存
            trend_t_sec=trend_t_sec_all,
            trend_covered=trend_covered_all,
            trend_offset_signed_proxy=trend_offset_signed_proxy_all,
            trend_split_index=trend_split_index_all,
            trend_source=np.asarray(TREND_SOURCE_LABEL),
            trend_method=np.asarray(TREND_METHOD_LABEL),
            trend_cfg=np.asarray(
                f'section_len={TREND_LOCAL_SECTION_LEN},stride={TREND_LOCAL_STRIDE},'
                f'huber_c={TREND_LOCAL_HUBER_C},iters={TREND_LOCAL_ITERS},'
                f'vmin={TREND_LOCAL_VMIN_MPS},vmax={TREND_LOCAL_VMAX_MPS},'
                f'sort_offsets={TREND_LOCAL_SORT_OFFSETS},side_split={TREND_SIDE_SPLIT_ENABLE}'
            ),
        )

        crd_path = out_dir / f'{stem}.fb.crd'
        dt_ms = dt_sec * 1000.0
        numpy2fbcrd_fn(
            dt=float(dt_ms),
            fbnum=fb_mat,
            gather_range=ffids_sorted,
            output_name=str(crd_path),
            original=None,
            mode='gather',
            header_comment='machine learning fb pick',
        )

        print(f'[OK] {segy_path.name} -> {npz_path.name}, {crd_path.name}')

    finally:
        info.segy_obj.close()



__all__ = ['process_one_segy']
