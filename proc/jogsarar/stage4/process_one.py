from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from common.segy_io import load_traces_by_indices
from jogsarar_shared import build_key_to_indices, compute_residual_statics_metrics
from stage4.io import open_and_load_stage4_inputs
from stage4.outputs import (
    resolve_stage4_out_paths,
    write_stage4_crd,
    write_stage4_pred_npz,
)
from seisai_pick.snap_picks_to_phase import snap_picks_to_phase


def process_one_pair(
    *,
    raw_path: Path,
    win_path: Path,
    sidecar_path: Path,
    model: torch.nn.Module,
    standardize_eps: float,
    cfg,
    load_sidecar_window_start_fn: Callable[..., np.ndarray],
    infer_pick512_from_win_fn: Callable[..., tuple[np.ndarray, np.ndarray]],
    replace_edge_picks_if_far_fn: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    post_trough_apply_mask_from_offsets_fn: Callable[..., np.ndarray],
    post_trough_adjust_picks_fn: Callable[..., np.ndarray],
    align_post_trough_shifts_to_neighbors_fn: Callable[..., np.ndarray],
    save_gather_viz_fn: Callable[..., None],
) -> None:
    out_dir, out_npz, out_crd = resolve_stage4_out_paths(raw_path=raw_path, cfg=cfg)

    with open_and_load_stage4_inputs(
        raw_path=raw_path,
        win_path=win_path,
        sidecar_path=sidecar_path,
        cfg=cfg,
        load_sidecar_window_start_fn=load_sidecar_window_start_fn,
    ) as (
        inputs,
        raw,
        win,
    ):
        n_traces = int(inputs.n_traces)
        n_samples_raw = int(inputs.n_samples_raw)
        dt_sec_raw = float(inputs.dt_sec_raw)
        ffid_values = inputs.ffid_values
        chno_values = inputs.chno_values
        offsets = inputs.offsets
        window_start_i = inputs.window_start_i

        pick_psn512 = np.zeros(n_traces, dtype=np.int32)
        pmax_psn = np.zeros(n_traces, dtype=np.float32)
        pick_psn_orig_f = np.zeros(n_traces, dtype=np.float32)
        pick_psn_orig_i = np.zeros(n_traces, dtype=np.int32)
        delta_pick_rs = np.zeros(n_traces, dtype=np.float32)
        cmax_rs = np.zeros(n_traces, dtype=np.float32)
        rs_valid_mask = np.zeros(n_traces, dtype=bool)
        pick_rs_i = np.zeros(n_traces, dtype=np.int32)
        pick_final = np.zeros(n_traces, dtype=np.int32)

        ffid_to_indices = build_key_to_indices(ffid_values)
        ffids_sorted = sorted(int(k) for k in ffid_to_indices)
        viz_ffids = (
            set(ffids_sorted[:: int(cfg.viz_every_n_shots)])
            if int(cfg.viz_every_n_shots) > 0
            else set()
        )
        viz_dir = out_dir / str(cfg.viz_dirname)

        max_chno = int(chno_values.max(initial=0))
        ffid_to_row = {ff: i for i, ff in enumerate(ffids_sorted)}
        fb_mat = np.zeros((len(ffids_sorted), max_chno), dtype=np.int32)

        for gather_i, ffid in enumerate(ffids_sorted):
            idx0 = ffid_to_indices[int(ffid)]
            ch = chno_values[idx0].astype(np.int64, copy=False)
            order = np.argsort(ch, kind='mergesort')
            idx = idx0[order]
            h_g = int(idx.size)
            if h_g < int(cfg.min_gather_h):
                # このgatherはPSN入力高さが小さすぎてConvNeXtのdownsampleで落ちるのでスキップ
                pick_psn512[idx] = 0
                pmax_psn[idx] = 0.0
                pick_psn_orig_f[idx] = 0.0
                pick_psn_orig_i[idx] = 0
                delta_pick_rs[idx] = 0.0
                cmax_rs[idx] = 0.0
                rs_valid_mask[idx] = False
                pick_rs_i[idx] = 0
                pick_final[idx] = 0

                print(
                    f'[SKIP_GATHER] {raw_path.name} ffid={ffid} '
                    f'H={h_g} < {cfg.min_gather_h} -> set picks=0'
                )
                continue
            chno_g = chno_values[idx].astype(np.int32, copy=False)
            offs_m = offsets[idx].astype(np.float32, copy=False)
            win_g = load_traces_by_indices(win, idx)  # (H, 512)
            raw_g = load_traces_by_indices(raw, idx)  # (H, ns_raw)
            wave_max_g = np.max(np.abs(raw_g), axis=1).astype(np.float32, copy=False)
            invalid_trace_g = (offs_m == 0.0) | (wave_max_g == 0.0)

            pick512_g, pmax_g = infer_pick512_from_win_fn(
                model=model,
                wave_hw=win_g,
                standardize_eps=standardize_eps,
                cfg=cfg,
            )

            win_start_g = window_start_i[idx].astype(np.float32, copy=False)
            pick_orig_f_g = win_start_g + pick512_g.astype(
                np.float32, copy=False
            ) / float(cfg.up_factor)
            pick_orig_i_g = np.rint(pick_orig_f_g).astype(np.int32, copy=False)
            valid_map = (pick_orig_f_g >= 0.0) & (pick_orig_f_g < float(n_samples_raw))

            # 範囲外は no-pick (0)
            if np.any(~valid_map):
                pick512_g = pick512_g.copy()
                pmax_g = pmax_g.copy()
                pick_orig_f_g = pick_orig_f_g.copy()
                pick_orig_i_g = pick_orig_i_g.copy()
                pick512_g[~valid_map] = 0
                pmax_g[~valid_map] = 0.0
                pick_orig_f_g[~valid_map] = 0.0
                pick_orig_i_g[~valid_map] = 0
            # Force invalid traces to no-pick (0):
            # - offset == 0
            # - all-samples amplitude == 0
            if np.any(invalid_trace_g):
                pick512_g = pick512_g.copy()
                pmax_g = pmax_g.copy()
                pick_orig_f_g = pick_orig_f_g.copy()
                pick_orig_i_g = pick_orig_i_g.copy()
                pick512_g[invalid_trace_g] = 0
                pmax_g[invalid_trace_g] = 0.0
                pick_orig_f_g[invalid_trace_g] = 0.0
                pick_orig_i_g[invalid_trace_g] = 0

            # Edge-fix before residual statics:
            # If the first/last valid pick differs from its neighbor by >= N samples,
            # replace it with the neighbor pick to avoid boundary outliers.
            pick512_g, pmax_g, pick_orig_f_g, pick_orig_i_g = (
                replace_edge_picks_if_far_fn(
                    pick512_g,
                    pmax_g,
                    pick_orig_f_g,
                    pick_orig_i_g,
                    max_gap_samples=int(cfg.edge_pick_max_gap_samples),
                )
            )

            rs_metrics = compute_residual_statics_metrics(
                wave_hw=raw_g,
                picks=pick_orig_i_g,
                pre=int(cfg.rs_pre),
                post=int(cfg.rs_post),
                fill=0.0,
                max_lag=int(cfg.rs_max_lag),
                k_neighbors=int(cfg.rs_k_neighbors),
                n_iter=int(cfg.rs_n_iter),
                mode=str(cfg.rs_mode),
                c_th=float(cfg.rs_c_th),
                smooth_method=str(cfg.rs_smooth_method),
                lam=float(cfg.rs_lam),
                subsample=bool(cfg.rs_subsample),
                propagate_low_corr=bool(cfg.rs_propagate_low_corr),
                taper=str(cfg.rs_taper),
                taper_power=float(cfg.rs_taper_power),
                lag_penalty=float(cfg.rs_lag_penalty),
                lag_penalty_power=float(cfg.rs_lag_penalty_power),
            )

            delta_g = rs_metrics.delta_pick
            cmax_g = rs_metrics.cmax
            valid_g = rs_metrics.valid_mask
            if delta_g.shape != (idx.shape[0],):
                msg = f'delta_pick shape mismatch for ffid={ffid}: {delta_g.shape}'
                raise ValueError(msg)
            if cmax_g.shape != (idx.shape[0],):
                msg = f'cmax shape mismatch for ffid={ffid}: {cmax_g.shape}'
                raise ValueError(msg)
            if valid_g.shape != (idx.shape[0],):
                msg = f'valid_mask shape mismatch for ffid={ffid}: {valid_g.shape}'
                raise ValueError(msg)

            if np.any(invalid_trace_g):
                delta_g = delta_g.copy()
                cmax_g = cmax_g.copy()
                valid_g = valid_g.copy()
                delta_g[invalid_trace_g] = 0.0
                cmax_g[invalid_trace_g] = 0.0
                valid_g[invalid_trace_g] = False

            pick_rs_f_g = pick_orig_i_g.astype(np.float32, copy=False) + delta_g
            pick_rs_i_g = np.rint(pick_rs_f_g).astype(np.int32, copy=False)
            np.clip(pick_rs_i_g, 0, int(n_samples_raw - 1), out=pick_rs_i_g)

            dbg = bool(cfg.post_trough_debug) and (
                int(cfg.post_trough_debug_every_n_gathers) > 0
                and (int(gather_i) % int(cfg.post_trough_debug_every_n_gathers) == 0)
            )
            dbg_label = f'{raw_path.name} ffid={ffid}'

            # apply post-trough refinement only within offset window
            pt_mask = post_trough_apply_mask_from_offsets_fn(offs_m, cfg=cfg)
            if np.any(invalid_trace_g):
                pt_mask = pt_mask & (~invalid_trace_g)

            pick_final_g = post_trough_adjust_picks_fn(
                pick_rs_i_g.copy(),
                raw_g,
                max_shift=int(cfg.post_trough_max_shift),
                scan_ahead=int(cfg.post_trough_scan_ahead),
                smooth_win=int(cfg.post_trough_smooth_win),
                a_th=float(cfg.post_trough_a_th),
                peak_search=cfg.post_trough_peak_search,
                apply_mask=pt_mask,
                debug=dbg,
                debug_label=dbg_label,
                debug_max_examples=int(cfg.post_trough_debug_max_examples),
            ).astype(np.int32, copy=False)

            pick_final_g = align_post_trough_shifts_to_neighbors_fn(
                pick_rs_i_g,
                pick_final_g,
                peak_search=cfg.post_trough_peak_search,
                radius=int(cfg.post_trough_outlier_radius),
                min_support=int(cfg.post_trough_outlier_min_support),
                max_dev=int(cfg.post_trough_outlier_max_dev),
                max_shift=int(cfg.post_trough_max_shift),
                propagate_zero=bool(cfg.post_trough_align_propagate_zero),
                zero_pin_tol=int(cfg.post_trough_align_zero_pin_tol),
                apply_mask=pt_mask,
                debug=dbg,
                debug_label=dbg_label,
            ).astype(np.int32, copy=False)

            # final cosmetic snap to phase
            np.clip(pick_final_g, 0, int(n_samples_raw - 1), out=pick_final_g)
            pick_final_g = snap_picks_to_phase(
                pick_final_g,
                raw_g,
                mode=str(cfg.snap_mode),
                ltcor=int(cfg.snap_ltcor),
            ).astype(np.int32, copy=False)
            # ensure out-of-range offsets are not modified by snap either
            if np.any(~pt_mask):
                pick_final_g = pick_final_g.copy()
                pick_final_g[~pt_mask] = pick_rs_i_g[~pt_mask]
            zero_in = pick_rs_i_g <= 0
            if np.any(zero_in):
                pick_final_g = pick_final_g.copy()
                pick_final_g[zero_in] = 0

            invalid_final = (
                (~np.isfinite(pick_rs_f_g))
                | (pick_final_g < 0)
                | (pick_final_g >= int(n_samples_raw))
            )
            if np.any(invalid_final):
                pick_final_g = pick_final_g.copy()
                pick_final_g[invalid_final] = 0

            if np.any(invalid_trace_g):
                pick_rs_i_g = pick_rs_i_g.copy()
                pick_final_g = pick_final_g.copy()
                pick_rs_i_g[invalid_trace_g] = 0
                pick_final_g[invalid_trace_g] = 0

            pick_psn512[idx] = pick512_g
            pmax_psn[idx] = pmax_g
            pick_psn_orig_f[idx] = pick_orig_f_g.astype(np.float32, copy=False)
            pick_psn_orig_i[idx] = pick_orig_i_g
            delta_pick_rs[idx] = delta_g
            cmax_rs[idx] = cmax_g
            rs_valid_mask[idx] = valid_g
            pick_rs_i[idx] = pick_rs_i_g
            pick_final[idx] = pick_final_g

            row = ffid_to_row[int(ffid)]
            for j in range(pick_final_g.shape[0]):
                cno = int(chno_g[j])
                if 1 <= cno <= max_chno:
                    fb_mat[row, cno - 1] = int(pick_final_g[j])

            if int(ffid) in viz_ffids:
                out_png = viz_dir / f'{raw_path.stem}.ffid{int(ffid)}.png'
                save_gather_viz_fn(
                    out_png=out_png,
                    raw_wave_hw=raw_g,
                    offsets_m=offs_m,
                    dt_sec=float(dt_sec_raw),
                    pick_psn_orig_i=pick_orig_i_g,
                    pick_rs_i=pick_rs_i_g,
                    pick_final_i=pick_final_g,
                    title=f'{raw_path.stem} ffid={int(ffid)}',
                    cfg=cfg,
                )

            if cfg.log_gather_rs:
                mean_cmax = float(np.mean(cmax_g)) if cmax_g.size > 0 else 0.0
                n_valid = int(np.count_nonzero(valid_g))
                n_forced0 = int(np.count_nonzero(invalid_trace_g))
                print(
                    f'[RS] {raw_path.name} ffid={ffid} '
                    f'n_valid={n_valid}/{valid_g.size} forced_zero={n_forced0} '
                    f'mean_cmax={mean_cmax:.3f}'
                )

    write_stage4_pred_npz(
        out_npz=out_npz,
        dt_sec_raw=dt_sec_raw,
        n_samples_raw=n_samples_raw,
        n_traces=n_traces,
        ffid_values=ffid_values,
        chno_values=chno_values,
        offsets=offsets,
        pick_psn512=pick_psn512,
        pmax_psn=pmax_psn,
        window_start_i=window_start_i,
        pick_psn_orig_f=pick_psn_orig_f,
        pick_psn_orig_i=pick_psn_orig_i,
        delta_pick_rs=delta_pick_rs,
        cmax_rs=cmax_rs,
        rs_valid_mask=rs_valid_mask,
        pick_rs_i=pick_rs_i,
        pick_final=pick_final,
    )

    write_stage4_crd(
        out_crd=out_crd,
        dt_ms=float(dt_sec_raw * 1000.0),
        fb_mat=fb_mat,
        ffids_sorted=ffids_sorted,
    )

    print(
        f'[OK] {raw_path.name} -> {out_npz.name}, {out_crd.name} '
        f'(traces={n_traces}, gathers={len(ffids_sorted)})'
    )


__all__ = ['process_one_pair']
