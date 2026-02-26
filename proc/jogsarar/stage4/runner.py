# %%
#!/usr/bin/env python3

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import segyio
import torch
from common.npz_io import npz_1d, npz_scalar_float, npz_scalar_int
from common.paths import (
    resolve_sidecar_path as _resolve_sidecar_path_common,
    stage4_pred_crd_path as _stage4_pred_crd_path,
    stage4_pred_npz_path as _stage4_pred_npz_path,
    stage4_pred_out_dir as _stage4_pred_out_dir,
    stem_without_win512 as _stem_without_win512_common,
    win512_lookup_key as _win512_lookup_key,
)
from common.segy_io import (
    load_traces_by_indices,
    read_trace_field,
    require_expected_samples,
    require_matching_tracecount,
)
from jogsarar_shared import (
    TilePerTraceStandardize,
    build_key_to_indices,
    compute_residual_statics_metrics,
    find_segy_files,
)
from jogsarar_viz.noop import save_stage4_gather_viz_noop
from stage4.cfg import DEFAULT_STAGE4_CFG, Stage4Cfg
from stage4.model import load_psn_model_and_eps
from stage4.process_one import process_one_pair as _process_one_pair
from seisai_engine.predict import infer_tiled_chw
from seisai_pick.pickio.io_grstat import numpy2fbcrd
from seisai_pick.snap_picks_to_phase import snap_picks_to_phase


def _stem_without_win512(stem: str) -> str:
    return _stem_without_win512_common(stem)


def _replace_edge_picks_if_far(
    pick512: np.ndarray,
    pmax: np.ndarray,
    pick_orig_f: np.ndarray,
    pick_orig_i: np.ndarray,
    *,
    max_gap_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p512 = np.asarray(pick512, dtype=np.int32)
    pm = np.asarray(pmax, dtype=np.float32)
    pf = np.asarray(pick_orig_f, dtype=np.float32)
    pi = np.asarray(pick_orig_i, dtype=np.int32)

    if p512.ndim != 1 or pm.ndim != 1 or pf.ndim != 1 or pi.ndim != 1:
        msg = (
            'all pick arrays must be 1D: '
            f'pick512={p512.shape} pmax={pm.shape} '
            f'pick_orig_f={pf.shape} pick_orig_i={pi.shape}'
        )
        raise ValueError(msg)
    if not (p512.shape == pm.shape == pf.shape == pi.shape):
        msg = (
            'pick array shapes must match: '
            f'pick512={p512.shape} pmax={pm.shape} '
            f'pick_orig_f={pf.shape} pick_orig_i={pi.shape}'
        )
        raise ValueError(msg)

    valid = pi > 0
    idx = np.flatnonzero(valid).astype(np.int64, copy=False)
    if int(idx.size) < 3:
        return p512, pm, pf, pi

    i0 = int(idx[0])
    i1 = int(idx[-1])
    n0 = int(idx[1])
    n1 = int(idx[-2])

    gap0 = abs(int(pi[i0]) - int(pi[n0]))
    gap1 = abs(int(pi[i1]) - int(pi[n1]))
    if gap0 < int(max_gap_samples) and gap1 < int(max_gap_samples):
        return p512, pm, pf, pi

    out512 = p512.copy()
    outpm = pm.copy()
    outf = pf.copy()
    outi = pi.copy()

    if gap0 >= int(max_gap_samples):
        out512[i0] = out512[n0]
        outpm[i0] = outpm[n0]
        outf[i0] = outf[n0]
        outi[i0] = outi[n0]
    if gap1 >= int(max_gap_samples):
        out512[i1] = out512[n1]
        outpm[i1] = outpm[n1]
        outf[i1] = outf[n1]
        outi[i1] = outi[n1]

    return out512, outpm, outf, outi


def _build_win512_lookup(
    win_root: Path, *, cfg: Stage4Cfg = DEFAULT_STAGE4_CFG
) -> dict[tuple[str, str], Path]:
    win_files = find_segy_files(win_root, exts=cfg.segy_exts, recursive=True)
    lookup: dict[tuple[str, str], Path] = {}

    for p in win_files:
        key = _win512_lookup_key(p, win_root=win_root)
        if key in lookup:
            msg = (
                f'duplicate win512 mapping key={key}: {lookup[key]} and {p} (ambiguous)'
            )
            raise ValueError(msg)
        lookup[key] = p
    return lookup


def _resolve_sidecar_path(win_path: Path) -> Path | None:
    return _resolve_sidecar_path_common(win_path)


def _load_sidecar_window_start(
    *,
    sidecar_path: Path,
    n_traces: int,
    n_samples_in: int,
    n_samples_out: int,
    dt_sec_in: float,
    dt_sec_out: float,
    cfg: Stage4Cfg = DEFAULT_STAGE4_CFG,
) -> np.ndarray:
    with np.load(sidecar_path, allow_pickle=False) as z:
        window_start_i = npz_1d(
            z,
            'window_start_i',
            context='sidecar',
            dtype=np.int64,
        )
        if window_start_i.shape != (n_traces,):
            msg = (
                f'window_start_i must be (n_traces,), got {window_start_i.shape}, '
                f'n_traces={n_traces}'
            )
            raise ValueError(msg)

        side_n_tr = (
            npz_scalar_int(z, 'n_traces', context='sidecar')
            if 'n_traces' in z.files
            else None
        )
        side_ns_in = npz_scalar_int(z, 'n_samples_in', context='sidecar')
        side_ns_out = npz_scalar_int(z, 'n_samples_out', context='sidecar')
        side_dt_in = npz_scalar_float(z, 'dt_sec_in', context='sidecar')
        side_dt_out = npz_scalar_float(z, 'dt_sec_out', context='sidecar')

    if side_n_tr is not None and side_n_tr != int(n_traces):
        msg = f'sidecar n_traces mismatch: side={side_n_tr}, segy={n_traces}'
        raise ValueError(msg)
    if side_ns_in != int(n_samples_in):
        msg = f'sidecar n_samples_in mismatch: side={side_ns_in}, raw={n_samples_in}'
        raise ValueError(msg)
    if side_ns_out != int(n_samples_out):
        msg = f'sidecar n_samples_out mismatch: side={side_ns_out}, win={n_samples_out}'
        raise ValueError(msg)
    if abs(float(side_dt_in) - float(dt_sec_in)) > float(cfg.dt_tol_sec):
        msg = f'sidecar dt_sec_in mismatch: side={side_dt_in}, raw={dt_sec_in}'
        raise ValueError(msg)
    if abs(float(side_dt_out) - float(dt_sec_out)) > float(cfg.dt_tol_sec):
        msg = f'sidecar dt_sec_out mismatch: side={side_dt_out}, win={dt_sec_out}'
        raise ValueError(msg)

    return window_start_i


@dataclass(frozen=True)
class _PostTroughTraceDebug:
    tr_in_gather: int
    p_in: int
    p_out: int
    peak_i: int
    trough_i: int
    a_th: float
    peak_amp: float
    trough_amp: float
    reason: str


def _post_trough_apply_mask_from_offsets(
    offsets_m: np.ndarray, *, cfg: Stage4Cfg = DEFAULT_STAGE4_CFG
) -> np.ndarray:
    off = np.asarray(offsets_m, dtype=np.float32)
    m = np.isfinite(off)
    if cfg.post_trough_offs_abs_min_m is not None:
        m &= np.abs(off) >= float(cfg.post_trough_offs_abs_min_m)
    if cfg.post_trough_offs_abs_max_m is not None:
        m &= np.abs(off) <= float(cfg.post_trough_offs_abs_max_m)
    return m


def _shift_pick_to_preceding_trough_1d(
    x: np.ndarray,
    p: int,
    *,
    max_shift: int,
    scan_ahead: int,
    smooth_win: int,
    a_th: float,
    peak_search: str,
) -> tuple[int, int, int, float, float, float, str]:
    # "Return (p_out, peak_i, trough_i, a_th, peak_amp, trough_amp, reason).
    n = int(x.size)
    if p <= 0 or p >= n - 2:
        return p, -1, -1, float(a_th), 0.0, 0.0, 'p_oob'

    scale = float(np.max(np.abs(x)))
    if (not np.isfinite(scale)) or scale <= 0.0:
        return p, -1, -1, float(a_th), 0.0, 0.0, 'scale0'

    sw = int(smooth_win)
    if sw <= 0 or (sw % 2) != 1:
        msg = f'smooth_win must be positive odd, got {smooth_win}'
        raise ValueError(msg)

    ath = float(a_th)
    if (not np.isfinite(ath)) or ath < 0.0:
        msg = f'a_th must be finite and >= 0, got {a_th}'
        raise ValueError(msg)

    p_i = int(p)
    scan_i = int(scan_ahead)
    max_shift_i = int(max_shift)

    if peak_search == 'after_pick':
        is_after_mode = True
    elif peak_search == 'before_pick':
        is_after_mode = False
    else:
        msg = (
            'peak_search must be either '
            f'"after_pick" or "before_pick", got {peak_search!r}'
        )
        raise ValueError(msg)

    if is_after_mode:
        peak_lo = p_i + 1
        peak_hi = min(n - 2, p_i + scan_i)
        needed_lo = p_i
        needed_hi = peak_hi + 1
        # Keep legacy behavior: require at least two samples in (p, p+scan_ahead].
        if peak_hi <= p_i + 1:
            return p, -1, -1, ath, 0.0, 0.0, 'scan_empty'
        trough_lo = p_i + 1
    else:
        peak_lo = max(1, p_i - scan_i)
        peak_hi = p_i - 1
        # trough accepted only when abs(trough_i - p) <= max_shift
        trough_lo = max(1, p_i - max_shift_i)
        if peak_hi < peak_lo:
            return p, -1, -1, ath, 0.0, 0.0, 'scan_empty'
        needed_lo = min(peak_lo - 1, trough_lo - 1)
        needed_hi = p_i

    seg0 = max(0, needed_lo - sw)
    seg1 = min(n, needed_hi + sw + 1)

    seg = (np.asarray(x[seg0:seg1], dtype=np.float32) / np.float32(scale)).astype(
        np.float32, copy=False
    )
    if seg.size < 3:
        return p, -1, -1, ath, 0.0, 0.0, 'seg_too_short'

    if sw == 1:
        xs = seg
    else:
        pad = sw // 2
        k = np.full((sw,), 1.0 / float(sw), dtype=np.float32)
        seg_pad = np.pad(seg, pad_width=pad, mode='edge')
        xs = np.convolve(seg_pad, k, mode='valid').astype(np.float32, copy=False)

    def v_at(i: int) -> float:
        return float(xs[int(i) - int(seg0)])

    def is_local_max(i: int) -> bool:
        return bool((v_at(i - 1) < v_at(i)) and (v_at(i) >= v_at(i + 1)))

    def is_local_min(i: int) -> bool:
        return bool((v_at(i - 1) > v_at(i)) and (v_at(i) <= v_at(i + 1)))

    peak_i = -1
    peak_amp = 0.0
    if is_after_mode:
        peak_iter = range(peak_lo, peak_hi + 1)
    else:
        peak_iter = range(peak_hi, peak_lo - 1, -1)

    for i in peak_iter:
        if is_local_max(i):
            vi = v_at(i)
            if float(vi) >= float(ath):
                peak_i = i
                peak_amp = float(vi)
                break

    if peak_i < 0:
        return p, -1, -1, ath, 0.0, 0.0, 'no_pos_peak'

    trough_i = -1
    trough_amp = 0.0
    if is_after_mode:
        trough_iter = range(peak_i - 1, p_i, -1)
    else:
        trough_iter = range(peak_i - 1, trough_lo - 1, -1)

    for i in trough_iter:
        if is_local_min(i):
            trough_i = i
            trough_amp = float(v_at(i))
            break

    if trough_i < 0:
        return p, int(peak_i), -1, ath, float(peak_amp), 0.0, 'no_trough'
    if is_after_mode:
        trough_too_far = (trough_i - p_i) > max_shift_i
    else:
        trough_too_far = (p_i - trough_i) > max_shift_i
    if trough_too_far:
        return (
            p,
            int(peak_i),
            int(trough_i),
            ath,
            float(peak_amp),
            float(trough_amp),
            'trough_too_far',
        )
    return (
        int(trough_i),
        int(peak_i),
        int(trough_i),
        ath,
        float(peak_amp),
        float(trough_amp),
        'shifted',
    )


def _post_trough_adjust_picks(
    picks_i: np.ndarray,
    raw_hw: np.ndarray,
    *,
    max_shift: int,
    scan_ahead: int,
    smooth_win: int,
    a_th: float,
    peak_search: str,
    apply_mask: np.ndarray | None,
    debug: bool,
    debug_label: str,
    debug_max_examples: int,
) -> np.ndarray:
    p = np.asarray(picks_i, dtype=np.int32)
    x = np.asarray(raw_hw, dtype=np.float32)
    if x.ndim != 2:
        msg = f'raw_hw must be 2D, got {x.shape}'
        raise ValueError(msg)
    if p.shape != (x.shape[0],):
        msg = f'picks_i shape mismatch: picks={p.shape}, raw_hw={x.shape}'
        raise ValueError(msg)

    if apply_mask is None:
        m = np.ones((x.shape[0],), dtype=bool)
    else:
        m = np.asarray(apply_mask, dtype=bool)
        if m.shape != (x.shape[0],):
            msg = f'apply_mask must be (H,), got {m.shape}, H={x.shape[0]}'
            raise ValueError(msg)

    if peak_search == 'after_pick':
        peak_search_mode = 'after_pick'
    elif peak_search == 'before_pick':
        peak_search_mode = 'before_pick'
    else:
        msg = (
            'peak_search must be either '
            f'"after_pick" or "before_pick", got {peak_search!r}'
        )
        raise ValueError(msg)

    out = p.copy()
    if not bool(debug):
        for j in range(int(out.size)):
            if not bool(m[j]):
                continue
            pj = int(out[j])
            if pj <= 0:
                continue
            out[j] = np.int32(
                _shift_pick_to_preceding_trough_1d(
                    x[j],
                    pj,
                    max_shift=int(max_shift),
                    scan_ahead=int(scan_ahead),
                    smooth_win=int(smooth_win),
                    a_th=float(a_th),
                    peak_search=peak_search_mode,
                )[0]
            )
        return out

    reason_counts: dict[str, int] = {}
    examples: list[_PostTroughTraceDebug] = []
    shifts: list[int] = []

    for j in range(int(out.size)):
        if not bool(m[j]):
            reason_counts['skip_mask'] = reason_counts.get('skip_mask', 0) + 1
            continue
        pj = int(out[j])
        if pj <= 0:
            reason_counts['p<=0'] = reason_counts.get('p<=0', 0) + 1
            continue

        p2, peak_i, trough_i, ath, peak_amp, trough_amp, reason = (
            _shift_pick_to_preceding_trough_1d(
                x[j],
                pj,
                max_shift=int(max_shift),
                scan_ahead=int(scan_ahead),
                smooth_win=int(smooth_win),
                a_th=float(a_th),
                peak_search=peak_search_mode,
            )
        )
        out[j] = np.int32(p2)
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if reason == 'shifted':
            shifts.append(int(p2) - int(pj))
        if len(examples) < int(debug_max_examples):
            examples.append(
                _PostTroughTraceDebug(
                    tr_in_gather=int(j),
                    p_in=int(pj),
                    p_out=int(p2),
                    peak_i=int(peak_i),
                    trough_i=int(trough_i),
                    a_th=float(ath),
                    peak_amp=float(peak_amp),
                    trough_amp=float(trough_amp),
                    reason=str(reason),
                )
            )

    n_in = int(np.count_nonzero(p > 0))
    n_changed = int(np.count_nonzero(out != p))
    mean_shift = float(np.mean(shifts)) if shifts else 0.0
    max_abs_shift = int(np.max(np.abs(shifts))) if shifts else 0
    reasons = ' '.join(f'{k}={v}' for k, v in sorted(reason_counts.items()))
    print(
        f'[POST_TROUGH] {debug_label} '
        f'H={int(out.size)} nonzero={n_in} changed={n_changed} '
        f'mean_shift={mean_shift:.3f} max_abs_shift={max_abs_shift} '
        f'a_th={float(a_th):.3g} peak_search={peak_search_mode} {reasons}'
    )
    for ex in examples:
        shift = int(ex.p_out) - int(ex.p_in)
        print(
            f'  ex tr={ex.tr_in_gather} p:{ex.p_in}->{ex.p_out} (d={shift}) '
            f'peak(max)={ex.peak_i} trough={ex.trough_i} '
            f'th={ex.a_th:.3g} peak_amp={ex.peak_amp:.3g} trough_amp={ex.trough_amp:.3g} '
            f'reason={ex.reason}'
        )

    return out


def _align_post_trough_shifts_to_neighbors(
    p_in: np.ndarray,
    p_post: np.ndarray,
    *,
    peak_search: str,
    radius: int,
    min_support: int,
    max_dev: int,
    max_shift: int,
    propagate_zero: bool,
    zero_pin_tol: int,
    apply_mask: np.ndarray | None,
    debug: bool,
    debug_label: str,
) -> np.ndarray:
    pin = np.asarray(p_in, dtype=np.int32)
    ppo = np.asarray(p_post, dtype=np.int32)
    if pin.shape != ppo.shape:
        msg = f'p_in/p_post shape mismatch: {pin.shape} vs {ppo.shape}'
        raise ValueError(msg)

    if apply_mask is None:
        mask_apply = np.ones(pin.shape, dtype=bool)
    else:
        mask_apply = np.asarray(apply_mask, dtype=bool)
        if mask_apply.shape != pin.shape:
            msg = (
                f'apply_mask must match picks shape: {mask_apply.shape} vs {pin.shape}'
            )
            raise ValueError(msg)

    d = (ppo - pin).astype(np.int32, copy=False)
    valid = (pin > 0) & (ppo > 0) & mask_apply
    d[~valid] = 0

    r = int(radius)
    if r < 0:
        msg = f'radius must be >=0, got {radius}'
        raise ValueError(msg)
    ms = int(min_support)
    if ms < 0:
        msg = f'min_support must be >=0, got {min_support}'
        raise ValueError(msg)
    md = int(max_dev)
    if md < 0:
        msg = f'max_dev must be >=0, got {max_dev}'
        raise ValueError(msg)
    mmax = int(max_shift)
    if peak_search == 'after_pick':
        shift_lo = 0
        shift_hi = mmax
    elif peak_search == 'before_pick':
        shift_lo = -mmax
        shift_hi = 0
    else:
        msg = (
            'peak_search must be either '
            f'"after_pick" or "before_pick", got {peak_search!r}'
        )
        raise ValueError(msg)
    ztol = int(zero_pin_tol)
    if ztol < 0:
        msg = f'zero_pin_tol must be >=0, got {zero_pin_tol}'
        raise ValueError(msg)
    out = ppo.copy()
    n_correct_dev = 0
    n_propagate = 0
    h = int(d.size)
    for i in range(h):
        if not bool(mask_apply[i]):
            continue
        di = int(d[i])
        if (not bool(propagate_zero)) and di == 0:
            continue
        j0 = max(0, i - r)
        j1 = min(h - 1, i + r)
        nb = d[j0 : j1 + 1]
        if nb.size <= 1:
            continue
        mask = nb != 0
        mask[i - j0] = False
        cand = nb[mask]
        if int(cand.size) < ms:
            continue
        med = float(np.median(cand.astype(np.float32)))
        m_round = int(np.rint(med))
        m_round = max(shift_lo, min(shift_hi, m_round))

        if di == 0:
            # propagate only if this trace was not an outlier before post-trough:
            # compare p_in[i] to median p_in of moved neighbors (same mask as cand).
            pin_nb = pin[j0 : j1 + 1]
            pin_cand = pin_nb[mask]
            if pin_cand.size == 0:
                continue
            med_pin = float(np.median(pin_cand.astype(np.float32)))
            if abs(float(pin[i]) - med_pin) > float(ztol):
                continue
            out[i] = np.int32(int(pin[i]) + int(m_round))
            n_propagate += 1
            continue

        if abs(float(di) - med) > float(md):
            out[i] = np.int32(int(pin[i]) + int(m_round))
            n_correct_dev += 1

    if bool(debug) and (n_correct_dev > 0 or n_propagate > 0):
        print(
            f'[POST_TROUGH_ALIGN] {debug_label} '
            f'corrected_dev={n_correct_dev} propagated={n_propagate} '
            f'r={r} min_support={ms} max_dev={md} max_shift={mmax} '
            f'peak_search={peak_search}'
        )
    return out


@torch.no_grad()
def infer_pick512_from_win(
    *,
    model: torch.nn.Module,
    wave_hw: np.ndarray,
    standardize_eps: float,
    cfg: Stage4Cfg = DEFAULT_STAGE4_CFG,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(wave_hw, dtype=np.float32)
    if x.ndim != 2:
        msg = f'wave_hw must be (H,W), got {x.shape}'
        raise ValueError(msg)

    h, w = x.shape
    if w != int(cfg.tile_w):
        msg = f'win512 width mismatch: expected {cfg.tile_w}, got {w}'
        raise ValueError(msg)

    x_chw = x[None, :, :]
    tile_h = min(int(cfg.tile_h), int(h))
    overlap_h = min(int(cfg.overlap_h), max(tile_h - 1, 0))

    logits = infer_tiled_chw(
        model,
        x_chw,
        tile=(int(tile_h), int(cfg.tile_w)),
        overlap=(int(overlap_h), int(cfg.overlap_w)),
        amp=bool(cfg.use_amp),
        use_tqdm=bool(cfg.use_tqdm),
        tiles_per_batch=int(cfg.tiles_per_batch),
        tile_transform=TilePerTraceStandardize(eps_std=float(standardize_eps)),
        post_tile_transform=None,
    )

    if int(logits.ndim) != 3:
        msg = f'logits must be (C,H,W), got shape={tuple(logits.shape)}'
        raise ValueError(msg)
    if int(logits.shape[1]) != int(h) or int(logits.shape[2]) != int(cfg.tile_w):
        msg = (
            f'logits shape mismatch: got {tuple(logits.shape)}, '
            f'expected (*,{h},{cfg.tile_w})'
        )
        raise ValueError(msg)
    if int(logits.shape[0]) <= int(cfg.p_class_index):
        msg = (
            f'p_class_index={cfg.p_class_index} is out of range for '
            f'logits={tuple(logits.shape)}'
        )
        raise ValueError(msg)

    probs = torch.softmax(logits, dim=0)
    prob_p = probs[int(cfg.p_class_index)]  # (H, 512)
    pick512 = torch.argmax(prob_p, dim=1).to(dtype=torch.int32)
    pmax = torch.max(prob_p, dim=1).values.to(dtype=torch.float32)

    return (
        pick512.detach().cpu().numpy().astype(np.int32, copy=False),
        pmax.detach().cpu().numpy().astype(np.float32, copy=False),
    )


def process_one_pair(
    *,
    raw_path: Path,
    win_path: Path,
    sidecar_path: Path,
    model: torch.nn.Module,
    standardize_eps: float,
    cfg: Stage4Cfg = DEFAULT_STAGE4_CFG,
) -> None:
    save_gather_viz_fn: Callable[..., None] = save_stage4_gather_viz_noop
    if int(cfg.viz_every_n_shots) > 0:
        from jogsarar_viz.stage4_gather import save_stage4_gather_viz

        save_gather_viz_fn = save_stage4_gather_viz

    _process_one_pair(
        raw_path=raw_path,
        win_path=win_path,
        sidecar_path=sidecar_path,
        model=model,
        standardize_eps=standardize_eps,
        cfg=cfg,
        load_sidecar_window_start_fn=_load_sidecar_window_start,
        infer_pick512_from_win_fn=infer_pick512_from_win,
        replace_edge_picks_if_far_fn=_replace_edge_picks_if_far,
        post_trough_apply_mask_from_offsets_fn=_post_trough_apply_mask_from_offsets,
        post_trough_adjust_picks_fn=_post_trough_adjust_picks,
        align_post_trough_shifts_to_neighbors_fn=_align_post_trough_shifts_to_neighbors,
        save_gather_viz_fn=save_gather_viz_fn,
    )


def run_stage4(
    *,
    cfg: Stage4Cfg = DEFAULT_STAGE4_CFG,
    raw_paths: list[Path] | None = None,
) -> None:
    model, standardize_eps, ckpt_path = load_psn_model_and_eps(cfg=cfg)
    if raw_paths is None:
        raw_segys = find_segy_files(
            cfg.in_raw_segy_root, exts=cfg.segy_exts, recursive=True
        )
    else:
        raw_segys = list(raw_paths)
    win_lookup = _build_win512_lookup(cfg.in_win512_segy_root, cfg=cfg)

    print(
        f'[RUN] raw={len(raw_segys)} win_lookup={len(win_lookup)} '
        f'ckpt={ckpt_path} device={next(model.parameters()).device}'
    )

    n_ok = 0
    n_skip = 0

    for raw_path in raw_segys:
        rel = raw_path.relative_to(cfg.in_raw_segy_root)
        key = (rel.parent.as_posix(), rel.stem)
        win_path = win_lookup.get(key)
        if win_path is None:
            print(f'[SKIP] win512 segy missing: raw={raw_path} key={key}')
            n_skip += 1
            continue

        sidecar_path = _resolve_sidecar_path(win_path)
        if sidecar_path is None:
            print(f'[SKIP] sidecar missing for win512 segy: {win_path}')
            n_skip += 1
            continue

        process_one_pair(
            raw_path=raw_path,
            win_path=win_path,
            sidecar_path=sidecar_path,
            model=model,
            standardize_eps=standardize_eps,
            cfg=cfg,
        )
        n_ok += 1

    print(f'[DONE] processed={n_ok} skipped={n_skip} out_root={cfg.out_pred_root}')

__all__ = [
    '_align_post_trough_shifts_to_neighbors',
    '_build_win512_lookup',
    '_post_trough_adjust_picks',
    '_post_trough_apply_mask_from_offsets',
    '_replace_edge_picks_if_far',
    '_resolve_sidecar_path',
    '_shift_pick_to_preceding_trough_1d',
    '_stem_without_win512',
    'infer_pick512_from_win',
    'process_one_pair',
    'run_stage4',
]
