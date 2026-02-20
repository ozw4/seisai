# %%
#!/usr/bin/env python3
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import segyio
import torch
from jogsarar_shared import (
    TilePerTraceStandardize,
    build_key_to_indices,
    build_pick_aligned_window,
    find_segy_files,
    read_trace_field,
    require_npz_key,
)
from seisai_engine.pipelines.common.checkpoint_io import load_checkpoint
from seisai_engine.pipelines.psn.build_model import build_model as build_psn_model
from seisai_engine.pipelines.psn.config import load_psn_train_config
from seisai_engine.predict import infer_tiled_chw
from seisai_pick.pickio.io_grstat import numpy2fbcrd
from seisai_pick.residual_statics import refine_firstbreak_residual_statics
from seisai_pick.snap_picks_to_phase import snap_picks_to_phase
from seisai_utils import config_yaml
from seisai_utils.viz_wiggle import PickOverlay, WiggleConfig, plot_wiggle

# =========================
# CONFIG (fixed constants)
# =========================
IN_RAW_SEGY_ROOT = Path('/home/dcuser/data/ActiveSeisField/jogsarar')
IN_WIN512_SEGY_ROOT = Path('/home/dcuser/data/ActiveSeisField/jogsarar_psn512')
OUT_PRED_ROOT = Path('/home/dcuser/data/ActiveSeisField/jogsarar_psn512_pred')

CFG_YAML = Path('configs/config_convnext_prestage2_drop005.yaml')
# If None, resolved from CFG_YAML: paths.out_dir/ckpt/best.pt
CKPT_PATH: Path | None = None

SEGY_EXTS = ('.sgy', '.segy')
DEVICE = 'cuda'

# PSN class order from seisai_dataset.builder.builder.PhasePSNMap:
# [P, S, Noise] -> P index is fixed to 0.
P_CLASS_INDEX = 0

# win512 -> raw sample inverse mapping
UP_FACTOR = 2.0

# tiled inference
TILE_H = 128
TILE_W = 512
OVERLAP_H = 96
OVERLAP_W = 0
TILES_PER_BATCH = 8
USE_AMP = True
USE_TQDM = False

# lightweight residual statics + final snap
RS_PRE = 20
RS_POST = 20
RS_MAX_LAG = 4
RS_K_NEIGHBORS = 5
RS_N_ITER = 1
RS_MODE = 'diff'
RS_C_TH = 0.8
RS_SMOOTH_METHOD = 'wls'
RS_LAM = 5.0
RS_SUBSAMPLE = True
RS_PROPAGATE_LOW_CORR = False
RS_TAPER = 'hann'
RS_TAPER_POWER = 1.0
RS_LAG_PENALTY = 0.10
RS_LAG_PENALTY_POWER = 1.0

SNAP_MODE = 'trough'
SNAP_LTCOR = 3
LOG_GATHER_RS = True

# post-trough refinement: find first positive peak ahead of pick, then snap to its preceding trough
POST_TROUGH_MAX_SHIFT = 16
POST_TROUGH_SCAN_AHEAD = 32
POST_TROUGH_SMOOTH_WIN = 5

# post-trough refinement offset gating (ABS offset in meters).
# Set to None to disable each bound.
# Example:
#   POST_TROUGH_OFFS_ABS_MIN_M = 200.0
#   POST_TROUGH_OFFS_ABS_MAX_M = 2200.0
POST_TROUGH_OFFS_ABS_MIN_M: float | None = None
POST_TROUGH_OFFS_ABS_MAX_M: float | None = 1500

# Peak threshold on positive amplitude after per-trace normalization (max(|amp|)=1).
# Example: 0.05 means "accept the first local MAXIMUM with amp >= 0.05".
POST_TROUGH_A_TH = 0.03

# align shifts to neighborhood (robust) to keep per-trace moves consistent
POST_TROUGH_OUTLIER_RADIUS = 4
POST_TROUGH_OUTLIER_MIN_SUPPORT = 3
POST_TROUGH_OUTLIER_MAX_DEV = 2
POST_TROUGH_ALIGN_PROPAGATE_ZERO = False

# debug prints for post-trough
POST_TROUGH_DEBUG = False
POST_TROUGH_DEBUG_MAX_EXAMPLES = 5
POST_TROUGH_DEBUG_EVERY_N_GATHERS = 10

DT_TOL_SEC = 1e-9

# visualization (like run_fbp_infer.py)
VIZ_EVERY_N_SHOTS = 20
VIZ_DIRNAME = 'viz'
VIZ_PLOT_START = 0
VIZ_PLOT_END = 1000
VIZ_FIGSIZE = (12, 9)
VIZ_DPI = 200
VIZ_GAIN = 2.0

MIN_GATHER_H = 32


def _stem_without_win512(stem: str) -> str:
    tag = '.win512'
    if stem.endswith(tag):
        return stem[: -len(tag)]
    return stem


def _build_win512_lookup(win_root: Path) -> dict[tuple[str, str], Path]:
    win_files = find_segy_files(win_root, exts=SEGY_EXTS, recursive=True)
    lookup: dict[tuple[str, str], Path] = {}

    for p in win_files:
        rel = p.relative_to(win_root)
        key = (rel.parent.as_posix(), _stem_without_win512(rel.stem))
        if key in lookup:
            msg = (
                f'duplicate win512 mapping key={key}: {lookup[key]} and {p} (ambiguous)'
            )
            raise ValueError(msg)
        lookup[key] = p
    return lookup


def _resolve_sidecar_path(win_path: Path) -> Path | None:
    cands: list[Path] = [win_path.with_suffix('.sidecar.npz')]
    if not win_path.stem.endswith('.win512'):
        cands.append(win_path.with_suffix('.win512.sidecar.npz'))
    for p in cands:
        if p.is_file():
            return p
    return None


def _require_scalar_int(z: np.lib.npyio.NpzFile, key: str) -> int:
    return int(require_npz_key(z, key, context='sidecar').item())


def _require_scalar_float(z: np.lib.npyio.NpzFile, key: str) -> float:
    return float(require_npz_key(z, key, context='sidecar').item())


def _load_sidecar_window_start(
    *,
    sidecar_path: Path,
    n_traces: int,
    n_samples_in: int,
    n_samples_out: int,
    dt_sec_in: float,
    dt_sec_out: float,
) -> np.ndarray:
    with np.load(sidecar_path, allow_pickle=False) as z:
        window_start_i = require_npz_key(z, 'window_start_i', context='sidecar').astype(
            np.int64, copy=False
        )
        if window_start_i.shape != (n_traces,):
            msg = (
                f'window_start_i must be (n_traces,), got {window_start_i.shape}, '
                f'n_traces={n_traces}'
            )
            raise ValueError(msg)

        side_n_tr = (
            _require_scalar_int(z, 'n_traces') if 'n_traces' in z.files else None
        )
        side_ns_in = _require_scalar_int(z, 'n_samples_in')
        side_ns_out = _require_scalar_int(z, 'n_samples_out')
        side_dt_in = _require_scalar_float(z, 'dt_sec_in')
        side_dt_out = _require_scalar_float(z, 'dt_sec_out')

    if side_n_tr is not None and side_n_tr != int(n_traces):
        msg = f'sidecar n_traces mismatch: side={side_n_tr}, segy={n_traces}'
        raise ValueError(msg)
    if side_ns_in != int(n_samples_in):
        msg = f'sidecar n_samples_in mismatch: side={side_ns_in}, raw={n_samples_in}'
        raise ValueError(msg)
    if side_ns_out != int(n_samples_out):
        msg = f'sidecar n_samples_out mismatch: side={side_ns_out}, win={n_samples_out}'
        raise ValueError(msg)
    if abs(float(side_dt_in) - float(dt_sec_in)) > DT_TOL_SEC:
        msg = f'sidecar dt_sec_in mismatch: side={side_dt_in}, raw={dt_sec_in}'
        raise ValueError(msg)
    if abs(float(side_dt_out) - float(dt_sec_out)) > DT_TOL_SEC:
        msg = f'sidecar dt_sec_out mismatch: side={side_dt_out}, win={dt_sec_out}'
        raise ValueError(msg)

    return window_start_i


def _is_contiguous(idx: np.ndarray) -> bool:
    if idx.size <= 1:
        return True
    return bool(np.all(np.diff(idx) == 1))


def _load_traces_by_indices(segy_obj: segyio.SegyFile, idx: np.ndarray) -> np.ndarray:
    i = np.asarray(idx, dtype=np.int64)
    if i.ndim != 1:
        msg = f'idx must be 1D, got {i.shape}'
        raise ValueError(msg)
    if i.size == 0:
        msg = 'idx must be non-empty'
        raise ValueError(msg)

    if _is_contiguous(i):
        sl = slice(int(i[0]), int(i[-1]) + 1)
        data = np.asarray(segy_obj.trace.raw[sl], dtype=np.float32)
    else:
        rows = [np.asarray(segy_obj.trace.raw[int(j)], dtype=np.float32) for j in i]
        data = np.asarray(rows, dtype=np.float32)

    if data.ndim == 1:
        data = data[None, :]
    if data.ndim != 2:
        msg = f'loaded trace block must be 2D, got {data.shape}'
        raise ValueError(msg)
    return data.astype(np.float32, copy=False)


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


def _post_trough_apply_mask_from_offsets(offsets_m: np.ndarray) -> np.ndarray:
    off = np.asarray(offsets_m, dtype=np.float32)
    m = np.isfinite(off)
    if POST_TROUGH_OFFS_ABS_MIN_M is not None:
        m &= np.abs(off) >= float(POST_TROUGH_OFFS_ABS_MIN_M)
    if POST_TROUGH_OFFS_ABS_MAX_M is not None:
        m &= np.abs(off) <= float(POST_TROUGH_OFFS_ABS_MAX_M)
    return m


def _shift_pick_to_preceding_trough_1d(
    x: np.ndarray,
    p: int,
    *,
    max_shift: int,
    scan_ahead: int,
    smooth_win: int,
    a_th: float,
) -> tuple[int, int, int, float, float, float, str]:
    # "Return (p_out, peak_i, trough_i, a_th, peak_amp, trough_amp, reason).
    n = int(x.size)
    if p <= 0 or p >= n - 2:
        return p, -1, -1, float(a_th), 0.0, 0.0, 'p_oob'

    scale = float(np.max(np.abs(x)))
    if (not np.isfinite(scale)) or scale <= 0.0:
        return p, -1, -1, float(a_th), 0.0, 0.0, 'scale0'

    end = min(n - 2, p + int(scan_ahead))
    if end <= p + 1:
        return p, -1, -1, float(a_th), 0.0, 0.0, 'scan_empty'

    sw = int(smooth_win)
    if sw <= 0 or (sw % 2) != 1:
        msg = f'smooth_win must be positive odd, got {smooth_win}'
        raise ValueError(msg)

    ath = float(a_th)
    if (not np.isfinite(ath)) or ath < 0.0:
        msg = f'a_th must be finite and >= 0, got {a_th}'
        raise ValueError(msg)

    seg0 = max(0, int(p) - sw)
    seg1 = min(n, int(end) + sw + 2)
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
    for i in range(int(p) + 1, int(end) + 1):
        if not is_local_max(i):
            continue
        vi = v_at(i)
        if float(vi) >= float(ath):
            peak_i = i
            peak_amp = float(vi)
            break

    if peak_i < 0:
        return p, -1, -1, ath, 0.0, 0.0, 'no_pos_peak'

    trough_i = -1
    trough_amp = 0.0
    for i in range(int(peak_i) - 1, int(p), -1):
        if is_local_min(i):
            trough_i = i
            trough_amp = float(v_at(i))
            break

    if trough_i < 0:
        return p, int(peak_i), -1, ath, float(peak_amp), 0.0, 'no_trough'
    if int(trough_i) - int(p) > int(max_shift):
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
        f'a_th={float(a_th):.3g} {reasons}'
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
    radius: int,
    min_support: int,
    max_dev: int,
    max_shift: int,
    propagate_zero: bool,
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
        m_round = max(0, min(int(max_shift), m_round))

        if di == 0:
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
            f'r={r} min_support={ms} max_dev={md} max_shift={int(max_shift)}'
        )
    return out


def _pick_to_window_samples(
    picks: np.ndarray,
    *,
    start: int,
    end: int,
    zero_is_invalid: bool,
    add_samples: float = 0.0,
) -> np.ndarray:
    p = np.asarray(picks, dtype=np.float32)
    if p.ndim != 1:
        msg = f'picks must be 1D, got {p.shape}'
        raise ValueError(msg)
    if end <= start:
        msg = f'end must be > start, got start={start}, end={end}'
        raise ValueError(msg)

    out = p - float(start) + float(add_samples)
    win_len = float(end - start)
    valid = np.isfinite(p) & (out >= 0.0) & (out < win_len)
    if zero_is_invalid:
        valid &= p > 0.0
    out[~valid] = np.nan
    return out.astype(np.float32, copy=False)


def _save_gather_viz(
    *,
    out_png: Path,
    raw_wave_hw: np.ndarray,
    offsets_m: np.ndarray,
    dt_sec: float,
    pick_psn_orig_i: np.ndarray,
    pick_rs_i: np.ndarray,
    pick_final_i: np.ndarray,
    title: str,
) -> None:
    wave = np.asarray(raw_wave_hw, dtype=np.float32)
    if wave.ndim != 2:
        msg = f'raw_wave_hw must be 2D, got {wave.shape}'
        raise ValueError(msg)

    n_traces, n_samples = wave.shape
    if n_samples <= 0:
        msg = f'raw_wave_hw has invalid sample count: {n_samples}'
        raise ValueError(msg)
    if dt_sec <= 0.0:
        msg = f'dt_sec must be positive, got {dt_sec}'
        raise ValueError(msg)

    offs = np.asarray(offsets_m, dtype=np.float32)
    if offs.shape != (n_traces,):
        msg = f'offsets_m must be (H,), got {offs.shape}, H={n_traces}'
        raise ValueError(msg)

    # NOTE: visualization should NOT apply LMO (no wave LMO, no pick LMO)

    p_psn = np.asarray(pick_psn_orig_i, dtype=np.float32)
    p_rs = np.asarray(pick_rs_i, dtype=np.float32)
    p_final = np.asarray(pick_final_i, dtype=np.float32)
    valid_psn = np.isfinite(p_psn) & (p_psn >= 0.0) & (p_psn < float(n_samples))
    valid_rs = np.isfinite(p_rs) & (p_rs > 0.0) & (p_rs < float(n_samples))
    valid_final = np.isfinite(p_final) & (p_final > 0.0) & (p_final < float(n_samples))

    p_psn_plot = p_psn.copy()
    p_rs_plot = p_rs.copy()
    p_final_plot = p_final.copy()
    p_psn_plot[~valid_psn] = np.nan
    p_rs_plot[~valid_rs] = np.nan
    p_final_plot[~valid_final] = np.nan

    start = int(max(0, VIZ_PLOT_START))
    end_cfg = int(VIZ_PLOT_END)
    if end_cfg <= 0:
        end = n_samples
    else:
        end = min(n_samples, end_cfg)
    if end <= start:
        msg = (
            f'invalid viz sample range: start={start}, end={end}, n_samples={n_samples}'
        )
        raise ValueError(msg)

    wave_win = wave[:, start:end].astype(np.float32, copy=False)
    keep = np.max(np.abs(wave_win), axis=1) > 0.0
    if not np.any(keep):
        return

    x_keep = np.flatnonzero(keep).astype(np.float32, copy=False)
    wave_plot = wave_win[keep]

    pick_psn_win = _pick_to_window_samples(
        p_psn_plot,
        start=start,
        end=end,
        zero_is_invalid=False,
        add_samples=0.0,
    )[keep]
    pick_rs_win = _pick_to_window_samples(
        p_rs_plot,
        start=start,
        end=end,
        zero_is_invalid=False,
        add_samples=0.0,
    )[keep]
    pick_final_win = _pick_to_window_samples(
        p_final_plot,
        start=start,
        end=end,
        zero_is_invalid=False,
        add_samples=0.0,
    )[keep]

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=VIZ_FIGSIZE)
    plot_wiggle(
        wave_plot,
        ax=ax,
        cfg=WiggleConfig(
            dt=float(dt_sec),
            t0=float(start) * float(dt_sec),
            time_axis=1,
            x=x_keep,
            normalize='trace',
            gain=float(VIZ_GAIN),
            fill_positive=True,
            picks=(
                PickOverlay(
                    pick_psn_win,
                    unit='sample',
                    label='psn_orig',
                    marker='o',
                    size=12.0,
                    color='r',
                    alpha=0.8,
                ),
                PickOverlay(
                    pick_rs_win,
                    unit='sample',
                    label='psn+rs',
                    marker='x',
                    size=16.0,
                    color='b',
                    alpha=0.9,
                ),
                PickOverlay(
                    pick_final_win,
                    unit='sample',
                    label='final_snap',
                    marker='+',
                    size=20.0,
                    color='g',
                    alpha=0.9,
                ),
            ),
            show_legend=True,
        ),
    )
    ax.set_title(f'{title} (no LMO)')
    fig.tight_layout()
    fig.savefig(out_png, dpi=int(VIZ_DPI))
    plt.close(fig)
    print(f'[VIZ] saved {out_png}')


def _resolve_config_loader() -> Callable[[str | Path], dict]:
    if hasattr(config_yaml, 'load_yaml_config'):
        fn = config_yaml.load_yaml_config
        if callable(fn):
            return fn
    if hasattr(config_yaml, 'load_yaml'):
        fn = config_yaml.load_yaml
        if callable(fn):
            return fn
    msg = 'seisai_utils.config_yaml must expose load_yaml_config() or load_yaml()'
    raise AttributeError(msg)


def _resolve_ckpt_path(cfg: dict, cfg_yaml_path: Path) -> Path:
    if CKPT_PATH is not None:
        ckpt_path = CKPT_PATH.expanduser()
        if not ckpt_path.is_absolute():
            ckpt_path = (Path.cwd() / ckpt_path).resolve()
        if not ckpt_path.is_file():
            msg = f'checkpoint not found: {ckpt_path}'
            raise FileNotFoundError(msg)
        return ckpt_path

    paths = cfg.get('paths')
    if not isinstance(paths, dict):
        msg = 'config.paths must be dict'
        raise TypeError(msg)
    out_dir_val = paths.get('out_dir')
    if not isinstance(out_dir_val, str) or not out_dir_val.strip():
        msg = 'config.paths.out_dir must be non-empty str'
        raise ValueError(msg)

    out_dir = Path(out_dir_val).expanduser()
    if not out_dir.is_absolute():
        out_dir = cfg_yaml_path.parent / out_dir
    ckpt_path = (out_dir / 'ckpt' / 'best.pt').resolve()
    if not ckpt_path.is_file():
        msg = (
            f'checkpoint not found: {ckpt_path} '
            '(set CKPT_PATH constant explicitly if needed)'
        )
        raise FileNotFoundError(msg)
    return ckpt_path


def _resolve_device() -> torch.device:
    dev = torch.device(str(DEVICE))
    if dev.type == 'cuda' and not torch.cuda.is_available():
        msg = 'DEVICE is cuda but CUDA is not available'
        raise RuntimeError(msg)
    return dev


def load_psn_model_and_eps() -> tuple[torch.nn.Module, float, Path]:
    cfg_yaml_path = CFG_YAML.expanduser().resolve()
    if not cfg_yaml_path.is_file():
        msg = f'CFG_YAML not found: {cfg_yaml_path}'
        raise FileNotFoundError(msg)

    load_yaml_fn = _resolve_config_loader()
    cfg = load_yaml_fn(cfg_yaml_path)
    if not isinstance(cfg, dict):
        msg = f'loaded config must be dict, got {type(cfg).__name__}'
        raise TypeError(msg)

    typed = load_psn_train_config(cfg)
    ckpt_path = _resolve_ckpt_path(cfg, cfg_yaml_path)
    ckpt = load_checkpoint(ckpt_path)
    if ckpt['pipeline'] != 'psn':
        msg = f'checkpoint pipeline must be "psn", got {ckpt["pipeline"]!r}'
        raise ValueError(msg)

    model_sig = ckpt['model_sig']
    expected_sig = asdict(typed.model)
    if model_sig != expected_sig:
        msg = 'checkpoint model_sig does not match CFG_YAML model definition'
        raise ValueError(msg)

    transform_cfg = cfg.get('transform', {})
    if not isinstance(transform_cfg, dict):
        msg = 'config.transform must be dict'
        raise TypeError(msg)
    standardize_eps = float(transform_cfg.get('standardize_eps', 1.0e-8))
    if standardize_eps <= 0.0:
        msg = f'transform.standardize_eps must be > 0, got {standardize_eps}'
        raise ValueError(msg)

    device = _resolve_device()
    model = build_psn_model(typed.model)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.to(device)
    model.eval()
    return model, float(standardize_eps), ckpt_path


@torch.no_grad()
def infer_pick512_from_win(
    *,
    model: torch.nn.Module,
    wave_hw: np.ndarray,
    standardize_eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(wave_hw, dtype=np.float32)
    if x.ndim != 2:
        msg = f'wave_hw must be (H,W), got {x.shape}'
        raise ValueError(msg)

    h, w = x.shape
    if w != int(TILE_W):
        msg = f'win512 width mismatch: expected {TILE_W}, got {w}'
        raise ValueError(msg)

    x_chw = x[None, :, :]
    tile_h = min(int(TILE_H), int(h))
    overlap_h = min(int(OVERLAP_H), max(tile_h - 1, 0))

    logits = infer_tiled_chw(
        model,
        x_chw,
        tile=(int(tile_h), int(TILE_W)),
        overlap=(int(overlap_h), int(OVERLAP_W)),
        amp=bool(USE_AMP),
        use_tqdm=bool(USE_TQDM),
        tiles_per_batch=int(TILES_PER_BATCH),
        tile_transform=TilePerTraceStandardize(eps_std=float(standardize_eps)),
        post_tile_transform=None,
    )

    if int(logits.ndim) != 3:
        msg = f'logits must be (C,H,W), got shape={tuple(logits.shape)}'
        raise ValueError(msg)
    if int(logits.shape[1]) != int(h) or int(logits.shape[2]) != int(TILE_W):
        msg = (
            f'logits shape mismatch: got {tuple(logits.shape)}, '
            f'expected (*,{h},{TILE_W})'
        )
        raise ValueError(msg)
    if int(logits.shape[0]) <= int(P_CLASS_INDEX):
        msg = f'P_CLASS_INDEX={P_CLASS_INDEX} is out of range for logits={tuple(logits.shape)}'
        raise ValueError(msg)

    probs = torch.softmax(logits, dim=0)
    prob_p = probs[int(P_CLASS_INDEX)]  # (H, 512)
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
) -> None:
    rel = raw_path.relative_to(IN_RAW_SEGY_ROOT)
    out_dir = OUT_PRED_ROOT / rel.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    out_npz = out_dir / f'{raw_path.stem}.psn_pred.npz'
    out_crd = out_dir / f'{raw_path.stem}.fb.crd'

    with (
        segyio.open(str(raw_path), 'r', ignore_geometry=True) as raw,
        segyio.open(str(win_path), 'r', ignore_geometry=True) as win,
    ):
        n_tr_raw = int(raw.tracecount)
        n_tr_win = int(win.tracecount)
        if n_tr_raw != n_tr_win:
            msg = (
                f'tracecount mismatch raw={n_tr_raw} win512={n_tr_win} '
                f'raw={raw_path} win={win_path}'
            )
            raise ValueError(msg)
        n_traces = int(n_tr_raw)
        if n_traces <= 0:
            msg = f'no traces in raw segy: {raw_path}'
            raise ValueError(msg)

        n_samples_raw = int(raw.samples.size)
        n_samples_win = int(win.samples.size)
        if n_samples_raw <= 0 or n_samples_win <= 0:
            msg = (
                f'invalid n_samples raw={n_samples_raw} win={n_samples_win} '
                f'raw={raw_path} win={win_path}'
            )
            raise ValueError(msg)
        if n_samples_win != int(TILE_W):
            msg = f'win512 segy must have {TILE_W} samples, got {n_samples_win}: {win_path}'
            raise ValueError(msg)

        dt_us_raw = int(raw.bin[segyio.BinField.Interval])
        dt_us_win = int(win.bin[segyio.BinField.Interval])
        if dt_us_raw <= 0 or dt_us_win <= 0:
            msg = f'invalid dt_us raw={dt_us_raw} win={dt_us_win}'
            raise ValueError(msg)
        dt_sec_raw = float(dt_us_raw) * 1.0e-6
        dt_sec_win = float(dt_us_win) * 1.0e-6

        ffid_values = read_trace_field(
            raw,
            segyio.TraceField.FieldRecord,
            dtype=np.int32,
            name='raw ffid_values',
        )
        chno_values = read_trace_field(
            raw,
            segyio.TraceField.TraceNumber,
            dtype=np.int32,
            name='raw chno_values',
        )
        offsets = read_trace_field(
            raw,
            segyio.TraceField.offset,
            dtype=np.float32,
            name='raw offsets',
        )

        ffid_win = read_trace_field(
            win,
            segyio.TraceField.FieldRecord,
            dtype=np.int32,
            name='win ffid_values',
        )
        chno_win = read_trace_field(
            win,
            segyio.TraceField.TraceNumber,
            dtype=np.int32,
            name='win chno_values',
        )

        if not np.array_equal(ffid_values, ffid_win):
            msg = f'raw/win ffid arrays differ (index mapping would break): {raw_path}'
            raise ValueError(msg)
        if not np.array_equal(chno_values, chno_win):
            msg = f'raw/win chno arrays differ (index mapping would break): {raw_path}'
            raise ValueError(msg)

        window_start_i = _load_sidecar_window_start(
            sidecar_path=sidecar_path,
            n_traces=n_traces,
            n_samples_in=n_samples_raw,
            n_samples_out=n_samples_win,
            dt_sec_in=dt_sec_raw,
            dt_sec_out=dt_sec_win,
        )

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
            set(ffids_sorted[:: int(VIZ_EVERY_N_SHOTS)])
            if int(VIZ_EVERY_N_SHOTS) > 0
            else set()
        )
        viz_dir = out_dir / str(VIZ_DIRNAME)

        max_chno = int(chno_values.max(initial=0))
        ffid_to_row = {ff: i for i, ff in enumerate(ffids_sorted)}
        fb_mat = np.zeros((len(ffids_sorted), max_chno), dtype=np.int32)

        for gather_i, ffid in enumerate(ffids_sorted):
            idx0 = ffid_to_indices[int(ffid)]
            ch = chno_values[idx0].astype(np.int64, copy=False)
            order = np.argsort(ch, kind='mergesort')
            idx = idx0[order]
            h_g = int(idx.size)
            if h_g < int(MIN_GATHER_H):
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
                    f'H={h_g} < {MIN_GATHER_H} -> set picks=0'
                )
                continue
            chno_g = chno_values[idx].astype(np.int32, copy=False)
            offs_m = offsets[idx].astype(np.float32, copy=False)
            win_g = _load_traces_by_indices(win, idx)  # (H, 512)
            raw_g = _load_traces_by_indices(raw, idx)  # (H, ns_raw)
            wave_max_g = np.max(np.abs(raw_g), axis=1).astype(np.float32, copy=False)
            invalid_trace_g = (offs_m == 0.0) | (wave_max_g == 0.0)

            pick512_g, pmax_g = infer_pick512_from_win(
                model=model,
                wave_hw=win_g,
                standardize_eps=standardize_eps,
            )

            win_start_g = window_start_i[idx].astype(np.float32, copy=False)
            pick_orig_f_g = win_start_g + pick512_g.astype(
                np.float32, copy=False
            ) / float(UP_FACTOR)
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

            x_rs = build_pick_aligned_window(
                raw_g,
                picks=pick_orig_i_g,
                pre=int(RS_PRE),
                post=int(RS_POST),
                fill=0.0,
            )
            rs_res = refine_firstbreak_residual_statics(
                x_rs,
                max_lag=int(RS_MAX_LAG),
                k_neighbors=int(RS_K_NEIGHBORS),
                n_iter=int(RS_N_ITER),
                mode=str(RS_MODE),
                c_th=float(RS_C_TH),
                smooth_method=str(RS_SMOOTH_METHOD),
                lam=float(RS_LAM),
                subsample=bool(RS_SUBSAMPLE),
                propagate_low_corr=bool(RS_PROPAGATE_LOW_CORR),
                taper=str(RS_TAPER),
                taper_power=float(RS_TAPER_POWER),
                lag_penalty=float(RS_LAG_PENALTY),
                lag_penalty_power=float(RS_LAG_PENALTY_POWER),
            )

            delta_g = np.asarray(rs_res['delta_pick'], dtype=np.float32)
            cmax_g = np.asarray(rs_res['cmax'], dtype=np.float32)
            valid_g = np.asarray(rs_res['valid_mask'], dtype=bool)
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

            dbg = bool(POST_TROUGH_DEBUG) and (
                int(POST_TROUGH_DEBUG_EVERY_N_GATHERS) > 0
                and (int(gather_i) % int(POST_TROUGH_DEBUG_EVERY_N_GATHERS) == 0)
            )
            dbg_label = f'{raw_path.name} ffid={ffid}'

            # apply post-trough refinement only within offset window
            pt_mask = _post_trough_apply_mask_from_offsets(offs_m)
            if np.any(invalid_trace_g):
                pt_mask = pt_mask & (~invalid_trace_g)

            pick_final_g = _post_trough_adjust_picks(
                pick_rs_i_g.copy(),
                raw_g,
                max_shift=int(POST_TROUGH_MAX_SHIFT),
                scan_ahead=int(POST_TROUGH_SCAN_AHEAD),
                smooth_win=int(POST_TROUGH_SMOOTH_WIN),
                a_th=float(POST_TROUGH_A_TH),
                apply_mask=pt_mask,
                debug=dbg,
                debug_label=dbg_label,
                debug_max_examples=int(POST_TROUGH_DEBUG_MAX_EXAMPLES),
            ).astype(np.int32, copy=False)

            pick_final_g = _align_post_trough_shifts_to_neighbors(
                pick_rs_i_g,
                pick_final_g,
                radius=int(POST_TROUGH_OUTLIER_RADIUS),
                min_support=int(POST_TROUGH_OUTLIER_MIN_SUPPORT),
                max_dev=int(POST_TROUGH_OUTLIER_MAX_DEV),
                max_shift=int(POST_TROUGH_MAX_SHIFT),
                propagate_zero=bool(POST_TROUGH_ALIGN_PROPAGATE_ZERO),
                apply_mask=pt_mask,
                debug=dbg,
                debug_label=dbg_label,
            ).astype(np.int32, copy=False)

            # final cosmetic snap to phase
            np.clip(pick_final_g, 0, int(n_samples_raw - 1), out=pick_final_g)
            pick_final_g = snap_picks_to_phase(
                pick_final_g,
                raw_g,
                mode=str(SNAP_MODE),
                ltcor=int(SNAP_LTCOR),
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
                _save_gather_viz(
                    out_png=out_png,
                    raw_wave_hw=raw_g,
                    offsets_m=offs_m,
                    dt_sec=float(dt_sec_raw),
                    pick_psn_orig_i=pick_orig_i_g,
                    pick_rs_i=pick_rs_i_g,
                    pick_final_i=pick_final_g,
                    title=f'{raw_path.stem} ffid={int(ffid)}',
                )

            if LOG_GATHER_RS:
                mean_cmax = float(np.mean(cmax_g)) if cmax_g.size > 0 else 0.0
                n_valid = int(np.count_nonzero(valid_g))
                n_forced0 = int(np.count_nonzero(invalid_trace_g))
                print(
                    f'[RS] {raw_path.name} ffid={ffid} '
                    f'n_valid={n_valid}/{valid_g.size} forced_zero={n_forced0} '
                    f'mean_cmax={mean_cmax:.3f}'
                )

    trace_indices = np.arange(n_traces, dtype=np.int64)
    np.savez_compressed(
        out_npz,
        dt_sec=np.float32(dt_sec_raw),
        n_samples_orig=np.int32(n_samples_raw),
        n_traces=np.int32(n_traces),
        ffid_values=ffid_values.astype(np.int32, copy=False),
        chno_values=chno_values.astype(np.int32, copy=False),
        offsets=offsets.astype(np.float32, copy=False),
        trace_indices=trace_indices,
        pick_psn512=pick_psn512.astype(np.int32, copy=False),
        pmax_psn=pmax_psn.astype(np.float32, copy=False),
        window_start_i=window_start_i.astype(np.int64, copy=False),
        pick_psn_orig_f=pick_psn_orig_f.astype(np.float32, copy=False),
        pick_psn_orig_i=pick_psn_orig_i.astype(np.int32, copy=False),
        delta_pick_rs=delta_pick_rs.astype(np.float32, copy=False),
        cmax_rs=cmax_rs.astype(np.float32, copy=False),
        rs_valid_mask=rs_valid_mask.astype(bool, copy=False),
        pick_rs_i=pick_rs_i.astype(np.int32, copy=False),
        pick_final=pick_final.astype(np.int32, copy=False),
    )

    numpy2fbcrd(
        dt=float(dt_sec_raw * 1000.0),
        fbnum=fb_mat,
        gather_range=ffids_sorted,
        output_name=str(out_crd),
        original=None,
        mode='gather',
        header_comment='machine learning fb pick',
    )

    print(
        f'[OK] {raw_path.name} -> {out_npz.name}, {out_crd.name} '
        f'(traces={n_traces}, gathers={len(ffids_sorted)})'
    )


def main() -> None:
    model, standardize_eps, ckpt_path = load_psn_model_and_eps()
    raw_segys = find_segy_files(IN_RAW_SEGY_ROOT, exts=SEGY_EXTS, recursive=True)
    win_lookup = _build_win512_lookup(IN_WIN512_SEGY_ROOT)

    print(
        f'[RUN] raw={len(raw_segys)} win_lookup={len(win_lookup)} '
        f'ckpt={ckpt_path} device={next(model.parameters()).device}'
    )

    n_ok = 0
    n_skip = 0

    for raw_path in raw_segys:
        rel = raw_path.relative_to(IN_RAW_SEGY_ROOT)
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
        )
        n_ok += 1

    print(f'[DONE] processed={n_ok} skipped={n_skip} out_root={OUT_PRED_ROOT}')


if __name__ == '__main__':
    main()

# %%
