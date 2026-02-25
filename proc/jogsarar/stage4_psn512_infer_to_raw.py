# %%
#!/usr/bin/env python3

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
from seisai_models.models.encdec2d import EncDec2D
from seisai_pick.pickio.io_grstat import numpy2fbcrd
from seisai_pick.residual_statics import refine_firstbreak_residual_statics
from seisai_pick.snap_picks_to_phase import snap_picks_to_phase
from seisai_utils import config_yaml
from seisai_utils.viz_wiggle import PickOverlay, WiggleConfig, plot_wiggle

# =========================
# CONFIG (fixed constants)
# =========================
@dataclass(frozen=True)
class Stage4Cfg:
    in_raw_segy_root: Path = Path('/home/dcuser/data/ActiveSeisField/jogsarar')
    in_win512_segy_root: Path = Path('/home/dcuser/data/ActiveSeisField/jogsarar_psn512')
    out_pred_root: Path = Path('/home/dcuser/data/ActiveSeisField/jogsarar_psn512_pred')
    # cfg_yaml=None で ckpt-only 起動。DEFAULT では従来互換の YAML 起動。
    cfg_yaml: Path | None = Path('configs/config_convnext_prestage2_drop005.yaml')
    # cfg_yaml=None のとき ckpt_path は必須。
    # cfg_yaml!=None のときは、ckpt_path 指定が最優先 / 未指定なら YAML の paths.out_dir から解決。
    ckpt_path: Path | None = None
    # cfg_yaml=None (ckpt-only) のときに使う標準化eps。
    standardize_eps: float = 1.0e-8
    segy_exts: tuple[str, ...] = ('.sgy', '.segy')
    device: str = 'cuda'
    # PSN class order from seisai_dataset.builder.builder.PhasePSNMap:
    # [P, S, Noise] -> P index is fixed to 0.
    p_class_index: int = 0
    # win512 -> raw sample inverse mapping
    up_factor: float = 2.0
    # tiled inference
    tile_h: int = 128
    tile_w: int = 512
    overlap_h: int = 96
    overlap_w: int = 0
    tiles_per_batch: int = 8
    use_amp: bool = True
    use_tqdm: bool = False
    # lightweight residual statics + final snap
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
    # post-trough refinement
    post_trough_max_shift: int = 16
    post_trough_scan_ahead: int = 32
    post_trough_smooth_win: int = 5
    # post-trough refinement offset gating (ABS offset in meters)
    post_trough_offs_abs_min_m: float | None = None
    post_trough_offs_abs_max_m: float | None = 1500
    # Peak threshold on positive amplitude after per-trace normalization.
    post_trough_a_th: float = 0.03
    # align shifts to neighborhood
    post_trough_outlier_radius: int = 4
    post_trough_outlier_min_support: int = 3
    post_trough_outlier_max_dev: int = 2
    post_trough_align_propagate_zero: bool = False
    # debug prints for post-trough
    post_trough_debug: bool = False
    post_trough_debug_max_examples: int = 5
    post_trough_debug_every_n_gathers: int = 10
    dt_tol_sec: float = 1e-9
    # visualization (like run_fbp_infer.py)
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


def _stem_without_win512(stem: str) -> str:
    tag = '.win512'
    if stem.endswith(tag):
        return stem[: -len(tag)]
    return stem


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
    cfg: Stage4Cfg = DEFAULT_STAGE4_CFG,
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
    if abs(float(side_dt_in) - float(dt_sec_in)) > float(cfg.dt_tol_sec):
        msg = f'sidecar dt_sec_in mismatch: side={side_dt_in}, raw={dt_sec_in}'
        raise ValueError(msg)
    if abs(float(side_dt_out) - float(dt_sec_out)) > float(cfg.dt_tol_sec):
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
    cfg: Stage4Cfg = DEFAULT_STAGE4_CFG,
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

    start = int(max(0, cfg.viz_plot_start))
    end_cfg = int(cfg.viz_plot_end)
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
    fig, ax = plt.subplots(figsize=cfg.viz_figsize)
    plot_wiggle(
        wave_plot,
        ax=ax,
        cfg=WiggleConfig(
            dt=float(dt_sec),
            t0=float(start) * float(dt_sec),
            time_axis=1,
            x=x_keep,
            normalize='trace',
            gain=float(cfg.viz_gain),
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
    fig.savefig(out_png, dpi=int(cfg.viz_dpi))
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


def _resolve_explicit_ckpt_path(path: Path) -> Path:
    ckpt_path = path.expanduser()
    if not ckpt_path.is_absolute():
        ckpt_path = (Path.cwd() / ckpt_path).resolve()
    if not ckpt_path.is_file():
        msg = f'checkpoint not found: {ckpt_path}'
        raise FileNotFoundError(msg)
    return ckpt_path


def _resolve_ckpt_path(
    loaded_cfg: dict | None,
    cfg_yaml_path: Path | None,
    *,
    cfg: Stage4Cfg = DEFAULT_STAGE4_CFG,
) -> Path:
    if cfg_yaml_path is None:
        if cfg.ckpt_path is None:
            msg = 'ckpt_path must be set when cfg_yaml is None'
            raise ValueError(msg)
        return _resolve_explicit_ckpt_path(cfg.ckpt_path)

    if cfg.ckpt_path is not None:
        return _resolve_explicit_ckpt_path(cfg.ckpt_path)

    if loaded_cfg is None:
        msg = 'loaded_cfg must be provided when cfg_yaml is set'
        raise ValueError(msg)

    paths = loaded_cfg.get('paths')
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
            '(set cfg.ckpt_path explicitly if needed)'
        )
        raise FileNotFoundError(msg)
    return ckpt_path


def _resolve_device(*, cfg: Stage4Cfg = DEFAULT_STAGE4_CFG) -> torch.device:
    dev = torch.device(str(cfg.device))
    if dev.type == 'cuda' and not torch.cuda.is_available():
        msg = 'cfg.device is cuda but CUDA is not available'
        raise RuntimeError(msg)
    return dev


def load_psn_model_and_eps(
    *, cfg: Stage4Cfg = DEFAULT_STAGE4_CFG
) -> tuple[torch.nn.Module, float, Path]:
    # cfg_yaml あり:
    #   - YAML→typed config→model_sig照合→epsはYAML(transform.standardize_eps)を使用。
    # cfg_yaml なし:
    #   - ckpt-only 起動。ckpt_path 必須、model_sig から EncDec2D を直接構築、
    #     pretrained 系キーがあれば False に強制して外部DLを抑止、
    #     epsは cfg.standardize_eps を使用。
    if cfg.cfg_yaml is not None:
        cfg_yaml_path = cfg.cfg_yaml.expanduser().resolve()
        if not cfg_yaml_path.is_file():
            msg = f'cfg_yaml not found: {cfg_yaml_path}'
            raise FileNotFoundError(msg)

        load_yaml_fn = _resolve_config_loader()
        loaded_cfg = load_yaml_fn(cfg_yaml_path)
        if not isinstance(loaded_cfg, dict):
            msg = f'loaded config must be dict, got {type(loaded_cfg).__name__}'
            raise TypeError(msg)

        typed = load_psn_train_config(loaded_cfg)
        ckpt_path = _resolve_ckpt_path(loaded_cfg, cfg_yaml_path, cfg=cfg)
        ckpt = load_checkpoint(ckpt_path)
        if ckpt['pipeline'] != 'psn':
            msg = f'checkpoint pipeline must be "psn", got {ckpt["pipeline"]!r}'
            raise ValueError(msg)

        model_sig = ckpt['model_sig']
        expected_sig = asdict(typed.model)
        if model_sig != expected_sig:
            msg = 'checkpoint model_sig does not match cfg_yaml model definition'
            raise ValueError(msg)

        transform_cfg = loaded_cfg.get('transform', {})
        if not isinstance(transform_cfg, dict):
            msg = 'config.transform must be dict'
            raise TypeError(msg)
        standardize_eps = float(transform_cfg.get('standardize_eps', 1.0e-8))
        if standardize_eps <= 0.0:
            msg = f'transform.standardize_eps must be > 0, got {standardize_eps}'
            raise ValueError(msg)
        model = build_psn_model(typed.model)
    else:
        ckpt_path = _resolve_ckpt_path(None, None, cfg=cfg)
        ckpt = load_checkpoint(ckpt_path)
        if ckpt['pipeline'] != 'psn':
            msg = f'checkpoint pipeline must be "psn", got {ckpt["pipeline"]!r}'
            raise ValueError(msg)
        model_sig = ckpt['model_sig']
        if not isinstance(model_sig, dict):
            msg = f'checkpoint model_sig must be dict, got {type(model_sig).__name__}'
            raise TypeError(msg)
        model_kwargs = dict(model_sig)
        if 'pretrained' in model_kwargs:
            model_kwargs['pretrained'] = False
        if 'backbone_pretrained' in model_kwargs:
            # Older signatures may carry backbone_pretrained; EncDec2D expects pretrained.
            if 'pretrained' not in model_kwargs:
                model_kwargs['pretrained'] = False
            model_kwargs.pop('backbone_pretrained')
        standardize_eps = float(cfg.standardize_eps)
        if standardize_eps <= 0.0:
            msg = f'cfg.standardize_eps must be > 0, got {standardize_eps}'
            raise ValueError(msg)
        model = EncDec2D(**model_kwargs)

    device = _resolve_device(cfg=cfg)
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
    rel = raw_path.relative_to(cfg.in_raw_segy_root)
    out_dir = cfg.out_pred_root / rel.parent
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
        if n_samples_win != int(cfg.tile_w):
            msg = (
                f'win512 segy must have {cfg.tile_w} samples, '
                f'got {n_samples_win}: {win_path}'
            )
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
            cfg=cfg,
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
            win_g = _load_traces_by_indices(win, idx)  # (H, 512)
            raw_g = _load_traces_by_indices(raw, idx)  # (H, ns_raw)
            wave_max_g = np.max(np.abs(raw_g), axis=1).astype(np.float32, copy=False)
            invalid_trace_g = (offs_m == 0.0) | (wave_max_g == 0.0)

            pick512_g, pmax_g = infer_pick512_from_win(
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
                _replace_edge_picks_if_far(
                    pick512_g,
                    pmax_g,
                    pick_orig_f_g,
                    pick_orig_i_g,
                    max_gap_samples=int(cfg.edge_pick_max_gap_samples),
                )
            )

            x_rs = build_pick_aligned_window(
                raw_g,
                picks=pick_orig_i_g,
                pre=int(cfg.rs_pre),
                post=int(cfg.rs_post),
                fill=0.0,
            )
            rs_res = refine_firstbreak_residual_statics(
                x_rs,
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

            dbg = bool(cfg.post_trough_debug) and (
                int(cfg.post_trough_debug_every_n_gathers) > 0
                and (
                    int(gather_i) % int(cfg.post_trough_debug_every_n_gathers) == 0
                )
            )
            dbg_label = f'{raw_path.name} ffid={ffid}'

            # apply post-trough refinement only within offset window
            pt_mask = _post_trough_apply_mask_from_offsets(offs_m, cfg=cfg)
            if np.any(invalid_trace_g):
                pt_mask = pt_mask & (~invalid_trace_g)

            pick_final_g = _post_trough_adjust_picks(
                pick_rs_i_g.copy(),
                raw_g,
                max_shift=int(cfg.post_trough_max_shift),
                scan_ahead=int(cfg.post_trough_scan_ahead),
                smooth_win=int(cfg.post_trough_smooth_win),
                a_th=float(cfg.post_trough_a_th),
                apply_mask=pt_mask,
                debug=dbg,
                debug_label=dbg_label,
                debug_max_examples=int(cfg.post_trough_debug_max_examples),
            ).astype(np.int32, copy=False)

            pick_final_g = _align_post_trough_shifts_to_neighbors(
                pick_rs_i_g,
                pick_final_g,
                radius=int(cfg.post_trough_outlier_radius),
                min_support=int(cfg.post_trough_outlier_min_support),
                max_dev=int(cfg.post_trough_outlier_max_dev),
                max_shift=int(cfg.post_trough_max_shift),
                propagate_zero=bool(cfg.post_trough_align_propagate_zero),
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
                _save_gather_viz(
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


def main() -> None:
    run_stage4(cfg=DEFAULT_STAGE4_CFG)


if __name__ == '__main__':
    main()

# %%
