# %%
#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import segyio

# =========================
# CONFIG（ここだけ触ればOK）
# =========================
IN_SEGY_ROOT = Path('/home/dcuser/data/ActiveSeisField/jogsarar')
IN_INFER_ROOT = Path('/home/dcuser/data/ActiveSeisField/jogsarar_out')
OUT_SEGY_ROOT = Path('/home/dcuser/data/ActiveSeisField/jogsarar_psn512')

SEGY_EXTS = ('.sgy', '.segy', '.SGY', '.SEGY')

HALF_WIN = 128  # ±128 => 256
UP_FACTOR = 2  # 256 -> 512
OUT_NS = 2 * HALF_WIN * UP_FACTOR  # 512

# 下位除外（p10）
DROP_LOW_FRAC = 0.10  # 下位10%除外
SCORE_KEYS = (
    'conf_prob1',
    'conf_trend1',
    'conf_rs1',
)  # infer npz側にある前提（p1成分）
PICK_KEY = 'pick_final'  # infer npz側の最終pick（p1）

# 閾値モード
#   'per_segy' : SEGYごとにp10
#   'global'   : 全SEGYまとめてp10（全データグローバル）
THRESH_MODE = 'global'  # 'global' or 'per_segy'


# =========================
# Utility
# =========================
def find_segy_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for ext in SEGY_EXTS:
        files.extend(root.rglob(f'*{ext}'))
    files = sorted(set(files))
    if not files:
        msg = f'no segy files under: {root}'
        raise FileNotFoundError(msg)
    return files


def infer_npz_path_for_segy(segy_path: Path) -> Path:
    rel = segy_path.relative_to(IN_SEGY_ROOT)
    return IN_INFER_ROOT / rel.parent / f'{segy_path.stem}.prob.npz'


def out_segy_path_for_in(segy_path: Path) -> Path:
    rel = segy_path.relative_to(IN_SEGY_ROOT)
    out_rel = rel.with_suffix('')  # drop ext
    return OUT_SEGY_ROOT / out_rel.parent / f'{out_rel.name}.win512.sgy'


def out_sidecar_npz_path_for_out(out_segy_path: Path) -> Path:
    return out_segy_path.with_suffix('.sidecar.npz')


def _npz_must_have(z: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    if key not in z.files:
        msg = f'npz missing key: {key}  available={sorted(z.files)}'
        raise KeyError(msg)
    return np.asarray(z[key])


def _percentile_threshold(x: np.ndarray, *, frac: float) -> float:
    v = np.asarray(x, dtype=np.float32)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float('nan')
    q = float(frac) * 100.0
    return float(np.nanpercentile(v, q))


def _valid_pick_mask(pick_i: np.ndarray, *, n_samples: int) -> np.ndarray:
    p = np.asarray(pick_i)
    return np.isfinite(p) & (p > 0) & (p < int(n_samples))


def _load_trend_center_i(
    z: np.lib.npyio.NpzFile, *, n_traces: int, dt_sec_from_segy: float
) -> np.ndarray:
    """Return per-trace trend center in ORIGINAL sample index (float32), shape (n_traces,).

    Accepted keys:
      - 'trend_center_i' : sample index (recommended)
      - 'trend_t_sec'    : trend time in seconds
      - 't_trend_sec'    : same meaning (alt name)
      - 'trend_center_sec' : same meaning (alt name)
    """
    keys = set(z.files)

    if 'trend_center_i' in keys:
        c = np.asarray(z['trend_center_i'], dtype=np.float32)
    else:
        if 'trend_t_sec' in keys:
            t = np.asarray(z['trend_t_sec'], dtype=np.float32)
        elif 't_trend_sec' in keys:
            t = np.asarray(z['t_trend_sec'], dtype=np.float32)
        elif 'trend_center_sec' in keys:
            t = np.asarray(z['trend_center_sec'], dtype=np.float32)
        else:
            msg = (
                "npz needs one of: 'trend_center_i', 'trend_t_sec', 't_trend_sec', 'trend_center_sec'. "
                f'available={sorted(keys)}'
            )
            raise KeyError(msg)

        if t.ndim == 2 and t.shape[0] == 1:
            t = t[0]
        if t.ndim != 1:
            msg = f'trend time must be 1D (n_traces,), got {t.shape}'
            raise ValueError(msg)

        if 'dt_sec' in keys:
            dt = float(np.asarray(z['dt_sec']).item())
        else:
            dt = float(dt_sec_from_segy)
        if dt <= 0.0:
            msg = f'dt_sec must be positive, got {dt}'
            raise ValueError(msg)

        c = t / dt

    if c.ndim == 2 and c.shape[0] == 1:
        c = c[0]

    if c.ndim != 1 or c.shape[0] != n_traces:
        msg = f'trend center must be (n_traces,), got {c.shape}, n_traces={n_traces}'
        raise ValueError(msg)

    return c.astype(np.float32, copy=False)


def _fill_trend_center_i(trend_center_i: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fill missing/invalid trend center with nearest valid value along trace index.

    Valid condition: finite and > 0.

    Returns:
      - filled_center_i (float32) shape (n_traces,)
      - filled_mask (bool): True where original was invalid and got filled

    """
    c0 = np.asarray(trend_center_i, dtype=np.float32)
    if c0.ndim != 1:
        msg = f'trend_center_i must be 1D, got {c0.shape}'
        raise ValueError(msg)

    valid = np.isfinite(c0) & (c0 > 0.0)
    idx_valid = np.flatnonzero(valid)
    if idx_valid.size == 0:
        filled = c0.copy()
        filled_mask = np.zeros_like(valid, dtype=bool)
        return filled, filled_mask

    filled = c0.copy()
    inv_idx = np.flatnonzero(~valid)
    m = int(idx_valid.size)

    for i in inv_idx:
        pos = int(np.searchsorted(idx_valid, int(i)))
        if pos <= 0:
            j = int(idx_valid[0])
        elif pos >= m:
            j = int(idx_valid[m - 1])
        else:
            j0 = int(idx_valid[pos - 1])
            j1 = int(idx_valid[pos])
            j = j0 if (i - j0) <= (j1 - i) else j1
        filled[i] = filled[j]

    filled_mask = ~valid
    return filled.astype(np.float32, copy=False), filled_mask.astype(bool, copy=False)


def _field_key_to_int(key: object) -> int:
    if isinstance(key, (int, np.integer)):
        return int(key)
    v = getattr(key, 'value', None)
    if isinstance(v, (int, np.integer)):
        return int(v)
    try:
        return int(key)
    except (TypeError, ValueError) as e:
        msg = f'cannot convert segy field key to int: {key!r} (type={type(key)})'
        raise TypeError(msg) from e


def _extract_256(trace: np.ndarray, *, center_i: float) -> tuple[np.ndarray, int]:
    x = np.asarray(trace, dtype=np.float32)
    if x.ndim != 1:
        msg = f'trace must be 1D, got {x.shape}'
        raise ValueError(msg)

    out = np.zeros(2 * HALF_WIN, dtype=np.float32)

    if (not np.isfinite(center_i)) or center_i <= 0.0:
        return out, -1

    c = int(np.rint(float(center_i)))
    start = c - HALF_WIN
    end = c + HALF_WIN

    n = x.shape[0]
    ov_l = max(0, start)
    ov_r = min(n, end)
    if ov_l >= ov_r:
        return out, start

    dst_l = ov_l - start
    dst_r = dst_l + (ov_r - ov_l)
    out[dst_l:dst_r] = x[ov_l:ov_r]
    return out, start


def _upsample_256_to_512_linear(win256: np.ndarray) -> np.ndarray:
    y = np.asarray(win256, dtype=np.float32)
    if y.shape != (2 * HALF_WIN,):
        msg = f'win256 must be (256,), got {y.shape}'
        raise ValueError(msg)

    xp = np.arange(2 * HALF_WIN, dtype=np.float32)  # 0..255
    xq = np.arange(OUT_NS, dtype=np.float32) / 2.0  # 0,0.5,..,255.5
    out = np.interp(xq, xp, y, left=0.0, right=0.0).astype(np.float32, copy=False)
    if out.shape != (OUT_NS,):
        msg = f'unexpected upsample output shape: {out.shape}'
        raise ValueError(msg)
    return out


def _base_valid_mask(
    *,
    pick_final_i: np.ndarray,
    trend_center_i: np.ndarray,
    n_samples_in: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (base_valid, reason_mask_partial)
    reason bits:
      bit0: invalid pick
      bit1: trend missing (even after fill)
      bit2: pick outside ±128 window
    """
    n_tr = int(pick_final_i.shape[0])
    if trend_center_i.shape != (n_tr,):
        msg = f'trend_center_i mismatch: {trend_center_i.shape} vs {(n_tr,)}'
        raise ValueError(msg)

    p = np.asarray(pick_final_i, dtype=np.int64)
    c = np.asarray(trend_center_i, dtype=np.float32)

    reason = np.zeros(n_tr, dtype=np.uint8)

    pick_ok = _valid_pick_mask(p, n_samples=n_samples_in)
    reason[~pick_ok] |= 1 << 0

    trend_ok = np.isfinite(c) & (c > 0.0)
    reason[~trend_ok] |= 1 << 1

    c_round = np.rint(c).astype(np.int64, copy=False)
    win_ok = pick_ok & trend_ok & (np.abs(p - c_round) <= int(HALF_WIN))
    reason[~win_ok] |= 1 << 2

    return win_ok, reason


def build_keep_mask(
    *,
    pick_final_i: np.ndarray,
    trend_center_i: np.ndarray,
    n_samples_in: int,
    scores: dict[str, np.ndarray],
    thresholds: dict[str, float] | None,
) -> tuple[np.ndarray, dict[str, float], np.ndarray, np.ndarray]:
    """keep_mask, thresholds_used, reason_mask, base_valid

    reason bits:
      bit0: invalid pick
      bit1: trend missing
      bit2: pick outside ±128 window
      bit3: conf_prob low
      bit4: conf_trend low
      bit5: conf_rs low
    """
    n_tr = int(pick_final_i.shape[0])

    base_valid, reason = _base_valid_mask(
        pick_final_i=pick_final_i,
        trend_center_i=trend_center_i,
        n_samples_in=n_samples_in,
    )

    bit_map = {'conf_prob1': 3, 'conf_trend1': 4, 'conf_rs1': 5}

    thresholds_used: dict[str, float] = {}
    keep = base_valid.copy()

    for k in SCORE_KEYS:
        s = np.asarray(scores[k], dtype=np.float32)
        if s.shape != (n_tr,):
            msg = f'score {k} must be (n_traces,), got {s.shape}, n_traces={n_tr}'
            raise ValueError(msg)

        if thresholds is None:
            th = _percentile_threshold(s[base_valid], frac=DROP_LOW_FRAC)
        else:
            if k not in thresholds:
                msg = f'thresholds missing key: {k} available={sorted(thresholds)}'
                raise KeyError(msg)
            th = float(thresholds[k])

        thresholds_used[k] = th

        keep_k = np.zeros(n_tr, dtype=bool) if np.isnan(th) else (s >= th)

        low = base_valid & (~keep_k)
        reason[low] |= 1 << bit_map[k]
        keep &= keep_k

    return keep, thresholds_used, reason, base_valid


def _load_minimal_inputs_for_thresholds(
    segy_path: Path,
) -> tuple[
    int,
    int,
    float,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
]:
    """Load only what we need to compute base_valid and score arrays.

    Returns:
      n_traces, ns_in, dt_sec_in,
      pick_final (int64),
      trend_center_i_raw (float32),
      scores dict,
      trend_center_i_filled (float32),
      trend_filled_mask (bool)

    """
    infer_npz = infer_npz_path_for_segy(segy_path)
    if not infer_npz.exists():
        msg = f'infer npz not found for segy: {segy_path} expected={infer_npz}'
        raise FileNotFoundError(msg)

    with segyio.open(str(segy_path), 'r', ignore_geometry=True) as src:
        n_traces = int(src.tracecount)
        if n_traces <= 0:
            msg = f'no traces: {segy_path}'
            raise ValueError(msg)

        ns_in = int(src.samples.size)
        if ns_in <= 0:
            msg = f'invalid n_samples: {ns_in}'
            raise ValueError(msg)

        dt_us_in = int(src.bin[segyio.BinField.Interval])
        if dt_us_in <= 0:
            msg = f'invalid dt_us: {dt_us_in}'
            raise ValueError(msg)
        dt_sec_in = float(dt_us_in) * 1e-6

    z = np.load(infer_npz, allow_pickle=False)

    pick_final = _npz_must_have(z, PICK_KEY).astype(np.int64, copy=False)
    if pick_final.ndim != 1 or pick_final.shape[0] != n_traces:
        msg = f'{PICK_KEY} must be (n_traces,), got {pick_final.shape}, n_traces={n_traces}'
        raise ValueError(msg)

    trend_center_i_raw = _load_trend_center_i(
        z, n_traces=n_traces, dt_sec_from_segy=dt_sec_in
    )
    trend_center_i_filled, trend_filled_mask = _fill_trend_center_i(trend_center_i_raw)

    scores: dict[str, np.ndarray] = {}
    for k in SCORE_KEYS:
        scores[k] = _npz_must_have(z, k).astype(np.float32, copy=False)

    return (
        n_traces,
        ns_in,
        dt_sec_in,
        pick_final,
        trend_center_i_raw,
        scores,
        trend_center_i_filled,
        trend_filled_mask,
    )


def compute_global_thresholds(segys: list[Path]) -> dict[str, float]:
    """Compute global p10 thresholds over ALL SEGY files, using base_valid traces only."""
    vals: dict[str, list[np.ndarray]] = {k: [] for k in SCORE_KEYS}

    n_files_used = 0
    n_base_total = 0

    for p in segys:
        infer_npz = infer_npz_path_for_segy(p)
        if not infer_npz.exists():
            continue

        (
            n_traces,
            ns_in,
            _dt_sec_in,
            pick_final,
            _trend_raw,
            scores,
            trend_filled,
            _trend_filled_mask,
        ) = _load_minimal_inputs_for_thresholds(p)

        base_valid, _reason = _base_valid_mask(
            pick_final_i=pick_final,
            trend_center_i=trend_filled,
            n_samples_in=ns_in,
        )

        n_b = int(np.count_nonzero(base_valid))
        if n_b <= 0:
            continue

        for k in SCORE_KEYS:
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
    for k in SCORE_KEYS:
        if not vals[k]:
            msg = f'no values accumulated for score={k}'
            raise RuntimeError(msg)
        v = np.concatenate(vals[k]).astype(np.float32, copy=False)
        thresholds[k] = _percentile_threshold(v, frac=DROP_LOW_FRAC)

        if np.isnan(thresholds[k]):
            msg = f'global threshold became NaN for score={k}'
            raise RuntimeError(msg)

    print(
        f'[GLOBAL_TH] files_used={n_files_used} base_valid_total={n_base_total} '
        f'p10 prob={thresholds["conf_prob1"]:.6g} trend={thresholds["conf_trend1"]:.6g} rs={thresholds["conf_rs1"]:.6g}'
    )
    return thresholds


# =========================
# Main per-file processing
# =========================
def process_one_segy(
    segy_path: Path, *, global_thresholds: dict[str, float] | None
) -> None:
    infer_npz = infer_npz_path_for_segy(segy_path)
    if not infer_npz.exists():
        msg = f'infer npz not found for segy: {segy_path}  expected={infer_npz}'
        raise FileNotFoundError(msg)

    out_segy = out_segy_path_for_in(segy_path)
    out_segy.parent.mkdir(parents=True, exist_ok=True)
    side_npz = out_sidecar_npz_path_for_out(out_segy)

    with segyio.open(str(segy_path), 'r', ignore_geometry=True) as src:
        n_traces = int(src.tracecount)
        if n_traces <= 0:
            msg = f'no traces: {segy_path}'
            raise ValueError(msg)

        ns_in = int(src.samples.size)
        if ns_in <= 0:
            msg = f'invalid n_samples: {ns_in}'
            raise ValueError(msg)

        dt_us_in = int(src.bin[segyio.BinField.Interval])
        if dt_us_in <= 0:
            msg = f'invalid dt_us: {dt_us_in}'
            raise ValueError(msg)

        if dt_us_in % UP_FACTOR != 0:
            msg = f'dt_us must be divisible by {UP_FACTOR}. got {dt_us_in}'
            raise ValueError(msg)

        dt_us_out = dt_us_in // UP_FACTOR
        dt_sec_in = float(dt_us_in) * 1e-6
        dt_sec_out = float(dt_us_out) * 1e-6

        z = np.load(infer_npz, allow_pickle=False)

        pick_final = _npz_must_have(z, PICK_KEY).astype(np.int64, copy=False)
        if pick_final.ndim != 1 or pick_final.shape[0] != n_traces:
            msg = f'{PICK_KEY} must be (n_traces,), got {pick_final.shape}, n_traces={n_traces}'
            raise ValueError(msg)

        # trend: load -> fill (flatten用はfilledを使う)
        trend_center_i_raw = _load_trend_center_i(
            z, n_traces=n_traces, dt_sec_from_segy=dt_sec_in
        )
        trend_center_i, trend_filled_mask = _fill_trend_center_i(trend_center_i_raw)

        scores: dict[str, np.ndarray] = {}
        for k in SCORE_KEYS:
            scores[k] = _npz_must_have(z, k).astype(np.float32, copy=False)

        thresholds_arg = None
        if THRESH_MODE == 'global':
            thresholds_arg = global_thresholds
            if thresholds_arg is None:
                msg = 'THRESH_MODE=global but global_thresholds is None'
                raise RuntimeError(msg)
        elif THRESH_MODE == 'per_segy':
            thresholds_arg = None
        else:
            msg = f"THRESH_MODE must be 'global' or 'per_segy', got {THRESH_MODE!r}"
            raise ValueError(msg)

        keep_mask, thresholds_used, reason_mask, _base_valid = build_keep_mask(
            pick_final_i=pick_final,
            trend_center_i=trend_center_i,
            n_samples_in=ns_in,
            scores=scores,
            thresholds=thresholds_arg,
        )

        # pick position in new 512 coordinate (float32; for later CSR creation)
        c_round = np.rint(trend_center_i).astype(np.int64, copy=False)
        win_start_i = c_round - int(HALF_WIN)
        pick_win_512 = (
            pick_final.astype(np.float32) - win_start_i.astype(np.float32)
        ) * float(UP_FACTOR)
        pick_win_512[~keep_mask] = np.nan

        # SEGY output spec
        spec = segyio.spec()
        spec.tracecount = n_traces
        spec.samples = np.arange(OUT_NS, dtype=np.int32)
        spec.format = 5  # IEEE float32

        sorting_val = getattr(src, 'sorting', 1)
        spec.sorting = (
            int(sorting_val) if isinstance(sorting_val, (int, np.integer)) else 1
        )

        with segyio.create(str(out_segy), spec) as dst:
            dst.text[0] = src.text[0]

            for k in src.bin:
                dst.bin[_field_key_to_int(k)] = src.bin[k]
            dst.bin[_field_key_to_int(segyio.BinField.Interval)] = dt_us_out
            dst.bin[_field_key_to_int(segyio.BinField.Samples)] = OUT_NS

            for i in range(n_traces):
                h = {_field_key_to_int(k): v for k, v in dict(src.header[i]).items()}
                h[_field_key_to_int(segyio.TraceField.TRACE_SAMPLE_INTERVAL)] = (
                    dt_us_out
                )
                h[_field_key_to_int(segyio.TraceField.TRACE_SAMPLE_COUNT)] = OUT_NS
                dst.header[i] = h

                tr = np.asarray(src.trace[i], dtype=np.float32)
                w256, _start = _extract_256(tr, center_i=float(trend_center_i[i]))
                w512 = _upsample_256_to_512_linear(w256)
                dst.trace[i] = w512

            dst.flush()

    np.savez_compressed(
        side_npz,
        src_segy=str(segy_path),
        src_infer_npz=str(infer_npz),
        out_segy=str(out_segy),
        thresh_mode=str(THRESH_MODE),
        drop_low_frac=np.float32(DROP_LOW_FRAC),
        dt_sec_in=np.float32(dt_sec_in),
        dt_sec_out=np.float32(dt_sec_out),
        dt_us_in=np.int32(dt_us_in),
        dt_us_out=np.int32(dt_us_out),
        n_traces=np.int32(n_traces),
        n_samples_in=np.int32(ns_in),
        n_samples_out=np.int32(OUT_NS),
        trend_center_i_raw=trend_center_i_raw.astype(np.float32, copy=False),
        trend_center_i=trend_center_i.astype(np.float32, copy=False),
        trend_filled_mask=trend_filled_mask.astype(bool, copy=False),
        trend_center_i_round=c_round.astype(np.int64, copy=False),
        window_start_i=win_start_i.astype(np.int64, copy=False),
        pick_final_i=pick_final.astype(np.int64, copy=False),
        pick_win_512=pick_win_512.astype(np.float32, copy=False),
        keep_mask=keep_mask.astype(bool, copy=False),
        reason_mask=reason_mask.astype(np.uint8, copy=False),
        th_conf_prob1=np.float32(thresholds_used['conf_prob1']),
        th_conf_trend1=np.float32(thresholds_used['conf_trend1']),
        th_conf_rs1=np.float32(thresholds_used['conf_rs1']),
        conf_prob1=scores['conf_prob1'].astype(np.float32, copy=False),
        conf_trend1=scores['conf_trend1'].astype(np.float32, copy=False),
        conf_rs1=scores['conf_rs1'].astype(np.float32, copy=False),
    )

    n_keep = int(np.count_nonzero(keep_mask))
    n_fill = int(np.count_nonzero(trend_filled_mask))
    tag = 'global' if THRESH_MODE == 'global' else 'per_segy'
    print(
        f'[OK] {segy_path.name} -> {out_segy.name}  keep={n_keep}/{n_traces} '
        f'filled_trend={n_fill}/{n_traces} '
        f'th({tag} p10) prob={thresholds_used["conf_prob1"]:.6g} '
        f'trend={thresholds_used["conf_trend1"]:.6g} rs={thresholds_used["conf_rs1"]:.6g}'
    )


def main() -> None:
    segys = find_segy_files(IN_SEGY_ROOT)
    print(f'[RUN] found {len(segys)} segy files under {IN_SEGY_ROOT}')

    # infer npzがあるやつだけ対象
    segys2: list[Path] = []
    for p in segys:
        infer_npz = infer_npz_path_for_segy(p)
        if not infer_npz.exists():
            print(f'[SKIP] infer npz missing: {p}  expected={infer_npz}')
            continue
        segys2.append(p)

    global_thresholds = None
    if THRESH_MODE == 'global':
        global_thresholds = compute_global_thresholds(segys2)

    for p in segys2:
        process_one_segy(p, global_thresholds=global_thresholds)


if __name__ == '__main__':
    main()
