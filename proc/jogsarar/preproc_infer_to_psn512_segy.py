# %%
#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import segyio

# =========================
# CONFIG（ここだけ触ればOK）
# =========================
IN_SEGY_ROOT = Path(
    '/home/dcuser/data/ActiveSeisField/jogsarar'
)  # infer済みSEGYがある場所（再帰で探す）
IN_INFER_ROOT = Path(
    '/home/dcuser/data/ActiveSeisField/jogsarar_out'
)  # run_fbp_infer.py が作った *.prob.npz がある場所（同じ相対パス想定）
OUT_SEGY_ROOT = Path(
    '/home/dcuser/data/ActiveSeisField/jogsarar_psn512'
)  # 新SEGYの出力先（相対パスを維持）

SEGY_EXTS = ('.sgy', '.segy', '.SGY', '.SEGY')

HALF_WIN = 128  # ±128 => 256
UP_FACTOR = 2  # 256 -> 512
OUT_NS = 2 * HALF_WIN * UP_FACTOR  # 512

# 下位除外
DROP_LOW_FRAC = 0.10  # 下位10%除外
SCORE_KEYS = (
    'conf_prob1',
    'conf_trend1',
    'conf_rs1',
)  # infer npz側にある前提（p1の成分）
PICK_KEY = 'pick_final'  # infer npz側の最終pick（p1）
# トレンド中心（推奨）
TREND_CENTER_KEY = 'trend_center_i'  # (n_traces,)


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


def _npz_must_have(z: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    if key not in z.files:
        msg = f'npz missing key: {key}  available={sorted(z.files)}'
        raise KeyError(msg)
    return np.asarray(z[key])


def _load_trend_center_i(
    z: np.lib.npyio.NpzFile, *, n_traces: int, dt_sec_from_segy: float
) -> np.ndarray:
    """Return per-trace trend center in ORIGINAL sample index (float32), shape (n_traces,).

    Accepted keys:
      - 'trend_center_i' : sample index (recommended)
      - 'trend_t_sec'    : trend time in seconds (your current likely output)
      - 't_trend_sec'    : same meaning (alt name)
      - 'trend_center_sec' : same meaning (alt name)

    dt_sec source priority:
      - z['dt_sec'] if exists
      - dt_sec_from_segy otherwise
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
            raise KeyError(
                msg
            )

        # allow (1, n_traces) -> (n_traces,)
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

    # allow (1, n_traces) -> (n_traces,) also for center
    if c.ndim == 2 and c.shape[0] == 1:
        c = c[0]

    if c.ndim != 1 or c.shape[0] != n_traces:
        msg = f'trend center must be (n_traces,), got {c.shape}, n_traces={n_traces}'
        raise ValueError(
            msg
        )

    return c.astype(np.float32, copy=False)


def _field_key_to_int(key: object) -> int:
    """Convert segyio field keys/enums to plain int for broad segyio compatibility."""
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
    """Return (win256, start_orig_i)
    start_orig_i is the original sample index corresponding to win256[0].
    """
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


def _valid_pick_mask(pick_i: np.ndarray, *, n_samples: int) -> np.ndarray:
    p = np.asarray(pick_i)
    return np.isfinite(p) & (p > 0) & (p < int(n_samples))


def _percentile_threshold(x: np.ndarray, *, frac: float) -> float:
    v = np.asarray(x, dtype=np.float32)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float('nan')
    q = float(frac) * 100.0
    return float(np.nanpercentile(v, q))


def build_keep_mask_per_segy(
    *,
    pick_final_i: np.ndarray,  # (n_traces,) int
    trend_center_i: np.ndarray,  # (n_traces,) float
    n_samples_in: int,
    scores: dict[str, np.ndarray],  # each (n_traces,)
) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    """keep_mask: True means "usable as teacher candidate"
    thresholds: per score percentile threshold (p10)
    reason_mask: uint8 bitmask per trace for debugging
      bit0: invalid pick
      bit1: trend missing
      bit2: pick outside ±128 window
      bit3: conf_prob low
      bit4: conf_trend low
      bit5: conf_rs low.
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

    base_valid = win_ok.copy()

    thresholds: dict[str, float] = {}
    keep = base_valid.copy()

    bit_map = {'conf_prob1': 3, 'conf_trend1': 4, 'conf_rs1': 5}

    for k in SCORE_KEYS:
        s = np.asarray(scores[k], dtype=np.float32)
        if s.shape != (n_tr,):
            msg = f'score {k} must be (n_traces,), got {s.shape}, n_traces={n_tr}'
            raise ValueError(msg)

        th = _percentile_threshold(s[base_valid], frac=DROP_LOW_FRAC)
        thresholds[k] = th

        keep_k = np.zeros(n_tr, dtype=bool) if np.isnan(th) else s >= th

        low = base_valid & (~keep_k)
        reason[low] |= 1 << bit_map[k]
        keep &= keep_k

    return keep, thresholds, reason


def infer_npz_path_for_segy(segy_path: Path) -> Path:
    rel = segy_path.relative_to(IN_SEGY_ROOT)
    return IN_INFER_ROOT / rel.parent / f'{segy_path.stem}.prob.npz'


def out_segy_path_for_in(segy_path: Path) -> Path:
    rel = segy_path.relative_to(IN_SEGY_ROOT)
    out_rel = rel.with_suffix('')  # drop ext
    return OUT_SEGY_ROOT / out_rel.parent / f'{out_rel.name}.win512.sgy'


def out_sidecar_npz_path_for_out(out_segy_path: Path) -> Path:
    return out_segy_path.with_suffix('.sidecar.npz')


# =========================
# Main per-file processing
# =========================
def process_one_segy(segy_path: Path) -> None:
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

        trend_center_i = _load_trend_center_i(
            z, n_traces=n_traces, dt_sec_from_segy=dt_sec_in
        )

        scores: dict[str, np.ndarray] = {}
        for k in SCORE_KEYS:
            scores[k] = _npz_must_have(z, k).astype(np.float32, copy=False)

        keep_mask, thresholds, reason_mask = build_keep_mask_per_segy(
            pick_final_i=pick_final,
            trend_center_i=trend_center_i,
            n_samples_in=ns_in,
            scores=scores,
        )

        # pick position in new 512 coordinate (float32; for later CSR creation)
        c_round = np.rint(trend_center_i).astype(np.int64, copy=False)
        win_start_i = c_round - int(HALF_WIN)
        pick_win_512 = (
            pick_final.astype(np.float32) - win_start_i.astype(np.float32)
        ) * float(UP_FACTOR)
        pick_win_512[~keep_mask] = np.nan  # 採用外は NaN（後段で捨てやすい）

        # SEGY output spec
        spec = segyio.spec()
        spec.tracecount = n_traces
        spec.samples = np.arange(OUT_NS, dtype=np.int32)
        spec.format = 5  # IEEE float32
        sorting_val = getattr(src, 'sorting', 1)
        try:
            spec.sorting = int(1 if sorting_val is None else sorting_val)
        except (TypeError, ValueError):
            spec.sorting = 1

        with segyio.create(str(out_segy), spec) as dst:
            dst.text[0] = src.text[0]

            for k in src.bin:
                dst.bin[_field_key_to_int(k)] = src.bin[k]
            dst.bin[_field_key_to_int(segyio.BinField.Interval)] = dt_us_out
            dst.bin[_field_key_to_int(segyio.BinField.Samples)] = OUT_NS

            # per trace
            for i in range(n_traces):
                h = {
                    _field_key_to_int(k): v for k, v in dict(src.header[i]).items()
                }
                h[_field_key_to_int(segyio.TraceField.TRACE_SAMPLE_INTERVAL)] = dt_us_out
                h[_field_key_to_int(segyio.TraceField.TRACE_SAMPLE_COUNT)] = OUT_NS
                dst.header[i] = h

                tr = np.asarray(src.trace[i], dtype=np.float32)
                w256, _start = _extract_256(tr, center_i=float(trend_center_i[i]))
                w512 = _upsample_256_to_512_linear(w256)
                dst.trace[i] = w512

            dst.flush()

    # sidecar npz（教師選定＆CSR生成に必要な情報をまとめて保存）
    np.savez_compressed(
        side_npz,
        src_segy=str(segy_path),
        src_infer_npz=str(infer_npz),
        out_segy=str(out_segy),
        dt_sec_in=np.float32(dt_sec_in),
        dt_sec_out=np.float32(dt_sec_out),
        dt_us_in=np.int32(dt_us_in),
        dt_us_out=np.int32(dt_us_out),
        n_traces=np.int32(n_traces),
        n_samples_in=np.int32(ns_in),
        n_samples_out=np.int32(OUT_NS),
        trend_center_i=trend_center_i.astype(np.float32, copy=False),
        trend_center_i_round=c_round.astype(np.int64, copy=False),
        window_start_i=win_start_i.astype(np.int64, copy=False),
        pick_final_i=pick_final.astype(np.int64, copy=False),
        pick_win_512=pick_win_512.astype(np.float32, copy=False),
        keep_mask=keep_mask.astype(bool, copy=False),
        reason_mask=reason_mask.astype(np.uint8, copy=False),
        th_conf_prob1=np.float32(thresholds['conf_prob1']),
        th_conf_trend1=np.float32(thresholds['conf_trend1']),
        th_conf_rs1=np.float32(thresholds['conf_rs1']),
        conf_prob1=scores['conf_prob1'].astype(np.float32, copy=False),
        conf_trend1=scores['conf_trend1'].astype(np.float32, copy=False),
        conf_rs1=scores['conf_rs1'].astype(np.float32, copy=False),
    )

    n_keep = int(np.count_nonzero(keep_mask))
    print(
        f'[OK] {segy_path.name} -> {out_segy.name}  keep={n_keep}/{n_traces} '
        f'th(p10) prob={thresholds["conf_prob1"]:.6g} trend={thresholds["conf_trend1"]:.6g} rs={thresholds["conf_rs1"]:.6g}'
    )


def main() -> None:
    segys = find_segy_files(IN_SEGY_ROOT)
    print(f'[RUN] found {len(segys)} segy files under {IN_SEGY_ROOT}')

    for p in segys:
        # infer済みだけ回したいなら、npz存在チェックでスキップも可
        infer_npz = infer_npz_path_for_segy(p)
        if not infer_npz.exists():
            print(f'[SKIP] infer npz missing: {p}  expected={infer_npz}')
            continue
        process_one_segy(p)


if __name__ == '__main__':
    main()
