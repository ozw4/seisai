# %%
#!/usr/bin/env python3
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import segyio
import torch
from common.paths import stage1_out_dir as _stage1_out_dir
from jogsarar_shared import (
    TilePerTraceStandardize,
    find_segy_files,
)
from jogsarar_viz.noop import save_conf_scatter_noop, save_stage1_gather_viz_noop
from stage1.cfg import (
    DEFAULT_STAGE1_CFG,
    INPUT_DIR,
    OUT_DIR,
    VIZ_DIRNAME,
    VIZ_EVERY_N_SHOTS,
    WEIGHTS_PATH,
    Stage1Cfg,
    _validate_stage1_cfg,
    _write_stage1_used_cfg_yaml,
)
from stage1.model import build_model_from_cfg
from stage1.process_one import process_one_segy as _process_one_segy
from seisai_dataset.config import LoaderConfig
from seisai_dataset.file_info import build_file_info_dataclass
from seisai_dataset.trace_subset_preproc import TraceSubsetLoader
from seisai_engine.postprocess.velocity_filter_op import apply_velocity_filt_prob
from seisai_engine.predict import _run_tiled
from seisai_pick.pickio.io_grstat import numpy2fbcrd
from seisai_pick.snap_picks_to_phase import snap_picks_to_phase
from seisai_pick.trend.trend_fit import robust_linear_trend

BuildFileInfoFn = Callable[..., Any]
SnapPicksFn = Callable[..., Any]
Numpy2FbCrdFn = Callable[..., Any]


def _scale01_by_percentile(
    y: np.ndarray,
    *,
    pct_lo: float,
    pct_hi: float,
    eps: float,
) -> tuple[np.ndarray, tuple[float, float]]:
    yy = np.asarray(y, dtype=np.float32)
    finite = np.isfinite(yy)
    if not np.any(finite):
        return np.zeros_like(yy, dtype=np.float32), (float('nan'), float('nan'))

    plo = float(np.percentile(yy[finite], float(pct_lo)))
    phi = float(np.percentile(yy[finite], float(pct_hi)))
    denom = float(phi - plo)
    denom = max(float(eps), denom)

    out = (yy - plo) / denom
    out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)
    out[~finite] = 0.0
    return out, (plo, phi)


def _cos_ramp01(s: torch.Tensor) -> torch.Tensor:
    s = s.clamp(0.0, 1.0)
    return 0.5 - 0.5 * torch.cos(torch.pi * s)


def make_velocity_feasible_filt_allow_vmin0(
    *,
    offsets_m: torch.Tensor,  # (B,H)
    dt_sec: float,
    W: int,
    vmin: float,
    vmax: float,
    t0_lo_ms: float = 0.0,
    t0_hi_ms: float = 0.0,
    taper_ms: float = 0.0,
) -> torch.Tensor:
    if vmax <= 0.0:
        msg = f'vmax must be positive, got {vmax}'
        raise ValueError(msg)
    if vmin < 0.0:
        msg = f'vmin must be >=0, got {vmin}'
        raise ValueError(msg)
    if W <= 0:
        msg = f'W must be positive, got {W}'
        raise ValueError(msg)
    if vmin > 0.0 and vmax < vmin:
        msg = f'vmax must be >= vmin. got vmin={vmin}, vmax={vmax}'
        raise ValueError(msg)
    if offsets_m.ndim != 2:
        msg = f'offsets_m must be (B,H), got {tuple(offsets_m.shape)}'
        raise ValueError(msg)

    dt = float(dt_sec)
    if dt <= 0.0:
        msg = f'dt_sec must be positive, got {dt_sec}'
        raise ValueError(msg)

    B, H = offsets_m.shape
    dev = offsets_m.device

    t = torch.arange(W, device=dev, dtype=torch.float32).view(1, 1, W) * dt
    x = offsets_m.to(device=dev, dtype=torch.float32).abs().view(B, H, 1)

    t_lo = x / float(vmax) + (float(t0_lo_ms) / 1000.0)

    if vmin == 0.0:
        t_hi = torch.full_like(t_lo, float('inf')) + (float(t0_hi_ms) / 1000.0)
    else:
        t_hi = x / float(vmin) + (float(t0_hi_ms) / 1000.0)

    m = torch.zeros((B, H, W), device=dev, dtype=torch.float32)
    inside = (t >= t_lo) & (t <= t_hi)
    m[inside] = 1.0

    if taper_ms > 0.0:
        w = float(taper_ms) / 1000.0
        s_lo = (t - (t_lo - w)) / w
        lower = _cos_ramp01(s_lo) * (t <= t_lo)

        if vmin == 0.0:
            upper = torch.zeros_like(lower)
        else:
            s_hi = (t_hi + w - t) / w
            upper = _cos_ramp01(s_hi) * (t >= t_hi)

        m = torch.maximum(m, lower)
        m = torch.maximum(m, upper)

    return m  # (B,H,W)


def pad_samples_to_6016(
    x_hw: np.ndarray, w_target: int = 6016
) -> tuple[np.ndarray, int]:
    if x_hw.ndim != 2:
        msg = f'x_hw must be (H,W), got {x_hw.shape}'
        raise ValueError(msg)
    _, w0 = x_hw.shape
    if w0 > w_target:
        msg = f'n_samples={w0} exceeds w_target={w_target}'
        raise ValueError(msg)
    if w0 == w_target:
        return x_hw.astype(np.float32, copy=False), w0
    out = np.zeros((x_hw.shape[0], w_target), dtype=np.float32)
    out[:, :w0] = x_hw.astype(np.float32, copy=False)
    return out, w0


@torch.no_grad()
def infer_gather_prob(
    *,
    model: torch.nn.Module,
    wave_hw: np.ndarray,
    offsets_m: np.ndarray,
    dt_sec: float,
    cfg: Stage1Cfg = DEFAULT_STAGE1_CFG,
) -> tuple[np.ndarray, int]:
    TILE_W = int(cfg.tile_w)
    TILE_H = int(cfg.tile_h)
    OVERLAP_H = int(cfg.overlap_h)
    TILES_PER_BATCH = int(cfg.tiles_per_batch)
    POLARITY_FLIP = bool(cfg.polarity_flip)
    VMIN_MASK = float(cfg.vmin_mask)
    VMAX_MASK = float(cfg.vmax_mask)
    T0_LO_MS = float(cfg.t0_lo_ms)
    T0_HI_MS = float(cfg.t0_hi_ms)
    TAPER_MS = float(cfg.taper_ms)

    device = next(model.parameters()).device

    if wave_hw.ndim != 2:
        msg = f'wave_hw must be (H,W0), got {wave_hw.shape}'
        raise ValueError(msg)
    H0, W0 = wave_hw.shape
    if offsets_m.shape != (H0,):
        msg = f'offsets_m must be (H,), got {offsets_m.shape}, H={H0}'
        raise ValueError(msg)

    n_samples_orig = int(W0)

    wave_in = wave_hw.astype(np.float32, copy=False)
    dt_in = float(dt_sec)

    wave_pad, n_samples_in = pad_samples_to_6016(wave_in, w_target=TILE_W)
    if POLARITY_FLIP:
        wave_pad = -wave_pad
    H, W = wave_pad.shape
    if H != H0 or W != TILE_W:
        msg = f'unexpected padded shape: {wave_pad.shape}'
        raise ValueError(msg)

    tile_h = min(TILE_H, H)
    ov_h = OVERLAP_H if H >= TILE_H else 0
    tile = (tile_h, TILE_W)
    overlap = (ov_h, 0)

    x = (
        torch.from_numpy(wave_pad)
        .to(device=device, dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
    )

    logits = _run_tiled(
        model,
        x,
        tile=tile,
        overlap=overlap,
        amp=True,
        use_tqdm=False,
        tiles_per_batch=TILES_PER_BATCH,
        tile_transform=TilePerTraceStandardize(eps_std=1e-10),
        post_tile_transform=None,
    )

    if logits.shape[1] != 1:
        msg = f'expected out_chans=1, got {logits.shape[1]}'
        raise ValueError(msg)

    prob = torch.softmax(logits, dim=-1)

    offs_t = (
        torch.from_numpy(offsets_m).to(device=device, dtype=torch.float32).view(1, H)
    )
    mask = make_velocity_feasible_filt_allow_vmin0(
        offsets_m=offs_t,
        dt_sec=dt_in,
        W=W,
        vmin=float(VMIN_MASK),
        vmax=float(VMAX_MASK),
        t0_lo_ms=float(T0_LO_MS),
        t0_hi_ms=float(T0_HI_MS),
        taper_ms=float(TAPER_MS),
    )

    prob = apply_velocity_filt_prob(prob, mask, renorm=False, time_dim=-1)

    if n_samples_in < W:
        prob[:, :, :, n_samples_in:] = 0.0

    prob_np = prob[0, 0].detach().to('cpu', dtype=torch.float32).numpy()

    prob_out = np.zeros((H, TILE_W), dtype=np.float32)
    prob_out[:, :n_samples_orig] = prob_np[:, :n_samples_orig]
    return prob_out, n_samples_orig


def run_stage1(
    *,
    in_segy_root: Path = INPUT_DIR,
    out_infer_root: Path = OUT_DIR,
    weights_path: Path = WEIGHTS_PATH,
    segy_paths: list[Path] | None = None,
    segy_exts: tuple[str, ...] = ('.sgy', '.segy'),
    recursive: bool = False,
    viz_every_n_shots: int = VIZ_EVERY_N_SHOTS,
    viz_dirname: str = VIZ_DIRNAME,
) -> None:
    cfg = replace(
        DEFAULT_STAGE1_CFG,
        in_segy_root=Path(in_segy_root),
        out_infer_root=Path(out_infer_root),
        weights_path=Path(weights_path),
        segy_exts=tuple(str(x) for x in segy_exts),
        recursive=bool(recursive),
        viz_every_n_shots=int(viz_every_n_shots),
        viz_dirname=str(viz_dirname),
    )
    run_stage1_cfg(cfg, segy_paths=segy_paths)


def run_stage1_cfg(
    cfg: Stage1Cfg,
    *,
    segy_paths: list[Path] | None = None,
) -> None:
    cfg = _validate_stage1_cfg(cfg)

    in_root = Path(cfg.in_segy_root).expanduser().resolve()
    if not in_root.is_dir():
        msg = f'in_segy_root must be an existing directory: {in_root}'
        raise FileNotFoundError(msg)

    out_root = Path(cfg.out_infer_root).expanduser().resolve()
    segy_exts = tuple(str(x) for x in cfg.segy_exts)

    if segy_paths is None:
        segys = find_segy_files(in_root, exts=segy_exts, recursive=bool(cfg.recursive))
    else:
        segys = [Path(p).expanduser().resolve() for p in segy_paths]

    if len(segys) == 0:
        msg = f'no segy files found under: {in_root}'
        raise RuntimeError(msg)

    used_cfg_path = _write_stage1_used_cfg_yaml(cfg, out_root=out_root)
    print(f'[CFG][STAGE1] saved {used_cfg_path}')

    model = build_model_from_cfg(cfg)

    print(f'[RUN][STAGE1] files={len(segys)} in_root={in_root} out_root={out_root}')
    n_ok = 0
    for segy_path in segys:
        if not segy_path.is_file():
            msg = f'segy file not found: {segy_path}'
            raise FileNotFoundError(msg)
        per_file_out_dir = _stage1_out_dir(
            segy_path,
            in_segy_root=in_root,
            out_infer_root=out_root,
        )
        process_one_segy(
            segy_path=segy_path,
            out_dir=per_file_out_dir,
            model=model,
            cfg=cfg,
            viz_every_n_shots=cfg.viz_every_n_shots,
            viz_dirname=cfg.viz_dirname,
        )
        n_ok += 1

    print(f'[DONE][STAGE1] processed={n_ok} out_root={out_root}')


def _fit_local_trend_sec(
    *,
    offsets_m: np.ndarray,  # (H,)
    t_pick_sec: np.ndarray,  # (H,)
    valid_mask: np.ndarray,  # (H,) bool
    pmax: np.ndarray,  # (H,) float32
    cfg: Stage1Cfg = DEFAULT_STAGE1_CFG,
) -> tuple[np.ndarray, np.ndarray]:
    TREND_LOCAL_USE_ABS_OFFSET = bool(cfg.trend_local_use_abs_offset)
    TREND_LOCAL_WEIGHT_MODE = str(cfg.trend_local_weight_mode)
    TREND_LOCAL_SECTION_LEN = int(cfg.trend_local_section_len)
    TREND_LOCAL_STRIDE = int(cfg.trend_local_stride)
    TREND_LOCAL_HUBER_C = float(cfg.trend_local_huber_c)
    TREND_LOCAL_ITERS = int(cfg.trend_local_iters)
    TREND_LOCAL_VMIN_MPS = cfg.trend_local_vmin_mps
    TREND_LOCAL_VMAX_MPS = cfg.trend_local_vmax_mps
    TREND_LOCAL_SORT_OFFSETS = bool(cfg.trend_local_sort_offsets)

    x0 = np.asarray(offsets_m, dtype=np.float32)
    x = np.abs(x0) if TREND_LOCAL_USE_ABS_OFFSET else x0
    y = np.asarray(t_pick_sec, dtype=np.float32)

    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        msg = f'offsets/t_pick_sec must be (H,), got x={x.shape}, y={y.shape}'
        raise ValueError(msg)

    v = np.asarray(valid_mask, dtype=bool)
    if v.shape != x.shape:
        msg = f'valid_mask must be (H,), got {v.shape}, expected {x.shape}'
        raise ValueError(msg)

    if TREND_LOCAL_WEIGHT_MODE == 'pmax':
        w = np.asarray(pmax, dtype=np.float32)
        if w.shape != x.shape:
            msg = f'pmax must be (H,), got {w.shape}, expected {x.shape}'
            raise ValueError(msg)
        w = w * v.astype(np.float32)
    elif TREND_LOCAL_WEIGHT_MODE == 'uniform':
        w = v.astype(np.float32)
    else:
        msg = f'unknown TREND_LOCAL_WEIGHT_MODE={TREND_LOCAL_WEIGHT_MODE!r}'
        raise ValueError(msg)

    trend_t_bh, _, _, _, covered_bh = robust_linear_trend(
        x,
        y,
        v.astype(np.uint8),
        w_conf=w,
        section_len=int(TREND_LOCAL_SECTION_LEN),
        stride=int(TREND_LOCAL_STRIDE),
        huber_c=float(TREND_LOCAL_HUBER_C),
        iters=int(TREND_LOCAL_ITERS),
        vmin=float(TREND_LOCAL_VMIN_MPS) if TREND_LOCAL_VMIN_MPS is not None else None,
        vmax=float(TREND_LOCAL_VMAX_MPS) if TREND_LOCAL_VMAX_MPS is not None else None,
        sort_offsets=bool(TREND_LOCAL_SORT_OFFSETS),
        use_taper=True,
        abs_velocity=False,
    )

    tt = np.asarray(trend_t_bh, dtype=np.float32)
    cc = np.asarray(covered_bh, dtype=bool)

    if tt.ndim != 2 or cc.ndim != 2 or tt.shape != cc.shape or tt.shape[0] != 1:
        msg = f'unexpected trend shape: trend_t={tt.shape}, covered={cc.shape}'
        raise ValueError(msg)

    t_trend_sec = tt[0].copy()
    covered = cc[0].copy()

    t_trend_sec[~covered] = np.nan
    return t_trend_sec, covered


def _fit_local_trend_split_sides_sec(
    *,
    offsets_abs_m: np.ndarray,  # (H,) ここは |offset| でOK（ヘッダーが絶対値でもOK）
    t_pick_sec: np.ndarray,  # (H,)
    valid_mask: np.ndarray,  # (H,) bool
    pmax: np.ndarray,  # (H,) float32
    invalid: np.ndarray,  # (H,) bool
    cfg: Stage1Cfg = DEFAULT_STAGE1_CFG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    TREND_MIN_PTS = int(cfg.trend_min_pts)

    x_abs = np.asarray(offsets_abs_m, dtype=np.float32)
    y = np.asarray(t_pick_sec, dtype=np.float32)
    v = np.asarray(valid_mask, dtype=bool)
    inv = np.asarray(invalid, dtype=bool)
    pm = np.asarray(pmax, dtype=np.float32)

    if x_abs.ndim != 1 or y.ndim != 1 or v.ndim != 1 or inv.ndim != 1 or pm.ndim != 1:
        msg = (
            'offsets_abs_m/t_pick_sec/valid_mask/invalid/pmax must be 1D, '
            f'got {x_abs.shape},{y.shape},{v.shape},{inv.shape},{pm.shape}'
        )
        raise ValueError(msg)
    if not (x_abs.shape == y.shape == v.shape == inv.shape == pm.shape):
        msg = (
            'shape mismatch: '
            f'x={x_abs.shape}, y={y.shape}, v={v.shape}, inv={inv.shape}, pm={pm.shape}'
        )
        raise ValueError(msg)

    H = x_abs.shape[0]
    cand = x_abs.copy()
    cand[inv | (~v)] = np.inf

    if not np.isfinite(cand).any():
        return (
            np.full(H, np.nan, np.float32),
            np.zeros(H, bool),
            np.full(H, np.nan, np.float32),
            -1,
        )

    split = int(np.argmin(cand))
    if not np.isfinite(cand[split]):
        return (
            np.full(H, np.nan, np.float32),
            np.zeros(H, bool),
            np.full(H, np.nan, np.float32),
            -1,
        )

    # 擬似符号付きoffset（保存用）：splitより前を負、split以降を正
    offset_signed_proxy = x_abs.astype(np.float32, copy=True)
    offset_signed_proxy[:split] *= -1.0

    t_trend = np.full(H, np.nan, dtype=np.float32)
    covered = np.zeros(H, dtype=bool)

    # split を両側に含めて重ね、予測は平均でつなぐ（段差抑制）
    sides = (
        (slice(0, split + 1), -1.0),
        (slice(split, H), +1.0),
    )

    for sl, sign in sides:
        vv = v[sl]
        if int(np.count_nonzero(vv)) < int(TREND_MIN_PTS):
            continue

        x_side = x_abs[sl] * float(sign)
        y_side = y[sl]
        p_side = pm[sl]

        t_side, cov_side = _fit_local_trend_sec(
            offsets_m=x_side,
            t_pick_sec=y_side,
            valid_mask=vv,
            pmax=p_side,
            cfg=cfg,
        )

        cur = t_trend[sl]
        both = np.isfinite(cur) & np.isfinite(t_side)
        only_new = (~np.isfinite(cur)) & np.isfinite(t_side)

        out = cur.copy()
        out[both] = 0.5 * (cur[both] + t_side[both])
        out[only_new] = t_side[only_new]

        t_trend[sl] = out
        covered[sl] = covered[sl] | cov_side

    t_trend[inv] = np.nan
    covered[inv] = False
    offset_signed_proxy[inv] = np.nan
    return t_trend, covered, offset_signed_proxy, split


def process_one_segy(
    *,
    segy_path: Path,
    out_dir: Path,
    model: torch.nn.Module,
    cfg: Stage1Cfg = DEFAULT_STAGE1_CFG,
    build_file_info_dataclass_fn: BuildFileInfoFn = build_file_info_dataclass,
    TraceSubsetLoaderCls: type[TraceSubsetLoader] = TraceSubsetLoader,
    LoaderConfigCls: type[LoaderConfig] = LoaderConfig,
    snap_picks_to_phase_fn: SnapPicksFn = snap_picks_to_phase,
    numpy2fbcrd_fn: Numpy2FbCrdFn = numpy2fbcrd,
    viz_every_n_shots: int = 0,
    viz_dirname: str = 'viz',
) -> None:
    save_conf_scatter_fn: Callable[..., None] = save_conf_scatter_noop
    if bool(cfg.conf_viz_enable):
        from jogsarar_viz.stage1_conf import save_conf_scatter

        save_conf_scatter_fn = save_conf_scatter

    save_gather_viz_fn: Callable[..., None] = save_stage1_gather_viz_noop
    if int(viz_every_n_shots) > 0:
        from jogsarar_viz.stage1_gather import save_stage1_gather_viz

        save_gather_viz_fn = save_stage1_gather_viz

    _process_one_segy(
        segy_path=segy_path,
        out_dir=out_dir,
        model=model,
        cfg=cfg,
        build_file_info_dataclass_fn=build_file_info_dataclass_fn,
        TraceSubsetLoaderCls=TraceSubsetLoaderCls,
        LoaderConfigCls=LoaderConfigCls,
        snap_picks_to_phase_fn=snap_picks_to_phase_fn,
        numpy2fbcrd_fn=numpy2fbcrd_fn,
        viz_every_n_shots=viz_every_n_shots,
        viz_dirname=viz_dirname,
        infer_gather_prob_fn=infer_gather_prob,
        fit_local_trend_split_sides_sec_fn=_fit_local_trend_split_sides_sec,
        fit_local_trend_sec_fn=_fit_local_trend_sec,
        save_conf_scatter_fn=save_conf_scatter_fn,
        scale01_by_percentile_fn=_scale01_by_percentile,
        save_gather_viz_fn=save_gather_viz_fn,
    )


__all__ = [
    'infer_gather_prob',
    'make_velocity_feasible_filt_allow_vmin0',
    'pad_samples_to_6016',
    'process_one_segy',
    'run_stage1',
    'run_stage1_cfg',
]
