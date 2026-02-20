# %%
#!/usr/bin/env python3
from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import segyio
import torch
from _model import NetAE as EncDec2D
from jogsarar_shared import (
    TilePerTraceStandardize,
    build_pick_aligned_window,
    find_segy_files,
    valid_pick_mask,
)
from seisai_dataset.config import LoaderConfig
from seisai_dataset.file_info import build_file_info_dataclass
from seisai_dataset.trace_subset_preproc import TraceSubsetLoader
from seisai_engine.postprocess.velocity_filter_op import apply_velocity_filt_prob
from seisai_engine.predict import _run_tiled
from seisai_pick.lmo import apply_lmo_linear, lmo_correct_picks
from seisai_pick.pickio.io_grstat import numpy2fbcrd
from seisai_pick.residual_statics import refine_firstbreak_residual_statics
from seisai_pick.score.confidence_from_prob import (
    trace_confidence_from_prob_local_window,
)
from seisai_pick.score.confidence_from_residual_statics import (
    trace_confidence_from_residual_statics,
)
from seisai_pick.score.confidence_from_trend_resid import (
    trace_confidence_from_trend_resid_gaussian,
    trace_confidence_from_trend_resid_var,
)
from seisai_pick.snap_picks_to_phase import snap_picks_to_phase
from seisai_pick.trend.trend_fit import robust_linear_trend
from seisai_utils.viz_wiggle import PickOverlay, WiggleConfig, plot_wiggle

BuildFileInfoFn = Callable[..., Any]
SnapPicksFn = Callable[..., Any]
Numpy2FbCrdFn = Callable[..., Any]

# =========================
# CONFIG (ここだけ直書き)
# =========================
INPUT_DIR = Path('/home/dcuser/data/ActiveSeisField/jogsarar')
OUT_DIR = Path('/home/dcuser/data/ActiveSeisField/jogsarar_out')
WEIGHTS_PATH = Path('/home/dcuser/data/model_weight/fbseg_caformer_b36.pth')

BACKBONE = 'caformer_b36.sail_in22k_ft_in1k'
DEVICE = 'cuda'  # "cpu" も可
USE_TTA = True

PMAX_TH = 0.0
LTCOR = 5  # legacy snap (comparison / fallback)

SEGY_ENDIAN = 'big'  # "big" or "little"
WAVEFORM_MODE = 'mmap'  # "mmap" or "eager"
HEADER_CACHE_DIR: str | None = None

# 可視化: 各SEGYファイル内で「50ショットおき」に可視化（0なら無効）
VIZ_EVERY_N_SHOTS = 100
VIZ_DIRNAME = 'viz'

# velocity mask params (inference)
VMIN_MASK = 100.0
VMAX_MASK = 5000.0
T0_LO_MS = -10.0
T0_HI_MS = 100.0
TAPER_MS = 10.0

# tile params
TILE_H = 128
TILE_W = 6016
OVERLAP_H = 96  # stride_h = 32
TILES_PER_BATCH = 8

POLARITY_FLIP = True  # model input only

# ---- LMO for visualization (display only) ----
LMO_VEL_MPS = 3200.0
LMO_BULK_SHIFT_SAMPLES = 50.0  # positive shifts later samples

PLOT_START = 0
PLOT_END = 350

VIZ_SCORE_COMPONENTS = True
VIZ_SCORE_STYLE = 'bar'  # 'bar' or 'line'

# ---- prob_conf 可視化スケーリング（保存値は生のまま） ----
VIZ_CONF_PROB_SCALE_ENABLE = True
VIZ_CONF_PROB_PCT_LO = 5.0
VIZ_CONF_PROB_PCT_HI = 99.0
VIZ_CONF_PROB_PCT_EPS = 1e-12

# y軸上限（None なら自動）
# conf_prob は percentile で 0..1 にして描く前提なので、基本は 1.0 推奨
VIZ_YMAX_CONF_PROB = 1.0
VIZ_YMAX_CONF_TREND = 1.0
VIZ_YMAX_CONF_RS = 1.0

# ---- trend line overlay on wiggle ----
VIZ_TREND_LINE_ENABLE = True
VIZ_TREND_LINE_LW = 1.6
VIZ_TREND_LINE_ALPHA = 0.9
VIZ_TREND_LINE_LABEL = 'trend'
VIZ_TREND_LINE_COLOR = 'g'

# --- residual statics refine ---
USE_RESIDUAL_STATICS = True

# RS_BASE_PICK:
#   'pre'  : snap前（pick0=argmax+threshold）を基準に窓を作ってrefine
#   'snap' : 一度phase snapしてから（pick_pre_snap）それを基準に窓を作ってrefine
RS_BASE_PICK = 'snap'  # 'pre' or 'snap'

# RS_BASE_PICK='snap' のときに使う “事前snap” 設定（refine前のsnap）
RS_PRE_SNAP_MODE = 'trough'
RS_PRE_SNAP_LTCOR = 3

# 窓切り出し（RS_BASE_PICKで選んだpickを基準に窓を作る）
RS_PRE_SAMPLES = 20
RS_POST_SAMPLES = 20

# residual statics params
RS_MAX_LAG = 8
RS_K_NEIGHBORS = 5
RS_N_ITER = 2
RS_MODE = 'diff'  # 'diff' or 'raw'
RS_C_TH = 0.5
RS_SMOOTH_METHOD = 'wls'
RS_LAM = 5.0
RS_SUBSAMPLE = True
RS_PROPAGATE_LOW_CORR = False
RS_TAPER = 'hann'
RS_TAPER_POWER = 1.0
RS_LAG_PENALTY = 0.10
RS_LAG_PENALTY_POWER = 1.0

# --- optional final snap (after refine) ---
USE_FINAL_SNAP = True
FINAL_SNAP_MODE = 'trough'
FINAL_SNAP_LTCOR = 3

# --- confidence scoring ---
CONF_ENABLE = True
CONF_VIZ_ENABLE = True
CONF_VIZ_FFID = 2147
CONF_HALF_WIN = 20

# ---- local trend (方針A: windowed IRLS) ----
TREND_LOCAL_ENABLE = True

# 重要: 左右分け（擬似符号付きoffset）を使うため abs は使わない
TREND_LOCAL_USE_ABS_OFFSET = False

# 重要: window は trace順で切りたいので、offsetで並べ替えない
TREND_LOCAL_SORT_OFFSETS = False

# 重要: 左右分割を有効化（ヘッダーが |offset| のみでもOK）
TREND_SIDE_SPLIT_ENABLE = True

TREND_LOCAL_USE_ABS_OFFSET_HEADER = True  # ヘッダーが絶対値なら True（通常 True）
TREND_LOCAL_SECTION_LEN = 16  # 小さいほどローカル
TREND_LOCAL_STRIDE = 4  # section_len/2 くらいが無難
TREND_LOCAL_HUBER_C = 1.345
TREND_LOCAL_ITERS = 3
TREND_LOCAL_VMIN_MPS = 300.0
TREND_LOCAL_VMAX_MPS = 8000.0
TREND_LOCAL_WEIGHT_MODE = 'uniform'  # 'uniform' or 'pmax'

# trend residual scoring
TREND_SIGMA_MS = 6.0
TREND_MIN_PTS = 12

# trend residual local-variance scoring (trace-direction)
TREND_VAR_HALF_WIN_TRACES = 8
TREND_VAR_SIGMA_STD_MS = 6.0
TREND_VAR_MIN_COUNT = 3

# RS confidence
RS_CMAX_TH = RS_C_TH
RS_ABS_LAG_SOFT = float(RS_MAX_LAG)

# ---- trend 保存（npzに残す） ----
SAVE_TREND_TO_NPZ = True
TREND_SOURCE_LABEL = 'pick_final'
TREND_METHOD_LABEL = 'local_irls_split_sides'


def _plot_score_panel_1d(
    *,
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    ymax: float | None,
    style: str,
) -> None:
    xx = np.asarray(x, dtype=np.float32)
    yy = np.asarray(y, dtype=np.float32)

    if xx.ndim != 1 or yy.ndim != 1 or xx.shape != yy.shape:
        msg = f'x/y must be (N,), got x={xx.shape}, y={yy.shape}'
        raise ValueError(msg)

    st = str(style).lower()
    if st not in ('bar', 'line'):
        msg = f"style must be 'bar' or 'line', got {style!r}"
        raise ValueError(msg)

    if st == 'bar':
        ax.bar(xx, yy, width=1.0, alpha=0.8)
    else:
        ax.plot(xx, yy, lw=1.2)

    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.2)

    if ymax is not None:
        ax.set_ylim(0.0, float(ymax))
    else:
        ax.set_ylim(bottom=0.0)


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
) -> tuple[np.ndarray, int]:
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


def _strip_prefix(sd: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith('model.'):
            out[k.removeprefix('model.')] = v
        elif k.startswith('module.'):
            out[k.removeprefix('module.')] = v
        else:
            out[k] = v
    return out


def build_model() -> torch.nn.Module:
    if DEVICE == 'cuda' and not torch.cuda.is_available():
        msg = 'CUDA requested but not available'
        raise RuntimeError(msg)

    model = EncDec2D(
        backbone=BACKBONE,
        in_chans=1,
        out_chans=1,
        pretrained=True,
        stage_strides=[(2, 4), (2, 2), (2, 4), (2, 2)],
        pre_stages=2,
        pre_stage_strides=((1, 1), (1, 2)),
    )
    model.out_chans = 1
    model.use_tta = bool(USE_TTA)

    ckpt = torch.load(WEIGHTS_PATH, map_location='cpu', weights_only=False)
    sd = ckpt['model_ema']

    if (
        not isinstance(sd, Mapping)
        or not sd
        or not all(isinstance(v, torch.Tensor) for v in sd.values())
    ):
        msg = "ckpt['model_ema'] is not a state_dict"
        raise ValueError(msg)

    sd = _strip_prefix(sd)
    print('[CKPT] load from: model_ema (direct state_dict)')

    model.load_state_dict(sd, strict=True)
    model.to(device=torch.device(DEVICE))
    model.eval()
    return model


def _save_conf_scatter(
    *,
    out_png: Path,
    x_abs: np.ndarray,
    picks_i: np.ndarray,
    dt_ms: float,
    conf_prob_viz01: np.ndarray,
    conf_rs: np.ndarray,
    conf_trend: np.ndarray,
    title: str,
    trend_hat_ms: np.ndarray | None = None,
) -> None:
    x = np.asarray(x_abs, dtype=np.float32)
    pk = np.asarray(picks_i)
    if x.ndim != 1 or pk.ndim != 1 or x.shape[0] != pk.shape[0]:
        msg = f'x_abs/picks_i must be (H,), got {x.shape}, {pk.shape}'
        raise ValueError(msg)

    y_ms = pk.astype(np.float32, copy=False) * float(dt_ms)
    valid = valid_pick_mask(pk)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    panels = [
        ('conf_prob(viz01)', np.asarray(conf_prob_viz01, dtype=np.float32)),
        ('conf_trend', np.asarray(conf_trend, dtype=np.float32)),
        ('conf_rs', np.asarray(conf_rs, dtype=np.float32)),
    ]

    for ax, (name, cval) in zip(axes.ravel(), panels, strict=True):
        if cval.shape != x.shape:
            msg = f'{name} shape mismatch: {cval.shape}, expected {x.shape}'
            raise ValueError(msg)

        if np.any(valid):
            sc = ax.scatter(
                x[valid],
                y_ms[valid],
                c=cval[valid],
                s=12.0,
                cmap='viridis',
                vmin=0.0,
                vmax=1.0,
            )
            if trend_hat_ms is not None:
                th = np.asarray(trend_hat_ms, dtype=np.float32)
                if th.shape != x.shape:
                    msg = f'trend_hat_ms shape mismatch: {th.shape}, expected {x.shape}'
                    raise ValueError(msg)
                tmask = valid & np.isfinite(th)
                if np.any(tmask):
                    ord_i = np.argsort(x[tmask], kind='mergesort')
                    ax.plot(
                        x[tmask][ord_i],
                        th[tmask][ord_i],
                        color='white',
                        lw=1.0,
                        alpha=0.8,
                    )
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title(name)
        ax.grid(alpha=0.2)

    axes[0].set_xlabel('|offset| [m]')
    axes[1].set_xlabel('|offset| [m]')
    axes[2].set_xlabel('|offset| [m]')
    axes[0].set_ylabel('pick [ms]')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _fit_local_trend_sec(
    *,
    offsets_m: np.ndarray,  # (H,)
    t_pick_sec: np.ndarray,  # (H,)
    valid_mask: np.ndarray,  # (H,) bool
    pmax: np.ndarray,  # (H,) float32
) -> tuple[np.ndarray, np.ndarray]:
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
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
    build_file_info_dataclass_fn: BuildFileInfoFn = build_file_info_dataclass,
    TraceSubsetLoaderCls: type[TraceSubsetLoader] = TraceSubsetLoader,
    LoaderConfigCls: type[LoaderConfig] = LoaderConfig,
    snap_picks_to_phase_fn: SnapPicksFn = snap_picks_to_phase,
    numpy2fbcrd_fn: Numpy2FbCrdFn = numpy2fbcrd,
    viz_every_n_shots: int = 0,
    viz_dirname: str = 'viz',
) -> None:
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

            prob_hw, n_samples_orig = infer_gather_prob(
                model=model,
                wave_hw=wave_hw,
                offsets_m=offs_m,
                dt_sec=dt_sec,
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

                X_rs = build_pick_aligned_window(
                    wave_pad[:, :n_samples_orig],
                    picks=pick_base,
                    pre=int(RS_PRE_SAMPLES),
                    post=int(RS_POST_SAMPLES),
                    fill=0.0,
                )

                res = refine_firstbreak_residual_statics(
                    X_rs,
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

                delta = np.asarray(res['delta_pick'], dtype=np.float32)
                cmax_rs = np.asarray(res['cmax'], dtype=np.float32)
                score_rs = np.asarray(res['score'], dtype=np.float32)
                valid_rs = np.asarray(res['valid_mask'], dtype=bool)

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

                hist_last = res['history'][-1] if res['history'] else {}
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
                            _fit_local_trend_split_sides_sec(
                                offsets_abs_m=np.abs(x_abs).astype(
                                    np.float32, copy=False
                                ),
                                t_pick_sec=t_pick_fit_sec,
                                valid_mask=trend_fit_mask,
                                pmax=pmax,
                                invalid=invalid,
                            )
                        )
                        t_trend_sec = t_trend_fit
                        trend_covered = cov_fit
                        trend_offset_signed_proxy = off_proxy
                        trend_split_index = int(split)
                    else:
                        t_trend_fit, cov_fit = _fit_local_trend_sec(
                            offsets_m=x_abs,
                            t_pick_sec=t_pick_fit_sec,
                            valid_mask=trend_fit_mask,
                            pmax=pmax,
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

                    conf0_g = trace_confidence_from_trend_resid_gaussian(
                        t0_sec,
                        t_trend_sec,
                        valid0,
                        sigma_ms=float(TREND_SIGMA_MS),
                    )
                    conf1_g = trace_confidence_from_trend_resid_gaussian(
                        t1_sec,
                        t_trend_sec,
                        valid1,
                        sigma_ms=float(TREND_SIGMA_MS),
                    )

                    conf0_v = trace_confidence_from_trend_resid_var(
                        t0_sec,
                        t_trend_sec,
                        valid0,
                        half_win_traces=int(TREND_VAR_HALF_WIN_TRACES),
                        sigma_std_ms=float(TREND_VAR_SIGMA_STD_MS),
                        min_count=int(TREND_VAR_MIN_COUNT),
                    )
                    conf1_v = trace_confidence_from_trend_resid_var(
                        t1_sec,
                        t_trend_sec,
                        valid1,
                        half_win_traces=int(TREND_VAR_HALF_WIN_TRACES),
                        sigma_std_ms=float(TREND_VAR_SIGMA_STD_MS),
                        min_count=int(TREND_VAR_MIN_COUNT),
                    )

                    conf_trend0 = (
                        np.asarray(conf0_g, dtype=np.float32)
                        * np.asarray(conf0_v, dtype=np.float32)
                    ).astype(np.float32, copy=False)
                    conf_trend1 = (
                        np.asarray(conf1_g, dtype=np.float32)
                        * np.asarray(conf1_v, dtype=np.float32)
                    ).astype(np.float32, copy=False)

                # ---- conf_rs: residual statics の信頼度 ----
                if USE_RESIDUAL_STATICS:
                    conf_rs1 = trace_confidence_from_residual_statics(
                        delta,
                        cmax_rs,
                        valid_rs,
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
                        conf_prob0_viz01, (plo0, phi0) = _scale01_by_percentile(
                            conf_prob0_viz01,
                            pct_lo=float(VIZ_CONF_PROB_PCT_LO),
                            pct_hi=float(VIZ_CONF_PROB_PCT_HI),
                            eps=float(VIZ_CONF_PROB_PCT_EPS),
                        )
                        conf_prob1_viz01, (plo1, phi1) = _scale01_by_percentile(
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

                    _save_conf_scatter(
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
                    _save_conf_scatter(
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
                if float(np.max(np.abs(wave_pad))) <= 0.0:
                    continue

                viz_dir = out_dir / viz_dirname
                viz_dir.mkdir(parents=True, exist_ok=True)
                png_path = viz_dir / f'{segy_path.stem}.ffid{int(ffid)}.png'

                seis_lmo = apply_lmo_linear(
                    wave_pad[:, :n_samples_orig],
                    offs_m,
                    dt_sec=dt_sec,
                    vel_mps=float(LMO_VEL_MPS),
                    fill=0.0,
                    bulk_shift_samples=float(LMO_BULK_SHIFT_SAMPLES),
                )

                p1_viz = pick_argmax.astype(np.float32, copy=False).copy()
                p1_viz[nopick] = np.nan

                p2_viz = pick_out_i.astype(np.float32, copy=False).copy()
                p2_viz[pick_out_i == 0] = np.nan
                p2_viz[invalid] = np.nan

                p1_lmo = lmo_correct_picks(
                    p1_viz, offs_m, dt_sec=dt_sec, vel_mps=float(LMO_VEL_MPS)
                )
                p2_lmo = lmo_correct_picks(
                    p2_viz, offs_m, dt_sec=dt_sec, vel_mps=float(LMO_VEL_MPS)
                )

                seis_win = seis_lmo[:, PLOT_START:PLOT_END].astype(
                    np.float32, copy=False
                )

                pred1_win = (
                    p1_lmo - float(PLOT_START) + float(LMO_BULK_SHIFT_SAMPLES)
                ).astype(np.float32, copy=False)
                pred2_win = (
                    p2_lmo - float(PLOT_START) + float(LMO_BULK_SHIFT_SAMPLES)
                ).astype(np.float32, copy=False)

                keep = np.max(np.abs(seis_win), axis=1) > 0.0
                if not np.any(keep):
                    continue

                x_keep = np.flatnonzero(keep).astype(np.float32)

                seis_win = seis_win[keep]
                pred1_win = pred1_win[keep]
                pred2_win = pred2_win[keep]

                seis_win = (seis_win - np.mean(seis_win, axis=1, keepdims=True)) / (
                    np.std(seis_win, axis=1, keepdims=True) + 1e-10
                )

                pick_overlays = (
                    PickOverlay(
                        pred1_win,
                        unit='sample',
                        label='argmax',
                        marker='o',
                        size=14.0,
                        color='r',
                        alpha=0.9,
                    ),
                    PickOverlay(
                        pred2_win,
                        unit='sample',
                        label=rs_label,
                        marker='x',
                        size=18.0,
                        color='b',
                        alpha=0.9,
                    ),
                )

                if VIZ_SCORE_COMPONENTS:
                    fig, axes = plt.subplots(
                        4,
                        1,
                        figsize=(15, 13),
                        sharex=True,
                        gridspec_kw={'height_ratios': [3.2, 1.0, 1.0, 1.0]},
                    )
                    ax_wiggle, ax_prob, ax_trend, ax_rs = axes
                else:
                    fig, ax_wiggle = plt.subplots(figsize=(15, 10))
                    ax_prob = ax_trend = ax_rs = None

                plot_wiggle(
                    seis_win,
                    ax=ax_wiggle,
                    cfg=WiggleConfig(
                        dt=float(dt_sec),
                        t0=float(PLOT_START) * float(dt_sec),
                        time_axis=1,
                        x=x_keep,
                        normalize='trace',
                        gain=2.0,
                        fill_positive=True,
                        picks=pick_overlays,
                        show_legend=True,
                    ),
                )

                # ---- trend line overlay ----
                if (
                    VIZ_TREND_LINE_ENABLE
                    and ('t_trend_sec' in locals())
                    and (t_trend_sec is not None)
                ):
                    dt = float(dt_sec)
                    if dt <= 0.0:
                        msg = f'dt_sec must be positive, got {dt_sec}'
                        raise ValueError(msg)

                    trend_i = (np.asarray(t_trend_sec, dtype=np.float32) / dt).astype(
                        np.float32, copy=False
                    )
                    trend_i = trend_i.copy()
                    trend_i[invalid] = np.nan

                    trend_lmo = lmo_correct_picks(
                        trend_i, offs_m, dt_sec=dt, vel_mps=float(LMO_VEL_MPS)
                    )

                    trend_win = (
                        trend_lmo - float(PLOT_START) + float(LMO_BULK_SHIFT_SAMPLES)
                    ).astype(np.float32, copy=False)
                    trend_win = trend_win[keep]

                    y_trend_sec = float(PLOT_START) * dt + trend_win * dt

                    ax_wiggle.plot(
                        x_keep,
                        y_trend_sec,
                        lw=float(VIZ_TREND_LINE_LW),
                        alpha=float(VIZ_TREND_LINE_ALPHA),
                        color=str(VIZ_TREND_LINE_COLOR),
                        label=str(VIZ_TREND_LINE_LABEL),
                        zorder=7,
                    )
                    ax_wiggle.legend(loc='best')

                # ---- p1成分スコア（conf_prob1 は percentile scaling で 0..1 描画） ----
                if ax_prob is not None:
                    s_prob_raw = np.asarray(conf_prob1, dtype=np.float32)[keep]
                    if VIZ_CONF_PROB_SCALE_ENABLE:
                        s_prob_viz01, (plo, phi) = _scale01_by_percentile(
                            s_prob_raw,
                            pct_lo=float(VIZ_CONF_PROB_PCT_LO),
                            pct_hi=float(VIZ_CONF_PROB_PCT_HI),
                            eps=float(VIZ_CONF_PROB_PCT_EPS),
                        )
                        prob_title = (
                            f'conf_prob (p1) viz01 pct[{VIZ_CONF_PROB_PCT_LO:.0f},{VIZ_CONF_PROB_PCT_HI:.0f}] '
                            f'raw={plo:.3g}..{phi:.3g}'
                        )
                        s_prob_plot = s_prob_viz01
                    else:
                        prob_title = 'conf_prob (p1) raw'
                        s_prob_plot = s_prob_raw

                    s_trend = np.asarray(conf_trend1, dtype=np.float32)[keep]
                    s_rs = np.asarray(conf_rs1, dtype=np.float32)[keep]

                    _plot_score_panel_1d(
                        ax=ax_prob,
                        x=x_keep,
                        y=s_prob_plot,
                        title=prob_title,
                        ymax=VIZ_YMAX_CONF_PROB,
                        style=VIZ_SCORE_STYLE,
                    )
                    _plot_score_panel_1d(
                        ax=ax_trend,
                        x=x_keep,
                        y=s_trend,
                        title='conf_trend (p1)',
                        ymax=VIZ_YMAX_CONF_TREND,
                        style=VIZ_SCORE_STYLE,
                    )
                    _plot_score_panel_1d(
                        ax=ax_rs,
                        x=x_keep,
                        y=s_rs,
                        title='conf_rs (p1)',
                        ymax=VIZ_YMAX_CONF_RS,
                        style=VIZ_SCORE_STYLE,
                    )
                    ax_rs.set_xlabel('trace index')

                ax_wiggle.set_title(
                    f'{segy_path.stem} ffid={int(ffid)} (LMO v={LMO_VEL_MPS:.1f} m/s)'
                )

                fig.tight_layout()
                fig.savefig(png_path, dpi=200)
                plt.close(fig)
                print(f'[VIZ] saved {png_path}')

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


def main() -> None:
    model = build_model()
    segys = find_segy_files(INPUT_DIR, exts=('.sgy', '.segy'), recursive=False)
    for segy_path in segys:
        print(f'[RUN] using first SEGY: {segys}')
        process_one_segy(
            segy_path=segy_path,
            out_dir=OUT_DIR,
            model=model,
            viz_every_n_shots=VIZ_EVERY_N_SHOTS,
            viz_dirname=VIZ_DIRNAME,
        )


if __name__ == '__main__':
    main()
