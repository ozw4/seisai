# %%
#!/usr/bin/env python3
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import segyio
import torch
from _model import NetAE as EncDec2D
from seisai_dataset.config import LoaderConfig
from seisai_dataset.file_info import build_file_info_dataclass
from seisai_dataset.trace_subset_preproc import TraceSubsetLoader
from seisai_engine.postprocess.velocity_filter_op import apply_velocity_filt_prob
from seisai_engine.predict import _run_tiled
from seisai_pick.pickio.io_grstat import numpy2fbcrd
from seisai_pick.residual_statics import refine_firstbreak_residual_statics
from seisai_pick.snap_picks_to_phase import snap_picks_to_phase
from seisai_transforms.signal_ops.scaling.standardize import standardize_per_trace_torch
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

PMAX_TH = 0.07
LTCOR = 5  # legacy snap (comparison / fallback)

SEGY_ENDIAN = 'big'  # "big" or "little"
WAVEFORM_MODE = 'mmap'  # "mmap" or "eager"
HEADER_CACHE_DIR: str | None = None

# 可視化: 各SEGYファイル内で「50ショットおき」に可視化（0なら無効）
VIZ_EVERY_N_SHOTS = 50
VIZ_DIRNAME = 'viz'

# velocity mask params (inference)
VMIN_MASK = 100.0
VMAX_MASK = 8000.0
T0_LO_MS = -10.0
T0_HI_MS = 100.0
TAPER_MS = 10.0

# tile params
TILE_H = 128
TILE_W = 6016
OVERLAP_H = 96  # stride_h = 32
TILES_PER_BATCH = 8

POLARITY_FLIP = False  # model input only

# ---- LMO for visualization (display only) ----
LMO_VEL_MPS = 3200.0
LMO_BULK_SHIFT_SAMPLES = 50.0  # positive shifts later samples

PLOT_START = 0
PLOT_END = 350

# --- residual statics refine ---
USE_RESIDUAL_STATICS = True

# RS_BASE_PICK:
#   'pre'  : snap前（pick0=argmax+threshold）を基準に窓を作ってrefine
#   'snap' : 一度phase snapしてから（pick_pre_snap）それを基準に窓を作ってrefine
RS_BASE_PICK = 'snap'  # 'pre' or 'snap'

# RS_BASE_PICK='snap' のときに使う “事前snap” 設定（refine前のsnap）
RS_PRE_SNAP_MODE = 'peak'
RS_PRE_SNAP_LTCOR = 3

# 窓切り出し（RS_BASE_PICKで選んだpickを基準に窓を作る）
RS_PRE_SAMPLES = 20
RS_POST_SAMPLES = 20

# residual statics params
RS_MAX_LAG = 8
RS_K_NEIGHBORS = 5
RS_N_ITER = 3
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
FINAL_SNAP_MODE = 'peak'
FINAL_SNAP_LTCOR = 3

# --- confidence scoring ---
CONF_ENABLE = True
CONF_VIZ_ENABLE = True
CONF_VIZ_FFID = 0
CONF_HALF_WIN = 20
CONF_KEEP_TH = 0.5
CONF_CENTER_P0 = 0.02
CONF_MASS0 = 0.08

# trend confidence (piecewise RANSAC over abs(offset))
TREND_N_BINS = 4
TREND_RANSAC_ITERS = 200
TREND_INLIER_MS = 4.0
TREND_SIGMA_MS = 6.0
TREND_MIN_PTS = 8

# RS confidence
RS_CMAX_TH = RS_C_TH
RS_ABS_LAG_SOFT = float(RS_MAX_LAG)


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


@dataclass(frozen=True)
class TileWaveStdOnly:
    eps_std: float = 1e-10

    @torch.no_grad()
    def __call__(self, patch: torch.Tensor, *, return_meta: bool = False):
        out = standardize_per_trace_torch(patch, eps=self.eps_std)
        return (out, {}) if return_meta else out


def pad_samples_to_6016(
    x_hw: np.ndarray, w_target: int = 6016
) -> tuple[np.ndarray, int]:
    if x_hw.ndim != 2:
        msg = f'x_hw must be (H,W), got {x_hw.shape}'
        raise ValueError(msg)
    h, w0 = x_hw.shape
    if w0 > w_target:
        msg = f'n_samples={w0} exceeds w_target={w_target}'
        raise ValueError(msg)
    if w0 == w_target:
        return x_hw.astype(np.float32, copy=False), w0
    out = np.zeros((h, w_target), dtype=np.float32)
    out[:, :w0] = x_hw.astype(np.float32, copy=False)
    return out, w0


def build_pick_aligned_window(
    wave_hw: np.ndarray,
    picks: np.ndarray,
    pre: int,
    post: int,
    fill: float = 0.0,
) -> np.ndarray:
    wave = np.asarray(wave_hw, dtype=np.float32)
    pk = np.asarray(picks)

    if wave.ndim != 2:
        msg = f'wave_hw must be 2D (H,W), got {wave.shape}'
        raise ValueError(msg)
    if pk.ndim != 1 or pk.shape[0] != wave.shape[0]:
        msg = f'picks must be 1D length H={wave.shape[0]}, got {pk.shape}'
        raise ValueError(msg)
    if pre < 0 or post <= 0:
        msg = f'pre must be >=0 and post must be >0, got pre={pre}, post={post}'
        raise ValueError(msg)

    H, W = wave.shape
    L = int(pre + post)
    out = np.full((H, L), np.float32(fill), dtype=np.float32)

    for i in range(H):
        p = float(pk[i])
        if (not np.isfinite(p)) or p <= 0.0:
            continue

        c = int(np.rint(p))
        src_l = c - int(pre)
        src_r = c + int(post)

        ov_l = max(0, src_l)
        ov_r = min(W, src_r)
        if ov_l >= ov_r:
            continue

        dst_l = ov_l - src_l
        dst_r = dst_l + (ov_r - ov_l)
        out[i, dst_l:dst_r] = wave[i, ov_l:ov_r]

    return out


def _lmo_shift_samples(
    offsets_m: np.ndarray, *, dt_sec: float, vel_mps: float
) -> np.ndarray:
    if vel_mps <= 0.0:
        msg = f'LMO velocity must be positive, got {vel_mps}'
        raise ValueError(msg)
    if dt_sec <= 0.0:
        msg = f'dt_sec must be positive, got {dt_sec}'
        raise ValueError(msg)
    off = np.asarray(offsets_m, dtype=np.float32)
    return (np.abs(off) / float(vel_mps)) / float(dt_sec)  # (H,) float samples


def apply_lmo_linear(
    wave_hw: np.ndarray,  # (H,W)
    offsets_m: np.ndarray,  # (H,)
    *,
    dt_sec: float,
    vel_mps: float,
    bulk_shift_samples: float = 0.0,
    fill: float = 0.0,
) -> np.ndarray:
    w = np.asarray(wave_hw, dtype=np.float32)
    if w.ndim != 2:
        msg = f'wave_hw must be 2D (H,W), got {w.shape}'
        raise ValueError(msg)
    H, W = w.shape

    shifts = _lmo_shift_samples(offsets_m, dt_sec=dt_sec, vel_mps=vel_mps)
    if shifts.shape != (H,):
        msg = f'offsets_m must be (H,), got {np.asarray(offsets_m).shape}, H={H}'
        raise ValueError(msg)

    xi = np.arange(W, dtype=np.float32)
    out = np.empty_like(w)
    for i in range(H):
        src = xi - float(shifts[i]) + float(bulk_shift_samples)
        out[i] = np.interp(xi, src, w[i], left=fill, right=fill)
    return out


def lmo_correct_picks(
    picks: np.ndarray,
    offsets_m: np.ndarray,
    *,
    dt_sec: float,
    vel_mps: float,
) -> np.ndarray:
    p = np.asarray(picks, dtype=np.float32)
    shifts = _lmo_shift_samples(offsets_m, dt_sec=dt_sec, vel_mps=vel_mps).astype(
        np.float32, copy=False
    )
    if p.shape != shifts.shape:
        msg = f'picks must be (H,), got {p.shape}, shifts={shifts.shape}'
        raise ValueError(msg)
    out = p - shifts
    out[~np.isfinite(p)] = np.nan
    return out


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
        tile_transform=TileWaveStdOnly(),
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


def find_segy_files(in_dir: Path) -> list[Path]:
    exts = ['.sgy', '.segy', '.SGY', '.SEGY']
    files: list[Path] = []
    for e in exts:
        files.extend(sorted(in_dir.glob(f'*{e}')))
    if not files:
        msg = f'no SEGY files found in {in_dir}'
        raise FileNotFoundError(msg)
    return sorted(set(files))


def _valid_pick_mask(picks: np.ndarray, n_samples: int | None = None) -> np.ndarray:
    pk = np.asarray(picks)
    mask = np.isfinite(pk) & (pk > 0)
    if n_samples is not None:
        mask &= pk < int(n_samples)
    return mask


def _prob_local_conf(prob_1d: np.ndarray, center: int, half_win: int) -> float:
    p = np.asarray(prob_1d, dtype=np.float32)
    if p.ndim != 1:
        msg = f'prob_1d must be 1D, got {p.shape}'
        raise ValueError(msg)
    if half_win < 0:
        msg = f'half_win must be >= 0, got {half_win}'
        raise ValueError(msg)
    if center <= 0 or center >= p.shape[0]:
        return 0.0

    l = max(0, int(center) - int(half_win))
    r = min(p.shape[0], int(center) + int(half_win) + 1)
    if l >= r:
        return 0.0

    w_raw = p[l:r].astype(np.float64, copy=False)
    s = float(np.sum(w_raw))
    if s <= 0.0:
        return 0.0
    p_center = float(p[int(center)])
    mass = float(np.clip(s / float(CONF_MASS0), 0.0, 1.0))
    pc = float(np.clip(p_center / float(CONF_CENTER_P0), 0.0, 1.0))

    w = w_raw / s

    n = int(w.size)
    if n <= 0:
        return 0.0
    if n == 1:
        h_norm = 0.0
    else:
        eps = 1e-12
        h = -float(np.sum(w * np.log(w + eps)))
        h_norm = h / float(np.log(float(n)))

    if n == 1:
        margin = float(w[0])
    else:
        top2 = np.partition(w, -2)[-2:]
        p1 = float(np.max(top2))
        p2 = float(np.min(top2))
        margin = p1 - p2

    m0 = 0.2
    conf_shape = (1.0 - h_norm) * float(np.clip(margin / m0, 0.0, 1.0))
    conf = conf_shape * mass * pc
    return float(np.clip(conf, 0.0, 1.0))


def compute_conf_prob(
    prob_hw: np.ndarray, picks_i: np.ndarray, half_win: int
) -> np.ndarray:
    prob = np.asarray(prob_hw, dtype=np.float32)
    picks = np.asarray(picks_i)
    if prob.ndim != 2:
        msg = f'prob_hw must be (H,W), got {prob.shape}'
        raise ValueError(msg)
    if picks.ndim != 1 or picks.shape[0] != prob.shape[0]:
        msg = f'picks_i must be (H,), got {picks.shape}, H={prob.shape[0]}'
        raise ValueError(msg)

    out = np.zeros(prob.shape[0], dtype=np.float32)
    for i in range(prob.shape[0]):
        c = int(picks[i])
        out[i] = np.float32(_prob_local_conf(prob[i], c, int(half_win)))
    return out


def _fit_line_ls(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    if xx.ndim != 1 or yy.ndim != 1 or xx.shape[0] != yy.shape[0]:
        msg = f'x/y must be 1D with same length, got {xx.shape}, {yy.shape}'
        raise ValueError(msg)
    if xx.shape[0] < 2:
        msg = f'need at least 2 points, got {xx.shape[0]}'
        raise ValueError(msg)

    A = np.stack([xx, np.ones_like(xx)], axis=1)
    coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
    a = float(coef[0])
    b = float(coef[1])
    return a, b


def _ransac_line(
    x: np.ndarray, y: np.ndarray, n_iter: int, inlier_th: float
) -> tuple[float, float, np.ndarray]:
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    if xx.ndim != 1 or yy.ndim != 1 or xx.shape[0] != yy.shape[0]:
        msg = f'x/y must be 1D with same length, got {xx.shape}, {yy.shape}'
        raise ValueError(msg)
    n = xx.shape[0]
    if n < 2:
        msg = f'need at least 2 points, got {n}'
        raise ValueError(msg)
    if n_iter <= 0:
        msg = f'n_iter must be > 0, got {n_iter}'
        raise ValueError(msg)
    if inlier_th <= 0.0:
        msg = f'inlier_th must be > 0, got {inlier_th}'
        raise ValueError(msg)

    if float(np.ptp(xx)) <= 1e-12:
        b0 = float(np.median(yy))
        resid0 = np.abs(yy - b0)
        in0 = resid0 <= float(inlier_th)
        if int(np.count_nonzero(in0)) < 2:
            in0 = np.ones(n, dtype=bool)
        return 0.0, float(np.mean(yy[in0])), in0

    rng = np.random.default_rng(0)
    best_in: np.ndarray | None = None
    best_n = -1
    best_err = float('inf')

    for _ in range(int(n_iter)):
        i0, i1 = rng.choice(n, size=2, replace=False)
        dx = float(xx[i1] - xx[i0])
        if abs(dx) <= 1e-12:
            continue
        a = float((yy[i1] - yy[i0]) / dx)
        b = float(yy[i0] - a * xx[i0])
        resid = np.abs(yy - (a * xx + b))
        inlier = resid <= float(inlier_th)
        nin = int(np.count_nonzero(inlier))
        if nin < 2:
            continue
        med = float(np.median(resid[inlier]))
        if nin > best_n or (nin == best_n and med < best_err):
            best_n = nin
            best_err = med
            best_in = inlier

    if best_in is None:
        best_in = np.ones(n, dtype=bool)

    if int(np.count_nonzero(best_in)) >= 2:
        a_ref, b_ref = _fit_line_ls(xx[best_in], yy[best_in])
    else:
        a_ref, b_ref = _fit_line_ls(xx, yy)
    return a_ref, b_ref, best_in


def fit_piecewise_ransac(
    x_abs: np.ndarray,
    y_ms: np.ndarray,
    n_bins: int,
    *,
    n_iter: int,
    inlier_th: float,
    min_pts: int,
) -> tuple[np.ndarray, np.ndarray]:
    xa = np.asarray(x_abs, dtype=np.float32)
    yy = np.asarray(y_ms, dtype=np.float32)
    if xa.ndim != 1 or yy.ndim != 1 or xa.shape[0] != yy.shape[0]:
        msg = f'x_abs/y_ms must be 1D same length, got {xa.shape}, {yy.shape}'
        raise ValueError(msg)
    if n_bins <= 0:
        msg = f'n_bins must be > 0, got {n_bins}'
        raise ValueError(msg)
    if min_pts < 2:
        msg = f'min_pts must be >= 2, got {min_pts}'
        raise ValueError(msg)
    if xa.size == 0:
        msg = 'x_abs is empty'
        raise ValueError(msg)

    q = np.linspace(0.0, 1.0, int(n_bins) + 1)
    edges = np.quantile(xa, q).astype(np.float32, copy=False)

    if not np.isfinite(edges).all():
        msg = 'non-finite quantile edges'
        raise ValueError(msg)
    if float(edges[-1] - edges[0]) <= 1e-6:
        x0 = float(edges[0])
        eps = 1e-3
        edges = np.linspace(
            x0 - eps, x0 + eps, int(n_bins) + 1, dtype=np.float32, endpoint=True
        )

    coef = np.full((int(n_bins), 2), np.nan, dtype=np.float32)

    for k in range(int(n_bins)):
        lo = float(edges[k])
        hi = float(edges[k + 1])
        if k < int(n_bins) - 1:
            mk = (xa >= lo) & (xa < hi)
        else:
            mk = (xa >= lo) & (xa <= hi)
        if int(np.count_nonzero(mk)) < int(min_pts):
            continue
        a, b, _ = _ransac_line(
            xa[mk], yy[mk], n_iter=int(n_iter), inlier_th=float(inlier_th)
        )
        coef[k, 0] = np.float32(a)
        coef[k, 1] = np.float32(b)

    return edges, coef


def predict_piecewise(x_abs: np.ndarray, edges: np.ndarray, coef: np.ndarray) -> np.ndarray:
    xa = np.asarray(x_abs, dtype=np.float32)
    ed = np.asarray(edges, dtype=np.float32)
    cf = np.asarray(coef, dtype=np.float32)
    if xa.ndim != 1:
        msg = f'x_abs must be 1D, got {xa.shape}'
        raise ValueError(msg)
    if ed.ndim != 1:
        msg = f'edges must be 1D, got {ed.shape}'
        raise ValueError(msg)
    if cf.ndim != 2 or cf.shape[1] != 2:
        msg = f'coef must be (n_bins,2), got {cf.shape}'
        raise ValueError(msg)
    if ed.shape[0] != cf.shape[0] + 1:
        msg = f'edges length must be n_bins+1, got edges={ed.shape[0]}, bins={cf.shape[0]}'
        raise ValueError(msg)

    out = np.full(xa.shape, np.nan, dtype=np.float32)
    valid_bin = np.isfinite(cf[:, 0]) & np.isfinite(cf[:, 1])
    valid_idx = np.flatnonzero(valid_bin)
    if valid_idx.size == 0:
        return out

    bins = np.searchsorted(ed, xa, side='right') - 1
    bins = np.clip(bins, 0, cf.shape[0] - 1)

    for i in range(xa.shape[0]):
        xv = float(xa[i])
        if not np.isfinite(xv):
            continue
        k = int(bins[i])
        if not valid_bin[k]:
            k = int(valid_idx[np.argmin(np.abs(valid_idx - k))])
        a = float(cf[k, 0])
        b = float(cf[k, 1])
        out[i] = np.float32(a * xv + b)
    return out


def compute_conf_trend(
    picks_i: np.ndarray,
    x_abs: np.ndarray,
    dt_ms: float,
    edges: np.ndarray | None,
    coef: np.ndarray | None,
    sigma_ms: float,
) -> np.ndarray:
    picks = np.asarray(picks_i)
    xa = np.asarray(x_abs, dtype=np.float32)
    if picks.ndim != 1 or xa.ndim != 1 or picks.shape[0] != xa.shape[0]:
        msg = f'picks_i/x_abs must be (H,), got {picks.shape}, {xa.shape}'
        raise ValueError(msg)
    if sigma_ms <= 0.0:
        msg = f'sigma_ms must be > 0, got {sigma_ms}'
        raise ValueError(msg)

    out = np.zeros(picks.shape[0], dtype=np.float32)
    if edges is None or coef is None:
        return out

    t_hat_ms = predict_piecewise(xa, edges, coef)
    t_ms = picks.astype(np.float32, copy=False) * float(dt_ms)
    valid = _valid_pick_mask(picks) & np.isfinite(t_hat_ms)
    if not np.any(valid):
        return out
    resid = t_ms[valid] - t_hat_ms[valid]
    out[valid] = np.exp(-((resid / float(sigma_ms)) ** 2)).astype(np.float32, copy=False)
    return out


def compute_conf_rs(
    delta_pick: np.ndarray,
    cmax: np.ndarray,
    *,
    c_th: float,
    max_lag: float,
    valid_rs: np.ndarray,
) -> np.ndarray:
    d = np.asarray(delta_pick, dtype=np.float32)
    c = np.asarray(cmax, dtype=np.float32)
    v = np.asarray(valid_rs, dtype=bool)
    if d.ndim != 1 or c.ndim != 1 or v.ndim != 1:
        msg = f'inputs must be 1D, got delta={d.shape}, cmax={c.shape}, valid={v.shape}'
        raise ValueError(msg)
    if d.shape[0] != c.shape[0] or d.shape[0] != v.shape[0]:
        msg = f'input lengths mismatch: {d.shape[0]}, {c.shape[0]}, {v.shape[0]}'
        raise ValueError(msg)

    denom = max(1.0 - float(c_th), 1e-6)
    conf_c = np.clip((c - float(c_th)) / denom, 0.0, 1.0).astype(np.float32, copy=False)
    conf_c = conf_c * v.astype(np.float32, copy=False)

    if max_lag > 0.0:
        conf_l = np.exp(-((np.abs(d) / float(max_lag)) ** 2)).astype(
            np.float32, copy=False
        )
    else:
        conf_l = np.ones(d.shape[0], dtype=np.float32)
    return (conf_c * conf_l).astype(np.float32, copy=False)


def _save_conf_scatter(
    *,
    out_png: Path,
    x_abs: np.ndarray,
    picks_i: np.ndarray,
    dt_ms: float,
    conf_prob: np.ndarray,
    conf_rs: np.ndarray,
    conf_trend: np.ndarray,
    conf_total: np.ndarray,
    title: str,
    trend_hat_ms: np.ndarray | None = None,
) -> None:
    x = np.asarray(x_abs, dtype=np.float32)
    pk = np.asarray(picks_i)
    if x.ndim != 1 or pk.ndim != 1 or x.shape[0] != pk.shape[0]:
        msg = f'x_abs/picks_i must be (H,), got {x.shape}, {pk.shape}'
        raise ValueError(msg)

    y_ms = pk.astype(np.float32, copy=False) * float(dt_ms)
    valid = _valid_pick_mask(pk)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)
    panels = [
        ('conf_prob', np.asarray(conf_prob, dtype=np.float32)),
        ('conf_rs', np.asarray(conf_rs, dtype=np.float32)),
        ('conf_trend', np.asarray(conf_trend, dtype=np.float32)),
        ('conf_total', np.asarray(conf_total, dtype=np.float32)),
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

    axes[1, 0].set_xlabel('|offset| [m]')
    axes[1, 1].set_xlabel('|offset| [m]')
    axes[0, 0].set_ylabel('pick [ms]')
    axes[1, 0].set_ylabel('pick [ms]')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


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
        conf_total0_all = np.zeros(n_traces, dtype=np.float32)
        conf_total1_all = np.zeros(n_traces, dtype=np.float32)
        pick_keep_all = np.zeros(n_traces, dtype=np.int32)
        conf_keep_all = np.zeros(n_traces, dtype=np.float32)
        keep_is_p1_all = np.zeros(n_traces, dtype=bool)
        keep_mask_all = np.zeros(n_traces, dtype=bool)

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
            # RS 無効時は、snapも一切しない（ここが今回の修正点）
            # ========
            if not USE_RESIDUAL_STATICS:
                rs_label = 'pick0'
                pick_pre_snap = (
                    pick0.copy()
                )  # 保存用の箱としては入れておく（意味的には=pick0）
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
                    pick_pre_snap = pick0.copy()  # 保存用（snapしてない）
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

            if CONF_ENABLE:
                dt_ms = float(dt_sec) * 1000.0
                x_abs = np.abs(offs_m).astype(np.float32, copy=False)

                trend_fit_mask = _valid_pick_mask(pick0, n_samples_orig) & (~invalid)
                if int(np.count_nonzero(trend_fit_mask)) >= int(TREND_MIN_PTS):
                    trend_edges, trend_coef = fit_piecewise_ransac(
                        x_abs[trend_fit_mask],
                        pick0[trend_fit_mask].astype(np.float32, copy=False) * dt_ms,
                        n_bins=int(TREND_N_BINS),
                        n_iter=int(TREND_RANSAC_ITERS),
                        inlier_th=float(TREND_INLIER_MS),
                        min_pts=int(TREND_MIN_PTS),
                    )
                else:
                    trend_edges, trend_coef = None, None

                conf_prob0 = compute_conf_prob(
                    prob_hw, pick_pre_snap, half_win=int(CONF_HALF_WIN)
                )
                conf_prob1 = compute_conf_prob(
                    prob_hw, pick_out_i, half_win=int(CONF_HALF_WIN)
                )

                conf_trend0 = compute_conf_trend(
                    pick_pre_snap,
                    x_abs,
                    dt_ms,
                    trend_edges,
                    trend_coef,
                    sigma_ms=float(TREND_SIGMA_MS),
                )
                conf_trend1 = compute_conf_trend(
                    pick_out_i,
                    x_abs,
                    dt_ms,
                    trend_edges,
                    trend_coef,
                    sigma_ms=float(TREND_SIGMA_MS),
                )

                if USE_RESIDUAL_STATICS:
                    conf_rs1 = compute_conf_rs(
                        delta,
                        cmax_rs,
                        c_th=float(RS_CMAX_TH),
                        max_lag=float(RS_ABS_LAG_SOFT),
                        valid_rs=valid_rs,
                    )
                else:
                    conf_rs1 = np.ones(idx.shape[0], dtype=np.float32)

                conf_total0 = (conf_prob0 * conf_trend0).astype(np.float32, copy=False)
                conf_total1 = (conf_prob1 * conf_trend1 * conf_rs1).astype(
                    np.float32, copy=False
                )

                keep_is_p1 = conf_total1 >= conf_total0
                conf_keep = np.where(keep_is_p1, conf_total1, conf_total0).astype(
                    np.float32, copy=False
                )
                pick_keep = np.where(keep_is_p1, pick_out_i, pick_pre_snap).astype(
                    np.int32, copy=False
                )
                keep_mask = conf_keep >= float(CONF_KEEP_TH)
                pick_keep = pick_keep.copy()
                pick_keep[~keep_mask] = 0

                if CONF_VIZ_ENABLE and int(ffid) == int(CONF_VIZ_FFID):
                    conf_viz_dir = out_dir / 'conf_viz'
                    conf_viz_dir.mkdir(parents=True, exist_ok=True)
                    trend_hat_ms = (
                        None
                        if trend_edges is None or trend_coef is None
                        else predict_piecewise(x_abs, trend_edges, trend_coef)
                    )

                    p0_png = conf_viz_dir / f'{segy_path.stem}.ffid{int(ffid)}.p0.conf.png'
                    p1_png = conf_viz_dir / f'{segy_path.stem}.ffid{int(ffid)}.p1.conf.png'
                    _save_conf_scatter(
                        out_png=p0_png,
                        x_abs=x_abs,
                        picks_i=pick_pre_snap,
                        dt_ms=dt_ms,
                        conf_prob=conf_prob0,
                        conf_rs=np.ones(idx.shape[0], dtype=np.float32),
                        conf_trend=conf_trend0,
                        conf_total=conf_total0,
                        title=f'{segy_path.stem} ffid={int(ffid)} p0 (pick_pre_snap)',
                        trend_hat_ms=trend_hat_ms,
                    )
                    _save_conf_scatter(
                        out_png=p1_png,
                        x_abs=x_abs,
                        picks_i=pick_out_i,
                        dt_ms=dt_ms,
                        conf_prob=conf_prob1,
                        conf_rs=conf_rs1,
                        conf_trend=conf_trend1,
                        conf_total=conf_total1,
                        title=f'{segy_path.stem} ffid={int(ffid)} p1 (pick_final)',
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
                conf_total0 = np.zeros(idx.shape[0], dtype=np.float32)
                conf_total1 = np.zeros(idx.shape[0], dtype=np.float32)
                pick_keep = np.zeros(idx.shape[0], dtype=np.int32)
                conf_keep = np.zeros(idx.shape[0], dtype=np.float32)
                keep_is_p1 = np.zeros(idx.shape[0], dtype=bool)
                keep_mask = np.zeros(idx.shape[0], dtype=bool)

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
            conf_prob0_all[idx] = conf_prob0
            conf_prob1_all[idx] = conf_prob1
            conf_trend0_all[idx] = conf_trend0
            conf_trend1_all[idx] = conf_trend1
            conf_rs1_all[idx] = conf_rs1
            conf_total0_all[idx] = conf_total0
            conf_total1_all[idx] = conf_total1
            pick_keep_all[idx] = pick_keep
            conf_keep_all[idx] = conf_keep
            keep_is_p1_all[idx] = keep_is_p1
            keep_mask_all[idx] = keep_mask

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
                p1_viz[nopick] = (
                    np.nan
                )  # pmax<th や invalid は表示しない(argmax側も揃えるな)

                p2_viz = pick_out_i.astype(np.float32, copy=False).copy()
                p2_viz[pick_out_i == 0] = np.nan  # 出力pickの no-pick は表示しない
                # もし invalid をさらに落とすなら:
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

                fig, ax = plt.subplots(figsize=(15, 10))
                plot_wiggle(
                    seis_win,
                    ax=ax,
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

                ax.set_title(
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
            conf_prob0=conf_prob0_all,
            conf_prob1=conf_prob1_all,
            conf_trend0=conf_trend0_all,
            conf_trend1=conf_trend1_all,
            conf_rs1=conf_rs1_all,
            conf_total0=conf_total0_all,
            conf_total1=conf_total1_all,
            pick_keep=pick_keep_all,
            conf_keep=conf_keep_all,
            keep_is_p1=keep_is_p1_all,
            keep_mask=keep_mask_all,
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
    segy_path = find_segy_files(INPUT_DIR)[0]
    print(f'[RUN] using first SEGY: {segy_path}')
    process_one_segy(
        segy_path=segy_path,
        out_dir=OUT_DIR,
        model=model,
        viz_every_n_shots=VIZ_EVERY_N_SHOTS,
        viz_dirname=VIZ_DIRNAME,
    )


if __name__ == '__main__':
    main()
