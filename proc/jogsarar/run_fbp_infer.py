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
