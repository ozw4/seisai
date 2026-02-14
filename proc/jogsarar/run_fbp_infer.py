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
from matplotlib.collections import PolyCollection
from seisai_dataset.config import LoaderConfig
from seisai_dataset.file_info import build_file_info_dataclass
from seisai_dataset.trace_subset_preproc import TraceSubsetLoader
from seisai_engine.postprocess.velocity_filter_op import apply_velocity_filt_prob
from seisai_engine.predict import _run_tiled
from seisai_pick.pickio.io_grstat import numpy2fbcrd
from seisai_pick.snap_picks_to_phase import snap_picks_to_phase
from seisai_transforms.signal_ops.scaling.standardize import standardize_per_trace_torch

BuildFileInfoFn = Callable[..., Any]
SnapPicksFn = Callable[..., Any]
Numpy2FbCrdFn = Callable[..., Any]
StdPerTraceFn = Callable[..., torch.Tensor]

# =========================
# CONFIG (ここだけ直書き)
# =========================
INPUT_DIR = Path('/home/dcuser/data/ActiveSeisField/jogsarar')
OUT_DIR = Path('/home/dcuser/data/ActiveSeisField/jogsarar_out')
WEIGHTS_PATH = Path('/home/dcuser/data/model_weight/fbseg_caformer_b36.pth')

BACKBONE = 'caformer_b36.sail_in22k_ft_in1k'
DEVICE = 'cuda'  # "cpu" も可
USE_TTA = True

PMAX_TH = 0.05
LTCOR = 5

SEGY_ENDIAN = 'big'  # "big" or "little"
WAVEFORM_MODE = 'mmap'  # "mmap" or "eager"
HEADER_CACHE_DIR: str | None = None

# 可視化: 各SEGYファイル内で「50ショットおき」に可視化（0なら無効）
VIZ_EVERY_N_SHOTS = 50
VIZ_DIRNAME = 'viz'

# velocity mask params
VMIN_MASK = 100.0
VMAX_MASK = 8000.0
T0_LO_MS = -10.0
T0_HI_MS = 80.0
TAPER_MS = 10.0

# tile params
TILE_H = 128
TILE_W = 6016
OVERLAP_H = 96  # stride_h = 32
TILES_PER_BATCH = 8

# sampling rate x2 (upsample) before inference, then downsample prob back
UPSAMPLE_X2 = True


# ---- velocity mask allowing vmin==0 ----
def _cos_ramp01(s: torch.Tensor) -> torch.Tensor:
    s = s.clamp(0.0, 1.0)
    return 0.5 - 0.5 * torch.cos(torch.pi * s)


_INVALID = (-9999, -1, 0)


def _valid_pick_mask(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p)
    m = np.ones(p.shape, dtype=bool)
    for v in _INVALID:
        m &= p != v
    return m & np.isfinite(p)


def _shift_1d_noroll(x: np.ndarray, shift: int, fill: float = 0.0) -> np.ndarray:
    """Non-circular shift. shift>0: move left (earlier), shift<0: move right (later)."""
    n = x.shape[0]
    out = np.full((n,), fill, dtype=x.dtype)
    if shift == 0:
        out[:] = x
        return out
    if shift > 0:
        if shift < n:
            out[: n - shift] = x[shift:]
        return out
    s = -shift
    if s < n:
        out[s:] = x[: n - s]
    return out


def align_by_picks_for_viz(
    seis: np.ndarray,  # (H,W)
    prob: np.ndarray | None,  # (H,W) or None
    pred: np.ndarray,  # (H,)
    pred2: np.ndarray | None = None,
    *,
    ref: int | None = None,  # None -> median of valid pred2 if exists else pred
    fill_seis: float = 0.0,
    fill_prob: float = 0.0,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None, int]:
    """Align traces so that pred_align becomes 'ref' (sample index).
    Returns aligned (seis, prob, pred, pred2, ref_used).
    """
    seis = np.asarray(seis)
    H, W = seis.shape

    pred = np.asarray(pred).astype(np.int64, copy=False)
    pred2a = None if pred2 is None else np.asarray(pred2).astype(np.int64, copy=False)

    pred_align = pred2a if pred2a is not None else pred
    m = _valid_pick_mask(pred_align)
    if not m.any():
        return seis, prob, pred, pred2a, 0

    ref_used = int(np.median(pred_align[m])) if ref is None else int(ref)

    shift = np.zeros((H,), dtype=np.int64)
    shift[m] = pred_align[m] - ref_used

    seis_al = np.empty_like(seis)
    prob_al = None if prob is None else np.empty_like(prob)

    pred_al = pred.astype(np.float32, copy=True)
    pred2_al = None if pred2a is None else pred2a.astype(np.float32, copy=True)

    for i in range(H):
        s = int(shift[i])
        seis_al[i] = _shift_1d_noroll(seis[i], s, fill=fill_seis)
        if prob_al is not None:
            prob_al[i] = _shift_1d_noroll(prob[i], s, fill=fill_prob)

        if _valid_pick_mask(pred[i]):
            pred_al[i] = float(pred[i] - s)
        else:
            pred_al[i] = np.nan

        if pred2_al is not None:
            if _valid_pick_mask(pred2a[i]):
                pred2_al[i] = float(pred2a[i] - s)
            else:
                pred2_al[i] = np.nan

    return seis_al, prob_al, pred_al, pred2_al, ref_used


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
        raise ValueError(f'vmax must be positive, got {vmax}')
    if vmin < 0.0:
        raise ValueError(f'vmin must be >=0, got {vmin}')
    if W <= 0:
        raise ValueError(f'W must be positive, got {W}')
    if vmin > 0.0 and vmax < vmin:
        raise ValueError(f'vmax must be >= vmin. got vmin={vmin}, vmax={vmax}')
    if offsets_m.ndim != 2:
        raise ValueError(f'offsets_m must be (B,H), got {tuple(offsets_m.shape)}')

    dt = float(dt_sec)
    if dt <= 0.0:
        raise ValueError(f'dt_sec must be positive, got {dt_sec}')

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


def plot(seis, pred, ax=None, title=None, pred2=None):
    amp = 2

    tmp = seis.copy()
    n_tr, n_samp = tmp.shape

    y = np.arange(n_samp)

    tmp_max = tmp.max()
    trace_idx = np.arange(n_tr, dtype=tmp.dtype)[:, None]
    xs = (tmp / tmp_max) * amp + trace_idx

    segs = np.zeros((n_tr, n_samp, 2))
    segs[:, :, 0] = xs
    segs[:, :, 1] = y

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))

    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(y.max(), y.min())

    line_segments = PolyCollection(
        segs,
        linewidths=(0.01, 0.02, 0.02, 0.04),
        facecolors='k',
        edgecolors='k',
        linestyle='solid',
    )
    ax.add_collection(line_segments)

    tr = np.arange(n_tr)
    ax.scatter(
        tr,
        pred,
        s=5,
        marker='o',
        linewidths=0.5,
        facecolor='None',
        edgecolor='r',
        alpha=0.9,
        label='Prediction',
    )

    if pred2 is not None:
        pred2 = pred2.astype('f')
        ax.scatter(
            tr,
            pred2,
            s=5,
            marker='o',
            linewidths=0.5,
            facecolor='None',
            edgecolor='b',
            alpha=0.9,
            label='snapped',
        )

    ax.set_ylabel('Sample')
    ax.legend(loc='upper left')

    plt.subplots_adjust(hspace=0.01)
    plt.tight_layout()
    plt.savefig(title, dpi=200)


# ---- tile transform: wave std only ----
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
        raise ValueError(f'x_hw must be (H,W), got {x_hw.shape}')
    h, w0 = x_hw.shape
    if w0 > w_target:
        raise ValueError(f'n_samples={w0} exceeds w_target={w_target}')
    if w0 == w_target:
        return x_hw.astype(np.float32, copy=False), w0
    out = np.zeros((h, w_target), dtype=np.float32)
    out[:, :w0] = x_hw.astype(np.float32, copy=False)
    return out, w0


def upsample_time_x2_linear(wave_hw: np.ndarray) -> np.ndarray:
    """Time upsample x2 (dt -> dt/2) by inserting midpoints.
    Output length = 2*W0. Original samples align to even indices.
    """
    if wave_hw.ndim != 2:
        raise ValueError(f'wave_hw must be (H,W), got {wave_hw.shape}')
    H, W0 = wave_hw.shape
    wave_hw = wave_hw.astype(np.float32, copy=False)

    out = np.empty((H, 2 * W0), dtype=np.float32)
    out[:, 0::2] = wave_hw
    if W0 > 1:
        out[:, 1:-1:2] = 0.5 * (wave_hw[:, :-1] + wave_hw[:, 1:])
    out[:, -1] = wave_hw[:, -1]
    return out


@torch.no_grad()
def infer_gather_prob(
    *,
    model: torch.nn.Module,
    wave_hw: np.ndarray,  # (H,W0) original dt
    offsets_m: np.ndarray,  # (H,)
    dt_sec: float,  # original dt
) -> tuple[np.ndarray, int]:
    device = next(model.parameters()).device

    if wave_hw.ndim != 2:
        raise ValueError(f'wave_hw must be (H,W0), got {wave_hw.shape}')
    H0, W0 = wave_hw.shape
    if offsets_m.shape != (H0,):
        raise ValueError(f'offsets_m must be (H,), got {offsets_m.shape}, H={H0}')

    n_samples_orig = int(W0)

    if UPSAMPLE_X2:
        wave_in = upsample_time_x2_linear(wave_hw)  # (H, 2*W0)
        dt_in = float(dt_sec) * 0.5
        n_samples_in_orig = int(wave_in.shape[1])  # 2*W0
    else:
        wave_in = wave_hw.astype(np.float32, copy=False)
        dt_in = float(dt_sec)
        n_samples_in_orig = int(wave_in.shape[1])

    wave_pad, n_samples_in = pad_samples_to_6016(wave_in, w_target=TILE_W)  # (H,6016)
    wave_pad = -wave_pad  # polarity flip
    H, W = wave_pad.shape
    if H != H0 or W != TILE_W:
        raise ValueError(f'unexpected padded shape: {wave_pad.shape}')

    tile_h = min(TILE_H, H)
    ov_h = OVERLAP_H if H >= TILE_H else 0
    tile = (tile_h, TILE_W)
    overlap = (ov_h, 0)

    x = (
        torch.from_numpy(wave_pad)
        .to(device=device, dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
    )  # (1,1,H,W)

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
        raise ValueError(f'expected out_chans=1, got {logits.shape[1]}')

    prob = torch.softmax(logits, dim=-1)  # (1,1,H,6016) on dt_in grid

    offs_t = (
        torch.from_numpy(offsets_m).to(device=device, dtype=torch.float32).view(1, H)
    )
    mask = make_velocity_feasible_filt_allow_vmin0(
        offsets_m=offs_t,
        dt_sec=float(dt_in),  # ★dt/2 when upsampled
        W=W,
        vmin=float(VMIN_MASK),
        vmax=float(VMAX_MASK),
        t0_lo_ms=float(T0_LO_MS),
        t0_hi_ms=float(T0_HI_MS),
        taper_ms=float(TAPER_MS),
    )

    prob = apply_velocity_filt_prob(prob, mask, renorm=False, time_dim=-1)

    # upsampled-pad領域は pick対象外
    if n_samples_in < W:
        prob[:, :, :, n_samples_in:] = 0.0

    prob_up = prob[0, 0].detach().to('cpu', dtype=torch.float32).numpy()  # (H,6016)

    if UPSAMPLE_X2:
        # downsample (dt/2 -> dt): take even indices aligned to original samples
        prob_ds = prob_up[:, :n_samples_in_orig:2]  # -> (H, W0)
        if prob_ds.shape[1] != n_samples_orig:
            raise ValueError(
                f'downsample length mismatch: got {prob_ds.shape[1]}, expected {n_samples_orig}'
            )

        prob_out = np.zeros((H, TILE_W), dtype=np.float32)
        prob_out[:, :n_samples_orig] = prob_ds
        return prob_out, n_samples_orig

    # no upsample: output already on original dt grid
    prob_out = np.zeros((H, TILE_W), dtype=np.float32)
    prob_out[:, :n_samples_orig] = prob_up[:, :n_samples_orig]
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
        raise RuntimeError('CUDA requested but not available')

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
        raise ValueError("ckpt['model_ema'] is not a state_dict")

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
        raise FileNotFoundError(f'no SEGY files found in {in_dir}')
    return sorted(set(files))


def process_one_segy(
    *,
    segy_path: Path,
    out_dir: Path,
    model: torch.nn.Module,
    build_file_info_dataclass: BuildFileInfoFn = build_file_info_dataclass,
    TraceSubsetLoaderCls: type[TraceSubsetLoader] = TraceSubsetLoader,
    LoaderConfigCls: type[LoaderConfig] = LoaderConfig,
    snap_picks_to_phase_fn: SnapPicksFn = snap_picks_to_phase,
    numpy2fbcrd_fn: Numpy2FbCrdFn = numpy2fbcrd,
    standardize_per_trace_torch_fn: StdPerTraceFn = standardize_per_trace_torch,
    viz_every_n_shots: int = 0,
    viz_dirname: str = 'viz',
) -> None:
    info = build_file_info_dataclass(
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
            raise ValueError('header arrays must be length n_traces')

        if info.ffid_key_to_indices is None:
            raise ValueError('ffid_key_to_indices is None (cannot group by ffid)')

        prob_all = np.zeros((n_traces, TILE_W), dtype=np.float16)
        trace_indices_all = np.arange(n_traces, dtype=np.int64)

        max_chno = int(chno_values.max(initial=0))
        ffids_sorted = sorted(int(x) for x in info.ffid_key_to_indices)
        ffid_to_row = {ff: i for i, ff in enumerate(ffids_sorted)}
        fb_mat = np.zeros((len(ffids_sorted), max_chno), dtype=np.int32)

        viz_ffids = (
            set(ffids_sorted[::viz_every_n_shots]) if viz_every_n_shots > 0 else set()
        )

        loader = TraceSubsetLoaderCls(LoaderConfigCls(pad_traces_to=1))

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
            )  # (H,6016) at original dt

            if invalid.any():
                prob_hw[invalid, :] = 0.0

            pick = np.argmax(prob_hw, axis=1).astype(np.int32, copy=False)
            pmax = np.max(prob_hw, axis=1).astype(np.float32, copy=False)

            nopick = (pmax < float(PMAX_TH)) | invalid
            pick = pick.copy()
            pick[nopick] = 0

            wave_pad = np.zeros((wave_hw.shape[0], TILE_W), dtype=np.float32)
            wave_pad[:, :n_samples_orig] = wave_hw.astype(np.float32, copy=False)
            pick = snap_picks_to_phase_fn(
                pick, wave_pad, mode='trough', ltcor=int(LTCOR)
            )

            too_late = pick >= int(n_samples_orig)
            if np.any(too_late):
                pick = pick.copy()
                pick[too_late] = 0

            prob_all[idx, :] = prob_hw.astype(np.float16, copy=False)

            row = ffid_to_row[int(ffid)]
            for j in range(pick.shape[0]):
                cno = int(chno_g[j])
                if 1 <= cno <= max_chno:
                    fb_mat[row, cno - 1] = int(pick[j])

            if viz_ffids and int(ffid) in viz_ffids:
                if float(np.max(np.abs(wave_pad))) <= 0.0:
                    continue

                viz_dir = out_dir / viz_dirname
                viz_dir.mkdir(parents=True, exist_ok=True)
                png_path = viz_dir / f'{segy_path.stem}.ffid{int(ffid)}.png'

                pred_argmax = np.argmax(prob_hw, axis=1).astype(np.int64, copy=False)
                pred_snap = pick.astype(np.int64, copy=False)

                seis_al, prob_al, pred1_al, pred2_al, ref_used = align_by_picks_for_viz(
                    wave_pad, prob_hw, pred_argmax, pred2=pred_snap
                )

                seis_al = (seis_al - np.mean(seis_al, axis=1, keepdims=True)) / (
                    np.std(seis_al, axis=1, keepdims=True) + 1e-10
                )

                plot(
                    seis_al[:, ref_used - 128 : ref_used + 128],
                    pred1_al - ref_used + 128,
                    pred2=pred2_al - ref_used + 128 if pred2_al is not None else None,
                    title=str(png_path),
                )
                plt.close()
                print(f'[VIZ] saved {png_path}')

        out_dir.mkdir(parents=True, exist_ok=True)
        stem = segy_path.stem

        npz_path = out_dir / f'{stem}.prob.npz'
        np.savez_compressed(
            npz_path,
            prob=prob_all,  # float16 (n_traces,6016) at original dt
            dt_sec=np.float32(dt_sec),
            n_samples_orig=np.int32(n_samples),
            ffid_values=ffid_values,
            chno_values=chno_values,
            offsets=offsets,
            trace_indices=trace_indices_all,
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
    segy_files = find_segy_files(INPUT_DIR)
    for segy_path in segy_files:
        process_one_segy(
            segy_path=segy_path,
            out_dir=OUT_DIR,
            model=model,
            viz_every_n_shots=VIZ_EVERY_N_SHOTS,
            viz_dirname=VIZ_DIRNAME,
        )


if __name__ == '__main__':
    main()
