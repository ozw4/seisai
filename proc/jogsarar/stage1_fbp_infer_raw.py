# %%
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import segyio
import torch
from _model import NetAE as EncDec2D
from common.paths import stage1_out_dir as _stage1_out_dir
from config_io import (
    build_yaml_defaults,
    coerce_optional_bool,
    coerce_optional_float,
    coerce_optional_int,
    coerce_path,
    load_yaml_dict,
    normalize_segy_exts,
    parse_args_with_yaml_defaults,
)
from stage1.process_one import process_one_segy as _process_one_segy
from jogsarar_shared import (
    TilePerTraceStandardize,
    compute_conf_rs_from_residual_statics,
    compute_conf_trend_gaussian_var,
    compute_residual_statics_metrics,
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
from seisai_pick.score.confidence_from_prob import (
    trace_confidence_from_prob_local_window,
)
from seisai_pick.snap_picks_to_phase import snap_picks_to_phase
from seisai_pick.trend.trend_fit import robust_linear_trend
from seisai_utils.viz_wiggle import PickOverlay, WiggleConfig, plot_wiggle

BuildFileInfoFn = Callable[..., Any]
SnapPicksFn = Callable[..., Any]
Numpy2FbCrdFn = Callable[..., Any]


@dataclass(frozen=True)
class Stage1Cfg:
    in_segy_root: Path = Path('/home/dcuser/data/ActiveSeisField/jogsarar')
    out_infer_root: Path = Path('/home/dcuser/data/ActiveSeisField/jogsarar_out')
    weights_path: Path = Path('/home/dcuser/data/model_weight/fbseg_caformer_b36.pth')
    segy_exts: tuple[str, ...] = ('.sgy', '.segy')
    recursive: bool = False
    backbone: str = 'caformer_b36.sail_in22k_ft_in1k'
    device: str = 'cuda'
    use_tta: bool = True
    pmax_th: float = 0.0
    ltcor: int = 5
    segy_endian: str = 'big'
    waveform_mode: str = 'mmap'
    header_cache_dir: str | None = None
    viz_every_n_shots: int = 100
    viz_dirname: str = 'viz'
    vmin_mask: float = 100.0
    vmax_mask: float = 5000.0
    t0_lo_ms: float = -10.0
    t0_hi_ms: float = 100.0
    taper_ms: float = 10.0
    tile_h: int = 128
    tile_w: int = 6016
    overlap_h: int = 96
    tiles_per_batch: int = 8
    polarity_flip: bool = True
    lmo_vel_mps: float = 3200.0
    lmo_bulk_shift_samples: float = 50.0
    plot_start: int = 0
    plot_end: int = 350
    viz_score_components: bool = True
    viz_score_style: str = 'bar'
    viz_conf_prob_scale_enable: bool = True
    viz_conf_prob_pct_lo: float = 5.0
    viz_conf_prob_pct_hi: float = 99.0
    viz_conf_prob_pct_eps: float = 1e-12
    viz_ymax_conf_prob: float | None = 1.0
    viz_ymax_conf_trend: float | None = 1.0
    viz_ymax_conf_rs: float | None = 1.0
    viz_trend_line_enable: bool = True
    viz_trend_line_lw: float = 1.6
    viz_trend_line_alpha: float = 0.9
    viz_trend_line_label: str = 'trend'
    viz_trend_line_color: str = 'g'
    use_residual_statics: bool = True
    rs_base_pick: str = 'snap'
    rs_pre_snap_mode: str = 'trough'
    rs_pre_snap_ltcor: int = 3
    rs_pre_samples: int = 20
    rs_post_samples: int = 20
    rs_max_lag: int = 8
    rs_k_neighbors: int = 5
    rs_n_iter: int = 2
    rs_mode: str = 'diff'
    rs_c_th: float = 0.5
    rs_smooth_method: str = 'wls'
    rs_lam: float = 5.0
    rs_subsample: bool = True
    rs_propagate_low_corr: bool = False
    rs_taper: str = 'hann'
    rs_taper_power: float = 1.0
    rs_lag_penalty: float = 0.10
    rs_lag_penalty_power: float = 1.0
    use_final_snap: bool = True
    final_snap_mode: str = 'trough'
    final_snap_ltcor: int = 3
    conf_enable: bool = True
    conf_viz_enable: bool = True
    conf_viz_ffid: int = 2147
    conf_half_win: int = 20
    trend_local_enable: bool = True
    trend_local_use_abs_offset: bool = False
    trend_local_sort_offsets: bool = False
    trend_side_split_enable: bool = True
    trend_local_use_abs_offset_header: bool = True
    trend_local_section_len: int = 16
    trend_local_stride: int = 4
    trend_local_huber_c: float = 1.345
    trend_local_iters: int = 3
    trend_local_vmin_mps: float | None = 300.0
    trend_local_vmax_mps: float | None = 8000.0
    trend_local_weight_mode: str = 'uniform'
    trend_sigma_ms: float = 6.0
    trend_min_pts: int = 12
    trend_var_half_win_traces: int = 8
    trend_var_sigma_std_ms: float = 6.0
    trend_var_min_count: int = 3
    rs_cmax_th: float = 0.5
    rs_abs_lag_soft: float = 8.0
    save_trend_to_npz: bool = True
    trend_source_label: str = 'pick_final'
    trend_method_label: str = 'local_irls_split_sides'


DEFAULT_STAGE1_CFG = Stage1Cfg()

INPUT_DIR = DEFAULT_STAGE1_CFG.in_segy_root
OUT_DIR = DEFAULT_STAGE1_CFG.out_infer_root
WEIGHTS_PATH = DEFAULT_STAGE1_CFG.weights_path
VIZ_EVERY_N_SHOTS = DEFAULT_STAGE1_CFG.viz_every_n_shots
VIZ_DIRNAME = DEFAULT_STAGE1_CFG.viz_dirname


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


def build_model(*, weights_path: Path = WEIGHTS_PATH) -> torch.nn.Module:
    cfg = replace(DEFAULT_STAGE1_CFG, weights_path=Path(weights_path))
    return build_model_from_cfg(cfg)


def build_model_from_cfg(cfg: Stage1Cfg) -> torch.nn.Module:
    DEVICE = str(cfg.device)
    BACKBONE = str(cfg.backbone)
    USE_TTA = bool(cfg.use_tta)
    weights_path = Path(cfg.weights_path)

    if DEVICE.startswith('cuda') and not torch.cuda.is_available():
        msg = 'CUDA requested but not available'
        raise RuntimeError(msg)

    ckpt_path = Path(weights_path).expanduser()
    if not ckpt_path.is_absolute():
        ckpt_path = (Path.cwd() / ckpt_path).resolve()
    if not ckpt_path.is_file():
        msg = f'weights not found: {ckpt_path}'
        raise FileNotFoundError(msg)

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

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt['model_ema']

    if (
        not isinstance(sd, Mapping)
        or not sd
        or not all(isinstance(v, torch.Tensor) for v in sd.values())
    ):
        msg = "ckpt['model_ema'] is not a state_dict"
        raise ValueError(msg)

    sd = _strip_prefix(sd)
    print(f'[CKPT] load from: {ckpt_path} (model_ema direct state_dict)')

    model.load_state_dict(sd, strict=True)
    model.to(device=torch.device(DEVICE))
    model.eval()
    return model


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
        save_conf_scatter_fn=_save_conf_scatter,
        scale01_by_percentile_fn=_scale01_by_percentile,
        plot_score_panel_1d_fn=_plot_score_panel_1d,
    )


_STAGE1_CFG_KEYS = tuple(Stage1Cfg.__dataclass_fields__.keys())
_PATH_KEYS = {'in_segy_root', 'out_infer_root', 'weights_path'}
_OPTIONAL_FLOAT_KEYS = {
    'viz_ymax_conf_prob',
    'viz_ymax_conf_trend',
    'viz_ymax_conf_rs',
    'trend_local_vmin_mps',
    'trend_local_vmax_mps',
}
_OPTIONAL_STR_KEYS = {'header_cache_dir'}


def _coerce_required_int(key: str, value: object) -> int:
    out = coerce_optional_int(key, value)
    if out is None:
        msg = f'config[{key}] must not be null'
        raise TypeError(msg)
    return int(out)


def _coerce_required_bool(key: str, value: object) -> bool:
    out = coerce_optional_bool(key, value)
    if out is None:
        msg = f'config[{key}] must not be null'
        raise TypeError(msg)
    return bool(out)


def _coerce_required_float(key: str, value: object) -> float:
    out = coerce_optional_float(key, value)
    if out is None:
        msg = f'config[{key}] must not be null'
        raise TypeError(msg)
    return float(out)


def _coerce_optional_float_field(key: str, value: object) -> float | None:
    out = coerce_optional_float(key, value)
    if out is None:
        return None
    return float(out)


def _coerce_required_str(key: str, value: object) -> str:
    if not isinstance(value, str):
        msg = f'config[{key}] must be str, got {type(value).__name__}'
        raise TypeError(msg)
    if value == '':
        msg = f'config[{key}] must not be empty'
        raise ValueError(msg)
    return value


def _coerce_optional_str_field(key: str, value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        msg = f'config[{key}] must be str or null, got {type(value).__name__}'
        raise TypeError(msg)
    return value


def _build_stage1_yaml_coercers() -> dict[str, Callable[[object], object]]:
    coercers: dict[str, Callable[[object], object]] = {}
    for key in _STAGE1_CFG_KEYS:
        if key in _PATH_KEYS:
            coercers[key] = partial(coerce_path, key, allow_none=False)
            continue
        if key == 'segy_exts':
            coercers[key] = normalize_segy_exts
            continue
        if key in _OPTIONAL_FLOAT_KEYS:
            coercers[key] = partial(_coerce_optional_float_field, key)
            continue
        if key in _OPTIONAL_STR_KEYS:
            coercers[key] = partial(_coerce_optional_str_field, key)
            continue

        default_value = getattr(DEFAULT_STAGE1_CFG, key)
        if isinstance(default_value, bool):
            coercers[key] = partial(_coerce_required_bool, key)
            continue
        if isinstance(default_value, int):
            coercers[key] = partial(_coerce_required_int, key)
            continue
        if isinstance(default_value, float):
            coercers[key] = partial(_coerce_required_float, key)
            continue
        if isinstance(default_value, str):
            coercers[key] = partial(_coerce_required_str, key)
            continue

        msg = f'unsupported stage1 config key type: key={key}'
        raise RuntimeError(msg)
    return coercers


def _load_yaml_defaults(config_path: Path) -> dict[str, object]:
    loaded = load_yaml_dict(config_path)
    return build_yaml_defaults(
        loaded,
        allowed_keys=set(_STAGE1_CFG_KEYS),
        coercers=_build_stage1_yaml_coercers(),
    )


def load_stage1_cfg_yaml(config_path: Path) -> Stage1Cfg:
    updates = _load_yaml_defaults(config_path)
    cfg = replace(DEFAULT_STAGE1_CFG, **updates)
    return _validate_stage1_cfg(cfg)


def _write_stage1_used_cfg_yaml(cfg: Stage1Cfg, *, out_root: Path) -> Path:
    serializable: dict[str, object] = {}
    for key in _STAGE1_CFG_KEYS:
        value = getattr(cfg, key)
        if isinstance(value, Path):
            serializable[key] = str(value)
            continue
        if isinstance(value, tuple):
            serializable[key] = [str(x) for x in value]
            continue
        serializable[key] = value

    target_root = Path(out_root).expanduser().resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    out_path = target_root / 'stage1_used.yaml'

    import yaml

    with out_path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(
            serializable,
            f,
            sort_keys=True,
        )
    return out_path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Run stage1 FBP inference.')
    p.add_argument(
        '--config',
        type=Path,
        default=None,
        help='YAML config path. CLI options override config values.',
    )
    p.add_argument('--in', dest='in_segy_root', type=Path, default=None)
    p.add_argument('--out', dest='out_infer_root', type=Path, default=None)
    p.add_argument('--weights', dest='weights_path', type=Path, default=None)

    for key in _STAGE1_CFG_KEYS:
        if key in {'in_segy_root', 'out_infer_root', 'weights_path'}:
            p.add_argument(
                f'--{key.replace("_", "-")}',
                dest=key,
                type=Path,
                default=None,
                help=argparse.SUPPRESS,
            )
            continue
        if key == 'segy_exts':
            p.add_argument('--segy-exts', dest='segy_exts', type=str, default=None)
            continue

        flag = f'--{key.replace("_", "-")}'
        default_value = getattr(DEFAULT_STAGE1_CFG, key)
        if isinstance(default_value, bool):
            p.add_argument(
                flag,
                dest=key,
                action=argparse.BooleanOptionalAction,
                default=None,
            )
            continue
        if isinstance(default_value, int):
            p.add_argument(flag, dest=key, type=int, default=None)
            continue
        if isinstance(default_value, float):
            p.add_argument(flag, dest=key, type=float, default=None)
            continue
        p.add_argument(flag, dest=key, type=str, default=None)
    return p


def _parse_args() -> argparse.Namespace:
    parser = _build_parser()
    return parse_args_with_yaml_defaults(
        parser,
        load_defaults=_load_yaml_defaults,
    )


def _validate_stage1_cfg(cfg: Stage1Cfg) -> Stage1Cfg:
    segy_exts = normalize_segy_exts(list(cfg.segy_exts))
    cfg = replace(cfg, segy_exts=segy_exts)

    if cfg.device == '':
        raise ValueError('device must not be empty')
    if cfg.viz_dirname == '':
        raise ValueError('viz_dirname must not be empty')
    if cfg.segy_endian not in {'big', 'little'}:
        raise ValueError(
            f"segy_endian must be 'big' or 'little', got {cfg.segy_endian!r}"
        )
    if cfg.waveform_mode not in {'mmap', 'eager'}:
        raise ValueError(
            f"waveform_mode must be 'mmap' or 'eager', got {cfg.waveform_mode!r}"
        )
    if cfg.viz_score_style not in {'bar', 'line'}:
        raise ValueError(
            f"viz_score_style must be 'bar' or 'line', got {cfg.viz_score_style!r}"
        )
    if cfg.rs_base_pick not in {'pre', 'snap'}:
        raise ValueError(
            f"rs_base_pick must be 'pre' or 'snap', got {cfg.rs_base_pick!r}"
        )
    if cfg.rs_mode not in {'diff', 'raw'}:
        raise ValueError(f"rs_mode must be 'diff' or 'raw', got {cfg.rs_mode!r}")
    if cfg.trend_local_weight_mode not in {'uniform', 'pmax'}:
        raise ValueError(
            f"trend_local_weight_mode must be 'uniform' or 'pmax', got {cfg.trend_local_weight_mode!r}"
        )

    if cfg.tile_h <= 0:
        raise ValueError(f'tile_h must be > 0, got {cfg.tile_h}')
    if cfg.tile_w <= 0:
        raise ValueError(f'tile_w must be > 0, got {cfg.tile_w}')
    if cfg.overlap_h < 0:
        raise ValueError(f'overlap_h must be >= 0, got {cfg.overlap_h}')
    if cfg.tiles_per_batch <= 0:
        raise ValueError(f'tiles_per_batch must be > 0, got {cfg.tiles_per_batch}')
    if cfg.viz_every_n_shots < 0:
        raise ValueError(f'viz_every_n_shots must be >= 0, got {cfg.viz_every_n_shots}')
    if cfg.plot_end <= cfg.plot_start:
        raise ValueError(
            f'plot_end must be > plot_start, got start={cfg.plot_start}, end={cfg.plot_end}'
        )
    if cfg.vmax_mask <= 0.0:
        raise ValueError(f'vmax_mask must be > 0, got {cfg.vmax_mask}')
    if cfg.vmin_mask < 0.0:
        raise ValueError(f'vmin_mask must be >= 0, got {cfg.vmin_mask}')
    if cfg.vmax_mask < cfg.vmin_mask:
        raise ValueError(
            f'vmax_mask must be >= vmin_mask, got vmin={cfg.vmin_mask}, vmax={cfg.vmax_mask}'
        )
    if cfg.conf_half_win < 0:
        raise ValueError(f'conf_half_win must be >= 0, got {cfg.conf_half_win}')
    if cfg.trend_min_pts < 0:
        raise ValueError(f'trend_min_pts must be >= 0, got {cfg.trend_min_pts}')
    if cfg.rs_abs_lag_soft < 0.0:
        raise ValueError(f'rs_abs_lag_soft must be >= 0, got {cfg.rs_abs_lag_soft}')

    return cfg


def _cfg_from_namespace(args: argparse.Namespace) -> Stage1Cfg:
    updates: dict[str, object] = {}
    for key in _STAGE1_CFG_KEYS:
        value = getattr(args, key)
        if value is None:
            continue
        if key == 'segy_exts':
            if isinstance(value, tuple):
                updates[key] = tuple(str(x) for x in value)
            else:
                updates[key] = normalize_segy_exts(value)
            continue
        if key in _PATH_KEYS:
            updates[key] = Path(value)
            continue
        updates[key] = value
    cfg = replace(DEFAULT_STAGE1_CFG, **updates)
    return _validate_stage1_cfg(cfg)


def main() -> None:
    args = _parse_args()
    cfg = _cfg_from_namespace(args)
    run_stage1_cfg(cfg, segy_paths=None)


if __name__ == '__main__':
    main()
