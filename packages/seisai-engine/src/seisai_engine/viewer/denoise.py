from __future__ import annotations

import inspect
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from seisai_models.models.encdec2d import EncDec2D
from seisai_transforms.mask_inference import cover_all_traces_predict_striped
from seisai_utils.validator import validate_array

from seisai_engine.infer.segy2segy_cli_common import select_state_dict
from seisai_engine.pipelines.common import build_encdec2d_model, load_checkpoint
from seisai_engine.predict import infer_tiled_bchw, infer_tiled_chw

from .fbpick import _resolve_device

__all__ = ['infer_denoise_hw']

_MAX_CACHE_SIZE = 8
_CACHE_LOCK = threading.Lock()
_MODEL_CACHE: dict[tuple[Path, int, str, str], _ViewerModelBundleDenoise] = {}
_CACHE_ORDER: list[tuple[Path, int, str, str]] = []


@dataclass
class _ViewerModelBundleDenoise:
    model: torch.nn.Module
    in_chans: int
    out_chans: int
    used_ema: bool


def _required_model_sig_keys() -> tuple[str, ...]:
    required: list[str] = []
    for name, param in inspect.signature(EncDec2D.__init__).parameters.items():
        if name == 'self':
            continue
        if (
            param.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            and param.default is inspect.Parameter.empty
        ):
            required.append(name)
    return tuple(required)


_REQUIRED_MODEL_SIG_KEYS = _required_model_sig_keys()


def _require_strict_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f'{name} must be int'
        raise ValueError(msg)
    return int(value)


def _validate_tile(tile: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(tile, tuple) or len(tile) != 2:
        msg = 'tile must be tuple[int, int]'
        raise ValueError(msg)
    tile_h = _require_strict_int(tile[0], name='tile[0]')
    tile_w = _require_strict_int(tile[1], name='tile[1]')
    if tile_h <= 0 or tile_w <= 0:
        msg = f'tile must be positive, got {tile}'
        raise ValueError(msg)
    return tile_h, tile_w


def _validate_overlap(
    overlap: tuple[int, int], *, tile: tuple[int, int]
) -> tuple[int, int]:
    if not isinstance(overlap, tuple) or len(overlap) != 2:
        msg = 'overlap must be tuple[int, int]'
        raise ValueError(msg)
    ov_h = _require_strict_int(overlap[0], name='overlap[0]')
    ov_w = _require_strict_int(overlap[1], name='overlap[1]')
    if ov_h < 0 or ov_w < 0:
        msg = f'overlap must be non-negative, got {overlap}'
        raise ValueError(msg)
    if ov_h >= tile[0] or ov_w >= tile[1]:
        msg = f'overlap must satisfy overlap < tile, got overlap={overlap}, tile={tile}'
        raise ValueError(msg)
    return ov_h, ov_w


def _require_strict_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        msg = f'{name} must be float'
        raise ValueError(msg)
    out = float(value)
    if not np.isfinite(out):
        msg = f'{name} must be finite'
        raise ValueError(msg)
    return out


def _validate_mask_ratio(mask_ratio: float) -> float:
    ratio = _require_strict_float(mask_ratio, name='mask_ratio')
    if ratio == 0.0:
        return 0.0
    if not (0.0 < ratio <= 1.0):
        msg = f'mask_ratio must be 0 or in (0,1], got {mask_ratio}'
        raise ValueError(msg)
    return ratio


def _validate_model_sig(model_sig: object) -> tuple[dict[str, Any], int, int]:
    if not isinstance(model_sig, dict):
        msg = 'checkpoint model_sig must be dict'
        raise ValueError(msg)

    for key in _REQUIRED_MODEL_SIG_KEYS:
        if key not in model_sig:
            msg = f'checkpoint model_sig missing: {key}'
            raise ValueError(msg)
    if 'in_chans' not in model_sig:
        msg = 'checkpoint model_sig missing: in_chans'
        raise ValueError(msg)
    if 'out_chans' not in model_sig:
        msg = 'checkpoint model_sig missing: out_chans'
        raise ValueError(msg)

    in_chans = _require_strict_int(model_sig['in_chans'], name='model_sig.in_chans')
    out_chans = _require_strict_int(model_sig['out_chans'], name='model_sig.out_chans')
    if in_chans != 1:
        msg = f'denoise viewer requires model_sig.in_chans=1, got {in_chans}'
        raise ValueError(msg)
    if out_chans != 1:
        msg = f'denoise viewer requires model_sig.out_chans=1, got {out_chans}'
        raise ValueError(msg)
    return dict(model_sig), in_chans, out_chans


def _select_checkpoint_state_dict(
    ckpt: dict[str, Any], *, use_ema: bool | None
) -> tuple[dict[str, Any], bool]:
    if use_ema is not None and not isinstance(use_ema, bool):
        msg = f'use_ema must be bool or None, got {type(use_ema)}'
        raise ValueError(msg)

    if use_ema is None:
        selected, used_ema = select_state_dict(ckpt)
        if not isinstance(selected, dict):
            msg = 'checkpoint state_dict must be dict'
            raise ValueError(msg)
        if not isinstance(used_ema, bool):
            msg = 'checkpoint infer_used_ema must resolve to bool'
            raise ValueError(msg)
        return selected, used_ema

    if use_ema:
        state_dict = ckpt.get('ema_state_dict')
        if not isinstance(state_dict, dict):
            msg = 'use_ema=True requested but ema_state_dict is missing'
            raise ValueError(msg)
        return state_dict, True

    state_dict = ckpt.get('model_state_dict')
    if not isinstance(state_dict, dict):
        msg = 'checkpoint model_state_dict must be dict'
        raise ValueError(msg)
    return state_dict, False


def _build_model_bundle(
    *,
    ckpt_path: Path,
    device: torch.device,
    use_ema: bool | None,
) -> _ViewerModelBundleDenoise:
    ckpt = load_checkpoint(ckpt_path)
    model_sig, in_chans, out_chans = _validate_model_sig(ckpt.get('model_sig'))

    model_kwargs = dict(model_sig)
    model_kwargs['pretrained'] = False
    model = build_encdec2d_model(model_kwargs)

    state_dict, used_ema = _select_checkpoint_state_dict(ckpt, use_ema=use_ema)
    model.load_state_dict(state_dict, strict=True)
    model.to(device=device)
    model.eval()

    return _ViewerModelBundleDenoise(
        model=model,
        in_chans=in_chans,
        out_chans=out_chans,
        used_ema=used_ema,
    )


def _make_cache_key(
    *,
    ckpt_path: str | Path,
    device: torch.device,
    use_ema: bool | None,
) -> tuple[Path, int, str, str]:
    resolved_ckpt = Path(ckpt_path).resolve()
    if not resolved_ckpt.is_file():
        raise FileNotFoundError(resolved_ckpt)

    device_str = str(device).strip()
    if not device_str:
        msg = 'device must be non-empty'
        raise ValueError(msg)

    use_ema_mode = (
        'auto' if use_ema is None else ('ema' if bool(use_ema) else 'model')
    )
    return resolved_ckpt, int(resolved_ckpt.stat().st_mtime_ns), device_str, use_ema_mode


def _get_model_bundle(
    *,
    ckpt_path: str | Path,
    device: torch.device,
    use_ema: bool | None,
) -> _ViewerModelBundleDenoise:
    key = _make_cache_key(ckpt_path=ckpt_path, device=device, use_ema=use_ema)

    with _CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached

    bundle = _build_model_bundle(ckpt_path=key[0], device=device, use_ema=use_ema)

    with _CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached

        _MODEL_CACHE[key] = bundle
        _CACHE_ORDER.append(key)
        while len(_CACHE_ORDER) > _MAX_CACHE_SIZE:
            old_key = _CACHE_ORDER.pop(0)
            _MODEL_CACHE.pop(old_key, None)
    return bundle


def _pad_chw_to_min_tile(
    x_chw: np.ndarray, tile: tuple[int, int]
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    validate_array(x_chw, allowed_ndims=(3,), name='x_chw', backend='numpy')
    x = np.ascontiguousarray(x_chw, dtype=np.float32)

    _, h, w = x.shape
    tile_h, tile_w = _validate_tile(tile)
    orig_hw = (int(h), int(w))
    target_hw = (max(h, tile_h), max(w, tile_w))
    if target_hw == orig_hw:
        return x, orig_hw, target_hw

    x_pad = np.zeros((x.shape[0], target_hw[0], target_hw[1]), dtype=np.float32)
    x_pad[:, :h, :w] = x
    return x_pad, orig_hw, target_hw


def _crop_hw(arr_hw: np.ndarray, orig_hw: tuple[int, int]) -> np.ndarray:
    validate_array(arr_hw, allowed_ndims=(2,), name='arr_hw', backend='numpy')
    arr = np.ascontiguousarray(arr_hw, dtype=np.float32)
    h, w = int(orig_hw[0]), int(orig_hw[1])
    if h > arr.shape[0] or w > arr.shape[1]:
        msg = f'orig_hw {orig_hw} exceeds array shape {tuple(arr.shape)}'
        raise ValueError(msg)
    return np.ascontiguousarray(arr[:h, :w], dtype=np.float32)


def _infer_tiled(
    model: torch.nn.Module,
    x_chw: np.ndarray,
    *,
    tile: tuple[int, int],
    overlap: tuple[int, int],
    amp: bool,
    tiles_per_batch: int,
) -> torch.Tensor:
    return infer_tiled_chw(
        model,
        x_chw,
        tile=tile,
        overlap=overlap,
        amp=amp,
        use_tqdm=False,
        tiles_per_batch=tiles_per_batch,
    )


def _tile_starts(full: int, tile_size: int, overlap_size: int) -> list[int]:
    stride = tile_size - overlap_size
    if stride <= 0:
        msg = f'invalid stride: tile_size={tile_size}, overlap_size={overlap_size}'
        raise ValueError(msg)
    starts = [0]
    while starts[-1] + tile_size < full:
        nxt = starts[-1] + stride
        if nxt + tile_size >= full:
            starts.append(max(full - tile_size, 0))
            break
        starts.append(nxt)
    return sorted(set(starts))


def _pad_bchw_to_min_tile(
    x_bchw: torch.Tensor,
    *,
    tile: tuple[int, int],
) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
    if not isinstance(x_bchw, torch.Tensor):
        msg = f'x_bchw must be torch.Tensor, got {type(x_bchw)}'
        raise TypeError(msg)
    if x_bchw.ndim != 4:
        msg = f'x_bchw must be (B,C,H,W), got {tuple(x_bchw.shape)}'
        raise ValueError(msg)
    tile_h, tile_w = _validate_tile(tile)
    _, _, h, w = x_bchw.shape
    orig_hw = (int(h), int(w))
    target_hw = (max(int(h), tile_h), max(int(w), tile_w))
    if target_hw == orig_hw:
        return x_bchw.contiguous(), orig_hw, target_hw
    x_pad = torch.zeros(
        (x_bchw.shape[0], x_bchw.shape[1], target_hw[0], target_hw[1]),
        device=x_bchw.device,
        dtype=x_bchw.dtype,
    )
    x_pad[:, :, : int(h), : int(w)] = x_bchw
    return x_pad, orig_hw, target_hw


def _make_chunk_blend_weight(
    chunk_h: int,
    overlap_h: int,
    *,
    has_top: bool,
    has_bottom: bool,
    device: torch.device,
) -> torch.Tensor:
    if chunk_h <= 0:
        msg = f'chunk_h must be positive, got {chunk_h}'
        raise ValueError(msg)
    w_h = torch.ones((chunk_h,), device=device, dtype=torch.float32)
    if overlap_h <= 0:
        return w_h.view(1, 1, chunk_h, 1)
    edge = min(overlap_h, chunk_h)
    hann = torch.hann_window(chunk_h, periodic=False, device=device, dtype=torch.float32)
    hann = torch.clamp(hann, min=1e-3)
    if has_top:
        w_h[:edge] *= hann[:edge]
    if has_bottom:
        w_h[-edge:] *= hann[-edge:]
    return w_h.view(1, 1, chunk_h, 1)


def _infer_cover_chunked(
    model: torch.nn.Module,
    x_chw: np.ndarray,
    *,
    device: torch.device,
    tile: tuple[int, int],
    overlap: tuple[int, int],
    amp: bool,
    tiles_per_batch: int,
    mask_ratio: float,
    noise_std: float,
    mask_noise_mode: Literal['replace', 'add'],
    seed: int | None,
    passes_batch: int,
) -> torch.Tensor:
    validate_array(x_chw, allowed_ndims=(3,), name='x_chw', backend='numpy')
    x = np.ascontiguousarray(x_chw, dtype=np.float32)
    x_bchw = torch.from_numpy(x).unsqueeze(0).to(device=device, dtype=torch.float32)
    if tuple(x_bchw.shape[:2]) != (1, 1):
        msg = f'cover input must be (1,1,H,W), got {tuple(x_bchw.shape)}'
        raise ValueError(msg)

    _, _, h, w = x_bchw.shape
    tile_h, _ = _validate_tile(tile)
    overlap_h, _ = _validate_overlap(overlap, tile=tile)
    starts = _tile_starts(int(h), tile_h, overlap_h)

    y_sum = torch.zeros((1, 1, int(h), int(w)), device=device, dtype=torch.float32)
    weight_sum = torch.zeros((1, 1, int(h), int(w)), device=device, dtype=torch.float32)

    def _predict_fn(xmb: torch.Tensor) -> torch.Tensor:
        return infer_tiled_bchw(
            model,
            xmb,
            tile=tile,
            overlap=overlap,
            amp=amp,
            use_tqdm=False,
            tiles_per_batch=tiles_per_batch,
        )

    rng_devices: list[int] | None = None
    if x_bchw.is_cuda:
        if x_bchw.device.index is None:
            rng_devices = [torch.cuda.current_device()]
        else:
            rng_devices = [int(x_bchw.device.index)]

    with torch.random.fork_rng(devices=rng_devices, enabled=seed is not None):
        if seed is not None:
            torch.manual_seed(seed)

        for idx, start in enumerate(starts):
            stop = min(int(start) + tile_h, int(h))
            x_chunk = x_bchw[:, :, int(start) : int(stop), :]
            x_chunk_pad, chunk_hw, chunk_target = _pad_bchw_to_min_tile(
                x_chunk, tile=tile
            )

            y_chunk_pad = cover_all_traces_predict_striped(
                model,
                x_chunk_pad,
                mask_ratio=mask_ratio,
                band_width=1,
                noise_std=noise_std,
                mask_noise_mode=mask_noise_mode,
                use_amp=amp,
                device=device,
                offsets=(0,),
                passes_batch=passes_batch,
                predict_fn=_predict_fn,
            )
            if tuple(y_chunk_pad.shape) != (
                1,
                1,
                int(chunk_target[0]),
                int(chunk_target[1]),
            ):
                msg = (
                    'cover chunk output shape mismatch: '
                    f'got {tuple(y_chunk_pad.shape)}, expected '
                    f'(1,1,{int(chunk_target[0])},{int(chunk_target[1])})'
                )
                raise ValueError(msg)

            y_chunk = y_chunk_pad[:, :, : int(chunk_hw[0]), : int(chunk_hw[1])].to(
                torch.float32
            )
            if tuple(y_chunk.shape) != (1, 1, int(stop - start), int(w)):
                msg = (
                    'cover chunk crop shape mismatch: '
                    f'got {tuple(y_chunk.shape)}, expected '
                    f'(1,1,{int(stop - start)},{int(w)})'
                )
                raise ValueError(msg)

            w_chunk = _make_chunk_blend_weight(
                int(stop - start),
                overlap_h,
                has_top=(idx > 0),
                has_bottom=(idx < (len(starts) - 1)),
                device=device,
            )
            y_sum[:, :, int(start) : int(stop), :] += y_chunk * w_chunk
            weight_sum[:, :, int(start) : int(stop), :] += w_chunk

    if not torch.all(weight_sum > 0):
        msg = 'cover chunk blending produced zero-weight pixels'
        raise RuntimeError(msg)
    return y_sum / weight_sum


def _to_float32_numpy(arr: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(arr, torch.Tensor):
        np_arr = arr.detach().cpu().numpy()
    elif isinstance(arr, np.ndarray):
        np_arr = arr
    else:
        msg = f'infer output must be torch.Tensor or numpy.ndarray, got {type(arr)}'
        raise TypeError(msg)
    return np.ascontiguousarray(np_arr, dtype=np.float32)


@torch.no_grad()
def infer_denoise_hw(
    section_hw: np.ndarray,
    *,
    ckpt_path: Path,
    device: str | torch.device = 'auto',
    tile: tuple[int, int] = (128, 6016),
    overlap: tuple[int, int] = (32, 1024),
    amp: bool = True,
    tiles_per_batch: int = 8,
    use_ema: bool | None = None,
    mask_ratio: float = 0.0,
    noise_std: float = 1.0,
    mask_noise_mode: Literal['replace', 'add'] = 'replace',
    seed: int | None = 12345,
    passes_batch: int = 4,
) -> np.ndarray:
    validate_array(section_hw, allowed_ndims=(2,), name='section_hw', backend='numpy')
    section = np.ascontiguousarray(section_hw, dtype=np.float32)
    orig_hw = (int(section.shape[0]), int(section.shape[1]))
    if orig_hw[0] <= 0 or orig_hw[1] <= 0:
        msg = f'section_hw must have positive shape, got {orig_hw}'
        raise ValueError(msg)

    tile_hw = _validate_tile(tile)
    overlap_hw = _validate_overlap(overlap, tile=tile_hw)
    if use_ema is not None and not isinstance(use_ema, bool):
        msg = f'use_ema must be bool or None, got {type(use_ema)}'
        raise ValueError(msg)
    tiles_batch = _require_strict_int(tiles_per_batch, name='tiles_per_batch')
    if tiles_batch <= 0:
        msg = f'tiles_per_batch must be > 0, got {tiles_per_batch}'
        raise ValueError(msg)
    mask_ratio_value = _validate_mask_ratio(mask_ratio)
    noise_std_value = _require_strict_float(noise_std, name='noise_std')
    if noise_std_value < 0.0:
        msg = f'noise_std must be >= 0, got {noise_std}'
        raise ValueError(msg)
    if mask_noise_mode not in ('replace', 'add'):
        msg = f'mask_noise_mode must be "replace" or "add", got {mask_noise_mode}'
        raise ValueError(msg)
    seed_value = None
    if seed is not None:
        seed_value = _require_strict_int(seed, name='seed')
    passes_batch_value = _require_strict_int(passes_batch, name='passes_batch')
    if passes_batch_value <= 0:
        msg = f'passes_batch must be > 0, got {passes_batch}'
        raise ValueError(msg)

    resolved_device = _resolve_device(device)
    bundle = _get_model_bundle(
        ckpt_path=ckpt_path,
        device=resolved_device,
        use_ema=use_ema,
    )
    if bundle.in_chans != 1:
        msg = f'denoise viewer expects in_chans=1, got {bundle.in_chans}'
        raise ValueError(msg)
    if bundle.out_chans != 1:
        msg = f'denoise viewer expects out_chans=1, got {bundle.out_chans}'
        raise ValueError(msg)

    x_chw = np.ascontiguousarray(section[None, :, :], dtype=np.float32)
    x_pad_chw, x_orig_hw, target_hw = _pad_chw_to_min_tile(x_chw, tile=tile_hw)
    if x_orig_hw != orig_hw:
        msg = f'input shape mismatch: x_orig_hw={x_orig_hw}, orig_hw={orig_hw}'
        raise ValueError(msg)

    if mask_ratio_value == 0.0:
        y_pred = _infer_tiled(
            bundle.model,
            x_pad_chw,
            tile=tile_hw,
            overlap=overlap_hw,
            amp=bool(amp),
            tiles_per_batch=tiles_batch,
        )
    else:
        y_pred = _infer_cover_chunked(
            bundle.model,
            x_pad_chw,
            device=resolved_device,
            tile=tile_hw,
            overlap=overlap_hw,
            amp=bool(amp),
            tiles_per_batch=tiles_batch,
            mask_ratio=mask_ratio_value,
            noise_std=noise_std_value,
            mask_noise_mode=mask_noise_mode,
            seed=seed_value,
            passes_batch=passes_batch_value,
        ).squeeze(0)
    y_chw = _to_float32_numpy(y_pred)
    if y_chw.ndim != 3:
        msg = f'infer output must be (C,H,W), got {tuple(y_chw.shape)}'
        raise ValueError(msg)
    if int(y_chw.shape[0]) != bundle.out_chans:
        msg = (
            f'infer output channel dim {int(y_chw.shape[0])} '
            f'!= out_chans {bundle.out_chans}'
        )
        raise ValueError(msg)
    if tuple(y_chw.shape[1:]) != target_hw:
        msg = (
            f'infer output spatial shape {tuple(y_chw.shape[1:])} '
            f'!= padded shape {target_hw}'
        )
        raise ValueError(msg)

    out_hw = _crop_hw(y_chw[0], orig_hw=orig_hw)
    if tuple(out_hw.shape) != orig_hw:
        msg = f'output shape {tuple(out_hw.shape)} != input shape {orig_hw}'
        raise ValueError(msg)
    return np.ascontiguousarray(out_hw, dtype=np.float32)
