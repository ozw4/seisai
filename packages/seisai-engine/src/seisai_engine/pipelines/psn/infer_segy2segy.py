from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from functools import partial
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from seisai_dataset.ffid_gather_iter import FFIDGatherIterator
from seisai_transforms.augment import PerTraceStandardize
from seisai_utils.config import (
    optional_str,
    require_dict,
)
from seisai_utils.segy_write import write_segy_float32_like_input_with_text0_append
from seisai_utils.validator import require_positive_float as _require_positive_float

from seisai_engine.infer.ckpt_meta import resolve_output_ids, resolve_softmax_axis
from seisai_engine.infer.ffid_segy2segy import (
    Tiled2DConfig,
    run_ffid_gather_infer_core_chw,
)
from seisai_engine.infer.segy2segy_infer_common import parse_infer_common
from seisai_engine.infer.segy2segy_cli_common import (
    apply_unknown_overrides as _apply_unknown_overrides,
)
from seisai_engine.infer.segy2segy_cli_common import (
    build_merged_cfg_with_ckpt_cfg,
)
from seisai_engine.infer.segy2segy_cli_common import cfg_hash as _cfg_hash
from seisai_engine.infer.segy2segy_cli_common import is_strict_int as _is_strict_int
from seisai_engine.infer.segy2segy_cli_common import merge_with_precedence
from seisai_engine.infer.segy2segy_cli_common import (
    resolve_ckpt_path as _resolve_ckpt_path,
)
from seisai_engine.infer.segy2segy_cli_common import (
    select_state_dict as _select_state_dict,
)
from seisai_engine.infer.segy2segy_cli_common import sig_hash as _sig_hash
from seisai_engine.pipelines.common import (
    build_encdec2d_model,
    load_cfg_with_base_dir,
    load_checkpoint,
    resolve_device,
)
from seisai_engine.predict import _run_tiled

__all__ = [
    'apply_unknown_overrides',
    'main',
    'merge_with_precedence',
    'run_infer_and_write',
]

DEFAULT_CONFIG_PATH = Path('examples/config_infer_psn.yaml')
PIPELINE_NAME = 'psn'
_SAFE_OVERRIDE_PATHS = frozenset(
    {
        'paths.segy_files',
        'paths.out_dir',
        'infer.ckpt_path',
        'infer.device',
        'infer.out_suffix',
        'infer.overwrite',
        'infer.sort_within',
        'infer.ffids',
        'infer.outputs',
        'infer.standardize_eps',
        'infer.note',
        'infer.allow_unsafe_override',
        'tile.tile_h',
        'tile.overlap_h',
        'tile.tile_w',
        'tile.overlap_w',
        'tile.tiles_per_batch',
        'tile.amp',
        'tile.use_tqdm',
        'tta',
    }
)


def _default_cfg() -> dict[str, Any]:
    return {
        'paths': {
            'segy_files': [],
            'out_dir': './_psn_infer_out',
        },
        'infer': {
            'ckpt_path': '',
            'device': 'auto',
            'out_suffix': '.sgy',
            'overwrite': False,
            'sort_within': 'chno',
            'ffids': None,
            'outputs': None,
            'standardize_eps': None,
            'allow_unsafe_override': False,
            'note': '',
        },
        'tile': {
            'tile_h': 128,
            'overlap_h': 64,
            'tile_w': 6016,
            'overlap_w': 1024,
            'tiles_per_batch': 16,
            'amp': True,
            'use_tqdm': False,
        },
        'tta': [],
    }


apply_unknown_overrides = partial(
    _apply_unknown_overrides,
    safe_paths=_SAFE_OVERRIDE_PATHS,
)


def _resolve_selected_outputs(
    *,
    infer_cfg: dict[str, Any],
    output_ids: tuple[str, ...],
) -> tuple[str, ...]:
    outputs_raw = infer_cfg.get('outputs')
    if outputs_raw is None:
        return tuple(output_ids)
    if not isinstance(outputs_raw, list):
        msg = 'infer.outputs must be list[str] or null'
        raise TypeError(msg)
    if len(outputs_raw) == 0:
        msg = 'infer.outputs must be non-empty when provided'
        raise ValueError(msg)

    selected: list[str] = []
    for idx, item in enumerate(outputs_raw):
        if not isinstance(item, str) or len(item.strip()) == 0:
            msg = f'infer.outputs[{idx}] must be non-empty str'
            raise TypeError(msg)
        token = item.strip()
        if token not in output_ids:
            msg = (
                f'infer.outputs[{idx}]={token} '
                f'not found in output_ids={list(output_ids)}'
            )
            raise ValueError(msg)
        selected.append(token)
    if len(set(selected)) != len(selected):
        msg = 'infer.outputs must not contain duplicates'
        raise ValueError(msg)
    return tuple(selected)


def _resolve_standardize_eps(cfg: dict[str, Any]) -> float:
    infer_cfg = require_dict(cfg, 'infer')
    eps_raw = infer_cfg.get('standardize_eps')
    if eps_raw is not None:
        return _require_positive_float(eps_raw, name='infer.standardize_eps')

    transform_cfg = cfg.get('transform')
    if transform_cfg is not None:
        if not isinstance(transform_cfg, dict):
            msg = 'transform must be dict'
            raise TypeError(msg)
        if 'standardize_eps' in transform_cfg:
            return _require_positive_float(
                transform_cfg['standardize_eps'],
                name='transform.standardize_eps',
            )

    return 1.0e-8


def _resolve_tta_requested(cfg: dict[str, Any]) -> list[Any]:
    tta_obj = cfg.get('tta', [])
    if tta_obj is None:
        return []
    if not isinstance(tta_obj, list):
        msg = 'tta must be list or null'
        raise TypeError(msg)
    return list(tta_obj)


def _build_model_from_ckpt(
    *,
    ckpt: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], bool, tuple[str, ...], str]:
    model_sig = ckpt['model_sig']
    if not isinstance(model_sig, dict):
        msg = 'checkpoint model_sig must be dict'
        raise TypeError(msg)
    if 'in_chans' not in model_sig or 'out_chans' not in model_sig:
        msg = 'checkpoint model_sig must contain in_chans and out_chans'
        raise KeyError(msg)

    in_chans = model_sig['in_chans']
    out_chans = model_sig['out_chans']
    if not _is_strict_int(in_chans):
        msg = 'checkpoint model_sig.in_chans must be int'
        raise TypeError(msg)
    if not _is_strict_int(out_chans):
        msg = 'checkpoint model_sig.out_chans must be int'
        raise TypeError(msg)
    if int(in_chans) != 1:
        msg = f'psn infer requires model_sig.in_chans=1, got {int(in_chans)}'
        raise ValueError(msg)
    if int(out_chans) != 3:
        msg = f'psn infer requires model_sig.out_chans=3, got {int(out_chans)}'
        raise ValueError(msg)

    output_ids = resolve_output_ids(
        ckpt=ckpt,
        out_chans=int(out_chans),
        pipeline_name=PIPELINE_NAME,
    )
    softmax_axis = resolve_softmax_axis(
        ckpt=ckpt,
        out_chans=int(out_chans),
        pipeline_name=PIPELINE_NAME,
    )

    model_kwargs = dict(model_sig)
    model_kwargs['pretrained'] = False
    model = build_encdec2d_model(model_kwargs)

    state_dict, used_ema = _select_state_dict(ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.to(device=device)
    model.eval()
    return model, dict(model_sig), used_ema, output_ids, softmax_axis


def _build_text0_lines(
    *,
    output_id: str,
    ckpt_path: Path,
    epoch: int,
    global_step: int,
    model_sig_hash: str,
    cfg_hash: str,
    ckpt_cfg_hash: str,
    softmax_axis: str,
    tile_cfg: Tiled2DConfig,
    inferred_at_utc: str,
    seisai_ver: str,
    note: str,
) -> list[str]:
    line1 = (
        f'ML pipeline=psn out={output_id} ckpt={ckpt_path.name} '
        f'ep={int(epoch)} gs={int(global_step)}'
    )
    line2 = (
        f'ML modelsig={model_sig_hash} cfgh={cfg_hash} '
        f'ckcfgh={ckpt_cfg_hash} smx={softmax_axis}'
    )
    line3 = (
        f'ML TILE H={int(tile_cfg.tile_h)}/{int(tile_cfg.overlap_h)} '
        f'W={int(tile_cfg.tile_w)}/{int(tile_cfg.overlap_w)} '
        f'AMP={int(bool(tile_cfg.amp))} TPB={int(tile_cfg.tiles_per_batch)}'
    )
    note_txt = note.strip() if isinstance(note, str) else ''
    if not note_txt:
        note_txt = 'none'
    line4 = f'ML UTC={inferred_at_utc} VER={seisai_ver} NOTE={note_txt}'
    return [line1, line2, line3, line4]


def _build_output_path(
    *,
    src_path: Path,
    out_dir: Path,
    out_suffix: str,
    output_id: str,
) -> Path:
    suffix = str(out_suffix)
    if not suffix:
        msg = 'infer.out_suffix must be non-empty'
        raise ValueError(msg)
    filename = f'{src_path.stem}.{PIPELINE_NAME}_{output_id}{suffix}'
    return out_dir / filename


def _write_sidecar_json(
    *,
    out_path: Path,
    src_path: Path,
    output_id: str,
    output_index: int,
    output_ids: tuple[str, ...],
    softmax_axis: str,
    ckpt_path: Path,
    ckpt_epoch: int,
    ckpt_global_step: int,
    infer_used_ema: bool,
    model_sig: dict[str, Any],
    model_sig_hash: str,
    cfg_hash: str,
    ckpt_cfg_hash: str,
    tile_cfg: Tiled2DConfig,
    standardize_eps: float,
    tta_requested: list[Any],
    inferred_at_utc: str,
    seisai_ver: str,
    note: str,
) -> Path:
    sidecar_path = out_path.with_suffix(out_path.suffix + '.mlmeta.json')
    payload = {
        'pipeline': PIPELINE_NAME,
        'source_segy': str(src_path),
        'output_segy': str(out_path),
        'output_id': str(output_id),
        'output_index': int(output_index),
        'output_ids': list(output_ids),
        'softmax_axis': str(softmax_axis),
        'ckpt_path': str(ckpt_path),
        'ckpt_epoch': int(ckpt_epoch),
        'ckpt_global_step': int(ckpt_global_step),
        'infer_used_ema': bool(infer_used_ema),
        'model_sig': model_sig,
        'model_sig_hash': model_sig_hash,
        'cfg_hash': cfg_hash,
        'ckpt_cfg_hash': ckpt_cfg_hash,
        'tile': {
            'tile_h': int(tile_cfg.tile_h),
            'overlap_h': int(tile_cfg.overlap_h),
            'tile_w': int(tile_cfg.tile_w),
            'overlap_w': int(tile_cfg.overlap_w),
            'tiles_per_batch': int(tile_cfg.tiles_per_batch),
            'amp': bool(tile_cfg.amp),
            'use_tqdm': bool(tile_cfg.use_tqdm),
        },
        'standardize_eps': float(standardize_eps),
        'tta_requested': tta_requested,
        'tta_applied': [],
        'datetime_utc': inferred_at_utc,
        'seisai_version': seisai_ver,
        'note': str(note),
    }
    sidecar_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
        + '\n',
        encoding='utf-8',
    )
    return sidecar_path


def run_infer_and_write(
    *,
    cfg: dict[str, Any],
    base_dir: Path,
) -> list[Path]:
    common = parse_infer_common(
        cfg=cfg,
        base_dir=base_dir,
        default_out_dir='./_psn_infer_out',
        default_out_suffix='.sgy',
    )
    segy_files = common.segy_files
    out_dir = common.out_dir
    overwrite = common.overwrite
    out_suffix = common.out_suffix
    sort_within = common.sort_within
    ffids = common.ffids
    tile_cfg = common.tiled2d
    infer_cfg = require_dict(cfg, 'infer')

    tta_requested = _resolve_tta_requested(cfg)
    standardize_eps = _resolve_standardize_eps(cfg)

    device = resolve_device(optional_str(infer_cfg, 'device', 'auto'))
    ckpt_path = _resolve_ckpt_path(cfg, base_dir=base_dir)
    ckpt = load_checkpoint(ckpt_path)
    if ckpt['pipeline'] != PIPELINE_NAME:
        msg = f'checkpoint pipeline must be "{PIPELINE_NAME}"'
        raise ValueError(msg)
    ckpt_cfg = ckpt.get('cfg')
    if not isinstance(ckpt_cfg, dict):
        msg = 'checkpoint must contain dict cfg'
        raise TypeError(msg)

    model, model_sig, used_ema, output_ids, softmax_axis = _build_model_from_ckpt(
        ckpt=ckpt,
        device=device,
    )
    if softmax_axis != 'channel':
        msg = f'psn infer requires softmax_axis="channel", got "{softmax_axis}"'
        raise ValueError(msg)
    selected_outputs = _resolve_selected_outputs(
        infer_cfg=infer_cfg,
        output_ids=output_ids,
    )

    output_id_to_chan = {name: idx for idx, name in enumerate(output_ids)}
    if len(output_id_to_chan) != len(output_ids):
        msg = 'output_ids must be unique'
        raise ValueError(msg)

    out_chans = len(output_ids)
    tile_h_cfg = int(tile_cfg.tile_h)
    tile_w_cfg = int(tile_cfg.tile_w)
    overlap_h_cfg = int(tile_cfg.overlap_h)
    overlap_w_cfg = int(tile_cfg.overlap_w)
    tile_amp = bool(tile_cfg.amp)
    tile_use_tqdm = bool(tile_cfg.use_tqdm)
    tiles_per_batch = int(tile_cfg.tiles_per_batch)
    standardize_eps_value = float(standardize_eps)
    overwrite_flag = bool(overwrite)

    cfg_hash = _cfg_hash(cfg)
    ckpt_cfg_hash = _cfg_hash(ckpt_cfg)
    sig_hash = _sig_hash(model_sig)
    note = optional_str(infer_cfg, 'note', '')
    seisai_ver = package_version('seisai-engine')
    ckpt_epoch = int(ckpt['epoch'])
    ckpt_global_step = int(ckpt['global_step'])

    out_paths: list[Path] = []
    with FFIDGatherIterator(segy_files, sort_within=sort_within) as iterator:
        for file_index, src in enumerate(iterator.segy_files):
            src_path = Path(src)
            standardize = PerTraceStandardize(eps=standardize_eps_value)

            def infer_one_gather(gather: Any) -> np.ndarray:
                x_hw = np.asarray(gather.x_hw, dtype=np.float32)
                if x_hw.ndim != 2:
                    msg = f'gather x_hw must be 2D, got {x_hw.shape}'
                    raise ValueError(msg)
                h = int(x_hw.shape[0])
                w = int(x_hw.shape[1])
                if h <= 0 or w <= 0:
                    msg = f'invalid gather shape: {x_hw.shape}'
                    raise ValueError(msg)

                x_std_hw = standardize(x_hw)
                if not isinstance(x_std_hw, np.ndarray):
                    msg = 'PerTraceStandardize must return numpy.ndarray'
                    raise TypeError(msg)
                x_std_hw = np.asarray(x_std_hw, dtype=np.float32)
                if tuple(x_std_hw.shape) != (h, w):
                    msg = (
                        'PerTraceStandardize shape mismatch: '
                        f'got {tuple(x_std_hw.shape)}, want {(h, w)}'
                    )
                    raise ValueError(msg)

                x_bchw = torch.from_numpy(x_std_hw[None, None, :, :]).to(
                    device=device, dtype=torch.float32
                )

                tile_h = min(tile_h_cfg, h)
                tile_w = min(tile_w_cfg, w)
                overlap_h = overlap_h_cfg
                overlap_w = overlap_w_cfg
                overlap_h = 0 if tile_h == 1 else min(overlap_h, tile_h - 1)
                overlap_w = 0 if tile_w == 1 else min(overlap_w, tile_w - 1)

                y_bchw = _run_tiled(
                    model,
                    x_bchw,
                    tile=(tile_h, tile_w),
                    overlap=(overlap_h, overlap_w),
                    amp=tile_amp,
                    use_tqdm=tile_use_tqdm,
                    tiles_per_batch=tiles_per_batch,
                )
                expected_logits = (1, out_chans, h, w)
                if tuple(y_bchw.shape) != expected_logits:
                    msg = (
                        f'predict output shape must be {expected_logits}, '
                        f'got {tuple(y_bchw.shape)}'
                    )
                    raise ValueError(msg)

                y_prob = torch.softmax(y_bchw.to(dtype=torch.float32), dim=1)
                y_chw = y_prob[0].detach().cpu().numpy().astype(np.float32, copy=False)
                expected_chw = (out_chans, h, w)
                if tuple(y_chw.shape) != expected_chw:
                    msg = (
                        f'prob output shape must be {expected_chw}, '
                        f'got {tuple(y_chw.shape)}'
                    )
                    raise ValueError(msg)
                return y_chw

            out_chw = run_ffid_gather_infer_core_chw(
                iterator=iterator,
                file_index=file_index,
                out_chans=out_chans,
                infer_one_gather_fn=infer_one_gather,
                ffids=ffids,
            )
            inferred_at = datetime.now(timezone.utc).isoformat()

            for output_id in selected_outputs:
                ch_idx = output_id_to_chan[output_id]
                out_hw = np.asarray(out_chw[ch_idx], dtype=np.float32)
                out_path = _build_output_path(
                    src_path=src_path,
                    out_dir=out_dir,
                    out_suffix=out_suffix,
                    output_id=output_id,
                )
                text0_lines = _build_text0_lines(
                    output_id=output_id,
                    ckpt_path=ckpt_path,
                    epoch=ckpt_epoch,
                    global_step=ckpt_global_step,
                    model_sig_hash=sig_hash,
                    cfg_hash=cfg_hash,
                    ckpt_cfg_hash=ckpt_cfg_hash,
                    softmax_axis=softmax_axis,
                    tile_cfg=tile_cfg,
                    inferred_at_utc=inferred_at,
                    seisai_ver=seisai_ver,
                    note=note,
                )
                write_segy_float32_like_input_with_text0_append(
                    src_path=src_path,
                    dst_path=out_path,
                    data_hw_float32=out_hw,
                    text0_append_lines=text0_lines,
                    overwrite=overwrite_flag,
                )
                _write_sidecar_json(
                    out_path=out_path,
                    src_path=src_path,
                    output_id=output_id,
                    output_index=int(ch_idx),
                    output_ids=output_ids,
                    softmax_axis=softmax_axis,
                    ckpt_path=ckpt_path,
                    ckpt_epoch=ckpt_epoch,
                    ckpt_global_step=ckpt_global_step,
                    infer_used_ema=used_ema,
                    model_sig=model_sig,
                    model_sig_hash=sig_hash,
                    cfg_hash=cfg_hash,
                    ckpt_cfg_hash=ckpt_cfg_hash,
                    tile_cfg=tile_cfg,
                    standardize_eps=standardize_eps,
                    tta_requested=tta_requested,
                    inferred_at_utc=inferred_at,
                    seisai_ver=seisai_ver,
                    note=note,
                )
                out_paths.append(out_path)

    return out_paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(DEFAULT_CONFIG_PATH))
    args, unknown = parser.parse_known_args(argv)

    infer_yaml_cfg, base_dir = load_cfg_with_base_dir(Path(args.config))
    merged_cfg = build_merged_cfg_with_ckpt_cfg(
        infer_yaml_cfg=infer_yaml_cfg,
        base_dir=base_dir,
        unknown_overrides=unknown,
        default_cfg=_default_cfg(),
        safe_paths=_SAFE_OVERRIDE_PATHS,
    )
    out_paths = run_infer_and_write(cfg=merged_cfg, base_dir=base_dir)
    for path in out_paths:
        print(str(path))


if __name__ == '__main__':
    main()
