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
from seisai_dataset.ffid_gather_iter import FFIDGatherIterator, SortWithinGather
from seisai_utils.config import (
    optional_bool,
    optional_str,
    require_dict,
    require_int,
    require_list_str,
)
from seisai_utils.segy_write import write_segy_float32_like_input_with_text0_append

from seisai_engine.infer.ffid_segy2segy import (
    Tiled2DConfig,
    _infer_hw_denorm_like_input,
    _validate_tiled2d_cfg,
    run_ffid_gather_infer_core,
)
from seisai_engine.infer.segy2segy_cli_common import (
    apply_unknown_overrides as _apply_unknown_overrides,
)
from seisai_engine.infer.segy2segy_cli_common import (
    build_merged_cfg as _build_merged_cfg_common,
)
from seisai_engine.infer.segy2segy_cli_common import cfg_hash as _cfg_hash
from seisai_engine.infer.segy2segy_cli_common import is_strict_int as _is_strict_int
from seisai_engine.infer.segy2segy_cli_common import merge_with_precedence
from seisai_engine.infer.segy2segy_cli_common import (
    resolve_ckpt_path as _resolve_ckpt_path,
)
from seisai_engine.infer.segy2segy_cli_common import (
    resolve_segy_files as _resolve_segy_files,
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
    resolve_relpath,
    validate_files_exist,
)

__all__ = [
    'apply_unknown_overrides',
    'main',
    'merge_with_precedence',
    'run_infer_and_write',
]


DEFAULT_CONFIG_PATH = Path('examples/config_infer_pair.yaml')
PIPELINE_NAME = 'pair'
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
        'infer.standardize_eps',
        'infer.allow_unsafe_override',
        'infer.note',
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
            'out_dir': './_pair_infer_out',
        },
        'infer': {
            'ckpt_path': '',
            'device': 'auto',
            'out_suffix': '_pred.sgy',
            'overwrite': False,
            'sort_within': 'chno',
            'ffids': None,
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


def _require_positive_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f'{name} must be float'
        raise TypeError(msg)
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        msg = f'{name} must be finite and > 0'
        raise ValueError(msg)
    return out


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
) -> tuple[torch.nn.Module, dict[str, Any], bool]:
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
        msg = f'pair infer requires model_sig.in_chans=1, got {int(in_chans)}'
        raise ValueError(msg)
    if int(out_chans) != 1:
        msg = f'pair infer requires model_sig.out_chans=1, got {int(out_chans)}'
        raise ValueError(msg)

    model_kwargs = dict(model_sig)
    model_kwargs['pretrained'] = False
    model = build_encdec2d_model(model_kwargs)

    state_dict, used_ema = _select_state_dict(ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.to(device=device)
    model.eval()
    return model, dict(model_sig), used_ema


def _build_text0_lines(
    *,
    ckpt_path: Path,
    epoch: int,
    global_step: int,
    inferred_at_utc: str,
    seisai_ver: str,
    cfg_hash: str,
    ckpt_cfg_hash: str,
    model_sig_hash: str,
    tile_cfg: Tiled2DConfig,
    used_ema: bool,
    note: str,
) -> list[str]:
    line1 = (
        f'ML pipeline=pair ckpt={ckpt_path.name} '
        f'ep={int(epoch)} gs={int(global_step)}'
    )
    line2 = (
        f'ML modelsig={model_sig_hash} cfgh={cfg_hash} '
        f'ckcfgh={ckpt_cfg_hash} ema={int(bool(used_ema))}'
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


def _build_output_path(*, src_path: Path, out_dir: Path, out_suffix: str) -> Path:
    suffix = str(out_suffix)
    if not suffix:
        msg = 'infer.out_suffix must be non-empty'
        raise ValueError(msg)
    filename = f'{src_path.stem}.{PIPELINE_NAME}{suffix}'
    return out_dir / filename


def _write_sidecar_json(
    *,
    out_path: Path,
    src_path: Path,
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
    paths_cfg = require_dict(cfg, 'paths')
    infer_cfg = require_dict(cfg, 'infer')
    tile_cfg_obj = require_dict(cfg, 'tile')

    segy_files = _resolve_segy_files(
        base_dir=base_dir,
        segy_files=require_list_str(paths_cfg, 'segy_files'),
    )
    validate_files_exist(segy_files)

    out_dir_raw = optional_str(paths_cfg, 'out_dir', './_pair_infer_out')
    out_dir = Path(resolve_relpath(base_dir, out_dir_raw))
    out_dir.mkdir(parents=True, exist_ok=True)

    overwrite = optional_bool(infer_cfg, 'overwrite', default=False)
    out_suffix = optional_str(infer_cfg, 'out_suffix', '_pred.sgy')
    sort_within_raw = optional_str(infer_cfg, 'sort_within', 'chno').lower()
    if sort_within_raw not in ('none', 'chno', 'offset'):
        msg = 'infer.sort_within must be one of: none, chno, offset'
        raise ValueError(msg)
    sort_within: SortWithinGather = sort_within_raw  # type: ignore[assignment]

    ffids_value = infer_cfg.get('ffids', None)
    ffids: list[int] | None
    if ffids_value is None:
        ffids = None
    else:
        if not isinstance(ffids_value, list):
            msg = 'infer.ffids must be list[int] or null'
            raise TypeError(msg)
        ffids = []
        for idx, item in enumerate(ffids_value):
            if not _is_strict_int(item):
                msg = f'infer.ffids[{idx}] must be int'
                raise TypeError(msg)
            ffids.append(int(item))
        if len(ffids) == 0:
            msg = 'infer.ffids must be non-empty when provided'
            raise ValueError(msg)

    tile_cfg = Tiled2DConfig(
        tile_h=require_int(tile_cfg_obj, 'tile_h'),
        overlap_h=require_int(tile_cfg_obj, 'overlap_h'),
        tile_w=require_int(tile_cfg_obj, 'tile_w'),
        overlap_w=require_int(tile_cfg_obj, 'overlap_w'),
        tiles_per_batch=require_int(tile_cfg_obj, 'tiles_per_batch'),
        amp=optional_bool(tile_cfg_obj, 'amp', default=True),
        use_tqdm=optional_bool(tile_cfg_obj, 'use_tqdm', default=False),
    )
    _validate_tiled2d_cfg(tile_cfg)
    standardize_eps = _resolve_standardize_eps(cfg)
    tta_requested = _resolve_tta_requested(cfg)

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

    model, model_sig, used_ema = _build_model_from_ckpt(ckpt=ckpt, device=device)

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

            def infer_one_gather(gather: Any) -> np.ndarray:
                x_hw = np.asarray(gather.x_hw, dtype=np.float32)
                return _infer_hw_denorm_like_input(
                    model,
                    x_hw,
                    device=device,
                    cfg=tile_cfg,
                    eps_std=float(standardize_eps),
                )

            out_hw = run_ffid_gather_infer_core(
                iterator=iterator,
                file_index=file_index,
                infer_one_gather_fn=infer_one_gather,
                ffids=ffids,
            )

            out_path = _build_output_path(
                src_path=src_path,
                out_dir=out_dir,
                out_suffix=out_suffix,
            )
            inferred_at = datetime.now(timezone.utc).isoformat()
            text0_lines = _build_text0_lines(
                ckpt_path=ckpt_path,
                epoch=ckpt_epoch,
                global_step=ckpt_global_step,
                inferred_at_utc=inferred_at,
                seisai_ver=seisai_ver,
                cfg_hash=cfg_hash,
                ckpt_cfg_hash=ckpt_cfg_hash,
                model_sig_hash=sig_hash,
                tile_cfg=tile_cfg,
                used_ema=used_ema,
                note=note,
            )
            write_segy_float32_like_input_with_text0_append(
                src_path=src_path,
                dst_path=out_path,
                data_hw_float32=out_hw,
                text0_append_lines=text0_lines,
                overwrite=bool(overwrite),
            )
            _write_sidecar_json(
                out_path=out_path,
                src_path=src_path,
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


def _load_ckpt_cfg_for_merge(
    *,
    base_dir: Path,
    infer_cfg_for_ckpt: dict[str, Any],
) -> dict[str, Any]:
    ckpt_path = _resolve_ckpt_path(infer_cfg_for_ckpt, base_dir=base_dir)
    ckpt = load_checkpoint(ckpt_path)
    cfg_from_ckpt = ckpt.get('cfg')
    if not isinstance(cfg_from_ckpt, dict):
        msg = 'checkpoint must contain dict cfg'
        raise TypeError(msg)
    return cfg_from_ckpt


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(DEFAULT_CONFIG_PATH))
    args, unknown = parser.parse_known_args(argv)

    infer_yaml_cfg, base_dir = load_cfg_with_base_dir(Path(args.config))
    merged_cfg = _build_merged_cfg_common(
        infer_yaml_cfg=infer_yaml_cfg,
        base_dir=base_dir,
        unknown_overrides=unknown,
        default_cfg=_default_cfg(),
        safe_paths=_SAFE_OVERRIDE_PATHS,
        ckpt_cfg_loader=lambda infer_cfg_for_ckpt, local_base_dir: _load_ckpt_cfg_for_merge(
            base_dir=local_base_dir,
            infer_cfg_for_ckpt=infer_cfg_for_ckpt,
        ),
    )
    out_paths = run_infer_and_write(cfg=merged_cfg, base_dir=base_dir)
    for path in out_paths:
        print(str(path))


if __name__ == '__main__':
    main()
