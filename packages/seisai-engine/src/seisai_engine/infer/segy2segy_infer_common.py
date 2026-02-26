from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from seisai_dataset.ffid_gather_iter import SortWithinGather
from seisai_utils.config import (
    optional_bool,
    optional_str,
    require_dict,
    require_int,
    require_list_str,
)
from seisai_utils.fs import validate_files_exist

from seisai_engine.infer.ffid_segy2segy import Tiled2DConfig, _validate_tiled2d_cfg
from seisai_engine.infer.segy2segy_cli_common import (
    is_strict_int,
    resolve_segy_files,
)
from seisai_engine.pipelines.common import resolve_relpath

__all__ = [
    'InferCommonParsed',
    'parse_infer_common',
]


@dataclass(frozen=True)
class InferCommonParsed:
    segy_files: list[str]
    out_dir: Path
    overwrite: bool
    out_suffix: str
    sort_within: SortWithinGather
    ffids: list[int] | None
    tiled2d: Tiled2DConfig


def parse_infer_common(
    *,
    cfg: dict[str, Any],
    base_dir: Path,
    default_out_dir: str = './_infer_out',
    default_out_suffix: str = '.sgy',
    default_sort_within: str = 'chno',
) -> InferCommonParsed:
    """
    Parse common segy2segy infer inputs for psn/pair/blindtrace pipelines.
    """
    paths_cfg = require_dict(cfg, 'paths')
    infer_cfg = require_dict(cfg, 'infer')
    tile_cfg_obj = require_dict(cfg, 'tile')

    segy_files = resolve_segy_files(
        base_dir=base_dir,
        segy_files=require_list_str(paths_cfg, 'segy_files'),
    )
    validate_files_exist(segy_files)

    out_dir_raw = optional_str(paths_cfg, 'out_dir', default_out_dir)
    out_dir = Path(resolve_relpath(base_dir, out_dir_raw))
    out_dir.mkdir(parents=True, exist_ok=True)

    overwrite = optional_bool(infer_cfg, 'overwrite', default=False)
    out_suffix = optional_str(infer_cfg, 'out_suffix', default_out_suffix)

    sort_within_raw = optional_str(infer_cfg, 'sort_within', default_sort_within).lower()
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
            if not is_strict_int(item):
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

    return InferCommonParsed(
        segy_files=segy_files,
        out_dir=out_dir,
        overwrite=overwrite,
        out_suffix=out_suffix,
        sort_within=sort_within,
        ffids=ffids,
        tiled2d=tile_cfg,
    )
