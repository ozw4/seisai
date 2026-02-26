from __future__ import annotations

from stage4.cfg import (
    DEFAULT_STAGE4_CFG,
    Stage4Cfg,
    _load_stage4_cfg_from_yaml as _cfg_load_stage4_cfg_from_yaml,
    _parse_segy_exts,
    load_stage4_cfg_yaml,
)
from stage4.model import load_psn_model_and_eps
from stage4.process_one import process_one_pair as _process_one_pair
from stage4.runner import (
    _align_post_trough_shifts_to_neighbors,
    _build_win512_lookup,
    _post_trough_adjust_picks,
    _post_trough_apply_mask_from_offsets,
    _resolve_sidecar_path,
    _shift_pick_to_preceding_trough_1d,
    _stem_without_win512,
    infer_pick512_from_win,
    process_one_pair,
    run_stage4,
)


def _load_stage4_cfg_from_yaml(path):
    return _cfg_load_stage4_cfg_from_yaml(path)


def main(argv: list[str] | None = None) -> None:
    from stage4.cli import main as _main

    _main(argv)


if __name__ == '__main__':
    main()


__all__ = [
    'DEFAULT_STAGE4_CFG',
    'Stage4Cfg',
    '_align_post_trough_shifts_to_neighbors',
    '_build_win512_lookup',
    '_load_stage4_cfg_from_yaml',
    '_post_trough_adjust_picks',
    '_post_trough_apply_mask_from_offsets',
    '_process_one_pair',
    '_parse_segy_exts',
    '_resolve_sidecar_path',
    '_shift_pick_to_preceding_trough_1d',
    '_stem_without_win512',
    'infer_pick512_from_win',
    'load_psn_model_and_eps',
    'load_stage4_cfg_yaml',
    'main',
    'process_one_pair',
    'run_stage4',
]
