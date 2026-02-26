from __future__ import annotations

from stage1.cfg import DEFAULT_STAGE1_CFG, Stage1Cfg, load_stage1_cfg_yaml
from stage1.model import build_model, build_model_from_cfg
from stage1.process_one import process_one_segy as _process_one_segy
from stage1.runner import (
    infer_gather_prob,
    make_velocity_feasible_filt_allow_vmin0,
    pad_samples_to_6016,
    INPUT_DIR,
    OUT_DIR,
    VIZ_DIRNAME,
    VIZ_EVERY_N_SHOTS,
    WEIGHTS_PATH,
    process_one_segy,
    run_stage1,
    run_stage1_cfg,
)


def main() -> None:
    from stage1.cli import main as _main

    _main()


if __name__ == '__main__':
    main()


__all__ = [
    'DEFAULT_STAGE1_CFG',
    'INPUT_DIR',
    'OUT_DIR',
    'Stage1Cfg',
    'VIZ_DIRNAME',
    'VIZ_EVERY_N_SHOTS',
    'WEIGHTS_PATH',
    '_process_one_segy',
    'build_model',
    'build_model_from_cfg',
    'infer_gather_prob',
    'load_stage1_cfg_yaml',
    'main',
    'make_velocity_feasible_filt_allow_vmin0',
    'pad_samples_to_6016',
    'process_one_segy',
    'run_stage1',
    'run_stage1_cfg',
]
