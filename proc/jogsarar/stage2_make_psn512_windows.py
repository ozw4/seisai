from __future__ import annotations

from stage2.cfg import DEFAULT_STAGE2_CFG, Stage2Cfg
from stage2.process_one import process_one_segy as _process_one_segy
from stage2.runner import (
    _percentile_threshold,
    build_keep_mask,
    compute_global_thresholds,
    infer_npz_path_for_segy,
    out_pick_csr_npz_path_for_out,
    out_segy_path_for_in,
    out_sidecar_npz_path_for_out,
    process_one_segy,
    run_stage2,
)


def main(argv: list[str] | None = None) -> None:
    from stage2.cli import main as _main

    _main(argv)


if __name__ == '__main__':
    main()


__all__ = [
    'DEFAULT_STAGE2_CFG',
    'Stage2Cfg',
    '_percentile_threshold',
    '_process_one_segy',
    'build_keep_mask',
    'compute_global_thresholds',
    'infer_npz_path_for_segy',
    'main',
    'out_pick_csr_npz_path_for_out',
    'out_segy_path_for_in',
    'out_sidecar_npz_path_for_out',
    'process_one_segy',
    'run_stage2',
]
