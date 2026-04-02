from __future__ import annotations

from collections.abc import Sequence

from seisai_dataset import InputOnlyPlan
from seisai_dataset.local_window_dataset import (
    LocalWindowDataset,
    LocalWindowDatasetConfig,
)
from seisai_utils.fs import validate_files_exist

from .build_dataset import FineLocalDataset
from .config import FineCoarseArtifactCfg, FineInputCfg, FineWindowCfg

__all__ = [
    'build_window_dataset',
    'resolve_coarse_artifact_paths',
]


def _require_input_cfg(cfg: FineInputCfg) -> FineInputCfg:
    if not isinstance(cfg, FineInputCfg):
        msg = 'input_cfg must be FineInputCfg'
        raise TypeError(msg)
    return cfg


def _require_window_cfg(cfg: FineWindowCfg) -> FineWindowCfg:
    if not isinstance(cfg, FineWindowCfg):
        msg = 'window_cfg must be FineWindowCfg'
        raise TypeError(msg)
    return cfg


def _require_coarse_seed_cfg(cfg: FineCoarseArtifactCfg) -> FineCoarseArtifactCfg:
    if not isinstance(cfg, FineCoarseArtifactCfg):
        msg = 'coarse_seed_cfg must be FineCoarseArtifactCfg'
        raise TypeError(msg)
    return cfg


def resolve_coarse_artifact_paths(coarse_seed_cfg: FineCoarseArtifactCfg) -> tuple[str, str]:
    cfg = _require_coarse_seed_cfg(coarse_seed_cfg)
    return str(cfg.artifact_npz_path), str(cfg.artifact_meta_path)


def build_window_dataset(
    *,
    segy_files: Sequence[str],
    transform,
    plan: InputOnlyPlan,
    input_cfg: FineInputCfg,
    window_cfg: FineWindowCfg,
    coarse_seed_cfg: FineCoarseArtifactCfg,
    use_header_cache: bool = True,
    header_cache_dir: str | None = None,
    waveform_mode: str = 'mmap',
    segy_endian: str = 'big',
) -> FineLocalDataset:
    input_cfg_checked = _require_input_cfg(input_cfg)
    window_cfg_checked = _require_window_cfg(window_cfg)
    coarse_cfg_checked = _require_coarse_seed_cfg(coarse_seed_cfg)

    if len(segy_files) == 0:
        msg = 'segy_files must be non-empty'
        raise ValueError(msg)
    if not isinstance(plan, InputOnlyPlan):
        msg = 'fine build_window_dataset requires InputOnlyPlan'
        raise TypeError(msg)
    if transform is not None and not callable(transform):
        msg = 'transform must be callable or None'
        raise TypeError(msg)

    validate_files_exist(list(segy_files))
    coarse_artifact_npz_path, coarse_artifact_meta_path = resolve_coarse_artifact_paths(
        coarse_cfg_checked
    )

    base_dataset = LocalWindowDataset(
        list(segy_files),
        cfg=LocalWindowDatasetConfig(
            local_window_len=int(window_cfg_checked.local_window_len),
            mode='infer',
        ),
        coarse_artifact_npz_path=coarse_artifact_npz_path,
        coarse_artifact_meta_path=coarse_artifact_meta_path,
        plan=plan,
        transform=transform,
        use_header_cache=bool(use_header_cache),
        header_cache_dir=header_cache_dir,
        waveform_mode=str(waveform_mode),
        segy_endian=str(segy_endian),
    )
    return FineLocalDataset(
        base_dataset,
        input_cfg=input_cfg_checked,
        require_target=False,
        expected_mode='infer',
        expected_seed_source='coarse',
    )
