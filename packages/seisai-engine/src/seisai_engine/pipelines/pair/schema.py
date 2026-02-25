from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from seisai_engine.loss import composite

if TYPE_CHECKING:
    from seisai_engine.pipelines.common.config_schema import CommonTrainConfig

__all__ = [
    'PairCkptCfg',
    'PairDatasetCfg',
    'PairInferCfg',
    'PairInferConfig',
    'PairModelCfg',
    'PairPaths',
    'PairTileCfg',
    'PairTransformCfg',
    'PairTrainCfg',
    'PairTrainConfig',
    'PairVisCfg',
]


@dataclass(frozen=True)
class PairPaths:
    input_segy_files: list[str]
    target_segy_files: list[str]
    out_dir: str


@dataclass(frozen=True)
class PairDatasetCfg:
    max_trials: int
    use_header_cache: bool
    verbose: bool
    progress: bool
    primary_keys: tuple[str, ...]
    secondary_key_fixed: bool
    waveform_mode: str
    train_input_endian: str
    train_target_endian: str
    infer_input_endian: str
    infer_target_endian: str


@dataclass(frozen=True)
class PairTrainCfg:
    batch_size: int
    epochs: int
    lr: float
    subset_traces: int
    samples_per_epoch: int
    seed: int
    use_amp: bool
    max_norm: float
    num_workers: int


@dataclass(frozen=True)
class PairTransformCfg:
    time_len: int


@dataclass(frozen=True)
class PairInferCfg:
    batch_size: int
    max_batches: int
    subset_traces: int
    seed: int
    num_workers: int


@dataclass(frozen=True)
class PairTileCfg:
    tile_h: int
    overlap_h: int
    tiles_per_batch: int
    amp: bool
    use_tqdm: bool


@dataclass(frozen=True)
class PairVisCfg:
    out_subdir: str
    n: int
    cmap: str
    vmin: float
    vmax: float
    transpose_for_trace_time: bool
    per_trace_norm: bool
    per_trace_eps: float
    figsize: tuple[float, float]
    dpi: int


@dataclass(frozen=True)
class PairModelCfg:
    backbone: str
    pretrained: bool
    in_chans: int
    out_chans: int
    stage_strides: list[tuple[int, int]] | None
    extra_stages: int
    extra_stage_strides: list[tuple[int, int]] | None
    extra_stage_channels: tuple[int, ...] | None
    extra_stage_use_bn: bool
    pre_stages: int
    pre_stage_strides: list[tuple[int, int]] | None
    pre_stage_kernels: tuple[int, ...] | None
    pre_stage_channels: tuple[int, ...] | None
    pre_stage_use_bn: bool
    pre_stage_antialias: bool
    pre_stage_aa_taps: int
    pre_stage_aa_pad_mode: str
    decoder_channels: tuple[int, ...]
    decoder_scales: tuple[int, ...]
    upsample_mode: str
    attention_type: str | None
    intermediate_conv: bool


@dataclass(frozen=True)
class PairCkptCfg:
    save_best_only: bool
    metric: str
    mode: str


@dataclass(frozen=True)
class PairTrainConfig:
    common: CommonTrainConfig
    paths: PairPaths
    infer_paths: PairPaths
    dataset: PairDatasetCfg
    train: PairTrainCfg
    loss_specs_train: tuple[composite.LossSpec, ...]
    loss_specs_eval: tuple[composite.LossSpec, ...]
    transform: PairTransformCfg
    infer: PairInferCfg
    tile: PairTileCfg
    vis: PairVisCfg
    ckpt: PairCkptCfg
    model: PairModelCfg


@dataclass(frozen=True)
class PairInferConfig:
    paths: PairPaths
    dataset: PairDatasetCfg
    infer: PairInferCfg
    tile: PairTileCfg
    vis: PairVisCfg
    ckpt: PairCkptCfg
    model: PairModelCfg
