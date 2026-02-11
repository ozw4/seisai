from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    'CommonTrainConfig',
    'InferLoopConfig',
    'OutputConfig',
    'SeedsConfig',
    'TrainLoopConfig',
]


@dataclass(frozen=True)
class SeedsConfig:
    seed_train: int
    seed_infer: int


@dataclass(frozen=True)
class TrainLoopConfig:
    epochs: int
    samples_per_epoch: int
    train_batch_size: int
    train_num_workers: int
    max_norm: float
    use_amp_train: bool
    gradient_accumulation_steps: int = 1
    print_freq: int = 10


@dataclass(frozen=True)
class InferLoopConfig:
    infer_batch_size: int
    infer_num_workers: int
    infer_max_batches: int
    vis_n: int


@dataclass(frozen=True)
class OutputConfig:
    out_dir: str
    vis_subdir: str


@dataclass(frozen=True)
class CommonTrainConfig:
    output: OutputConfig
    seeds: SeedsConfig
    train: TrainLoopConfig
    infer: InferLoopConfig

    def __post_init__(self) -> None:
        if self.train.epochs <= 0:
            msg = 'train.epochs must be positive'
            raise ValueError(msg)
        if self.train.samples_per_epoch <= 0:
            msg = 'train.samples_per_epoch must be positive'
            raise ValueError(msg)
        if self.train.train_batch_size <= 0:
            msg = 'train.batch_size must be positive'
            raise ValueError(msg)
        if self.train.gradient_accumulation_steps <= 0:
            msg = 'train.gradient_accumulation_steps must be positive'
            raise ValueError(msg)
        if self.infer.infer_batch_size <= 0:
            msg = 'infer.batch_size must be positive'
            raise ValueError(msg)
        if self.infer.infer_max_batches <= 0:
            msg = 'infer.max_batches must be positive'
            raise ValueError(msg)
        if self.infer.vis_n < 0:
            msg = 'vis.n must be >= 0'
            raise ValueError(msg)
        if self.infer.infer_num_workers != 0:
            msg = 'infer.num_workers must be 0 to keep fixed inference samples'
            raise ValueError(
                msg
            )
