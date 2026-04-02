from .build_dataset import FineLocalDataset, build_dataset
from .build_model import build_model
from .build_plan import (
    build_input_only_plan,
    build_input_only_plan_from_config,
    build_plan,
    build_plan_from_config,
)
from .build_window_dataset import build_window_dataset, resolve_coarse_artifact_paths
from .config import (
    FineCkptCfg,
    FineCoarseArtifactCfg,
    FineEvalCfg,
    FineInferCfg,
    FineInferConfig,
    FineInputCfg,
    FineModelCfg,
    FineTargetCfg,
    FineTrainCfg,
    FineTrainConfig,
    FineWindowCfg,
    load_fine_infer_config,
    load_fine_train_config,
    resolve_default_coarse_artifact_paths,
)
from .infer import (
    FineInferBatchResult,
    local_pick_to_raw_pick,
    local_prob_to_pick_confidence,
    logits_to_local_prob,
    run_infer_batch,
    run_infer_epoch,
)
from .infer_from_coarse import run_infer_and_write
from .loss import build_fine_criterion

STAGE_NAME = 'fine'
TRAIN_MAIN_TARGET = 'seisai_engine.pipelines.fbpick.fine.train.main'
INFER_MAIN_TARGET = 'seisai_engine.pipelines.fbpick.fine.infer_from_coarse.main'

__all__ = [
    'FineCkptCfg',
    'FineCoarseArtifactCfg',
    'FineEvalCfg',
    'FineInferCfg',
    'FineInferConfig',
    'FineInferBatchResult',
    'FineInputCfg',
    'FineLocalDataset',
    'FineModelCfg',
    'FineTargetCfg',
    'FineTrainCfg',
    'FineTrainConfig',
    'FineWindowCfg',
    'STAGE_NAME',
    'TRAIN_MAIN_TARGET',
    'INFER_MAIN_TARGET',
    'build_dataset',
    'build_input_only_plan',
    'build_input_only_plan_from_config',
    'build_fine_criterion',
    'build_model',
    'build_plan',
    'build_plan_from_config',
    'build_window_dataset',
    'local_pick_to_raw_pick',
    'local_prob_to_pick_confidence',
    'load_fine_infer_config',
    'load_fine_train_config',
    'logits_to_local_prob',
    'resolve_coarse_artifact_paths',
    'resolve_default_coarse_artifact_paths',
    'run_infer_and_write',
    'run_infer_batch',
    'run_infer_epoch',
]
