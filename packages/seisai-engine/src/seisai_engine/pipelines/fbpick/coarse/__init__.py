from .build_model import build_model
from .build_plan import (
    build_input_only_plan,
    build_input_only_plan_from_config,
    build_plan,
    build_plan_from_config,
)
from .config import (
    CoarseCkptCfg,
    CoarseEvalCfg,
    CoarseInputCfg,
    CoarseInferCfg,
    CoarseModelCfg,
    CoarseTargetCfg,
    CoarseTrainCfg,
    CoarseTrainConfig,
    load_coarse_train_config,
)

STAGE_NAME = 'coarse'
TRAIN_MAIN_TARGET = 'seisai_engine.pipelines.fbpick.coarse.train.main'
INFER_MAIN_TARGET = 'seisai_engine.pipelines.fbpick.coarse.infer_segy2npz.main'

__all__ = [
    'CoarseCkptCfg',
    'CoarseEvalCfg',
    'CoarseInputCfg',
    'CoarseInferCfg',
    'CoarseModelCfg',
    'CoarseTargetCfg',
    'CoarseTrainCfg',
    'CoarseTrainConfig',
    'STAGE_NAME',
    'TRAIN_MAIN_TARGET',
    'INFER_MAIN_TARGET',
    'build_input_only_plan',
    'build_input_only_plan_from_config',
    'build_model',
    'build_plan',
    'build_plan_from_config',
    'load_coarse_train_config',
]
