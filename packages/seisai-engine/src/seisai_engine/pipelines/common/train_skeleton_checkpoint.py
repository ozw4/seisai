from __future__ import annotations

from typing import TYPE_CHECKING, Any

from seisai_engine.ema_controller import EmaConfig, EmaController

if TYPE_CHECKING:
    from .train_skeleton import TrainSkeletonSpec

__all__ = ['build_ckpt_payload']


def build_ckpt_payload(
    *,
    spec: TrainSkeletonSpec,
    epoch: int,
    global_step: int,
    scheduler_sig: dict[str, Any] | None,
    scheduler_state_dict: dict[str, Any] | None,
    ema_cfg_obj: EmaConfig | None = None,
    ema_controller: EmaController | None = None,
) -> dict[str, Any]:
    ckpt_payload: dict[str, Any] = {
        'version': 1,
        'pipeline': spec.pipeline,
        'epoch': int(epoch),
        'global_step': int(global_step),
        'model_sig': spec.model_sig,
        'model_state_dict': spec.model.state_dict(),
        'optimizer_state_dict': spec.optimizer.state_dict(),
        'lr_scheduler_sig': scheduler_sig,
        'lr_scheduler_state_dict': scheduler_state_dict,
        'cfg': spec.cfg,
    }
    if ema_cfg_obj is not None:
        if ema_controller is None:
            msg = 'ema_controller is required when ema_cfg_obj is provided'
            raise ValueError(msg)
        dev = None
        if ema_cfg_obj.device is not None:
            dev = str(ema_cfg_obj.device)
        ckpt_payload['ema_cfg'] = {
            'decay': float(ema_cfg_obj.decay),
            'start_step': int(ema_cfg_obj.start_step),
            'update_every': int(ema_cfg_obj.update_every),
            'use_for_infer': bool(ema_cfg_obj.use_for_infer),
            'device': dev,
        }
        ckpt_payload['ema_state_dict'] = ema_controller.state_dict_cpu()
        ckpt_payload['ema_step'] = int(ema_controller.step)
        ckpt_payload['infer_used_ema'] = bool(ema_cfg_obj.use_for_infer)

    if spec.ckpt_extra is not None:
        if not isinstance(spec.ckpt_extra, dict):
            msg = 'ckpt_extra must be dict when provided'
            raise TypeError(msg)
        for key, value in spec.ckpt_extra.items():
            if not isinstance(key, str):
                msg = 'ckpt_extra keys must be str'
                raise TypeError(msg)
            if key in ckpt_payload:
                msg = f'ckpt_extra key collides with base payload: {key}'
                raise ValueError(msg)
            ckpt_payload[key] = value

    return ckpt_payload
