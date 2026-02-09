from __future__ import annotations

from .config import TrackingConfig
from .tracker import BaseTracker, NoOpTracker

__all__ = ['create_tracker']


def create_tracker(cfg: TrackingConfig) -> BaseTracker:
    if not isinstance(cfg, TrackingConfig):
        msg = 'cfg must be TrackingConfig'
        raise TypeError(msg)
    if not cfg.enabled:
        return NoOpTracker()

    from .mlflow_tracker import MLflowTracker

    return MLflowTracker()
