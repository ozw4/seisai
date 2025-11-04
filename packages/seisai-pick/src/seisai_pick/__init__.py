from .detectors import (
	detect_event_pick_cluster,
	detect_event_stalta_majority,
)
from .stalta import STALTA

__all__ = [
	'STALTA',
	'detect_event_pick_cluster',
	'detect_event_stalta_majority',
]
