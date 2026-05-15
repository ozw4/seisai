from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import TextIO

from .config import PhysicalProgressCfg

__all__ = [
    'ConsoleProgressReporter',
    'NullProgressReporter',
    'build_progress_reporter',
]

_LEVEL_ORDER = {
    'none': 0,
    'batch': 1,
    'sgy': 2,
    'stage': 3,
    'fit': 4,
}


class NullProgressReporter:
    def emit(self, event: str, **fields: object) -> None:
        return


@dataclass
class _ThrottleState:
    last_time: float = 0.0
    last_done: int = 0


class ConsoleProgressReporter:
    def __init__(
        self,
        cfg: PhysicalProgressCfg,
        *,
        stream_obj: TextIO | None = None,
    ) -> None:
        self.cfg = cfg
        self._stream = stream_obj if stream_obj is not None else _stream_from_name(cfg.stream)
        self._started_at = time.perf_counter()
        self._throttle: dict[str, _ThrottleState] = {}

    def emit(self, event: str, **fields: object) -> None:
        if not bool(self.cfg.enabled):
            return
        if not bool(self.cfg.print_on_non_tty) and not self._stream.isatty():
            return
        if not self._event_enabled(event):
            return
        if event == 'fit.progress' and not self._fit_progress_allowed(event, fields):
            return

        line = self._format_line(event, fields)
        print(line, file=self._stream, flush=True)

    def _event_enabled(self, event: str) -> bool:
        level = str(self.cfg.level)
        if level == 'none':
            return False
        required = _required_level(event)
        if required == 'stage' and not bool(self.cfg.include_stage_events):
            return False
        if event == 'physics.done' and not bool(self.cfg.include_summary):
            return False
        return _LEVEL_ORDER[level] >= _LEVEL_ORDER[required]

    def _fit_progress_allowed(self, event: str, fields: dict[str, object]) -> bool:
        done = _as_int(fields.get('done'))
        total = _as_int(fields.get('total'))
        force = bool(fields.get('force', False))
        if force or (total is not None and done is not None and done >= total):
            self._record_throttle(event, done)
            return True

        now = time.perf_counter()
        state = self._throttle.get(event)
        if state is None:
            self._throttle[event] = _ThrottleState(last_time=now, last_done=done or 0)
            return True

        elapsed_ok = (now - state.last_time) >= float(self.cfg.interval_sec)
        done_ok = False
        if done is not None:
            done_ok = (done - state.last_done) >= int(self.cfg.min_interval_fit_calls)
        if elapsed_ok or done_ok:
            state.last_time = now
            state.last_done = done or state.last_done
            return True
        return False

    def _record_throttle(self, event: str, done: int | None) -> None:
        self._throttle[event] = _ThrottleState(
            last_time=time.perf_counter(),
            last_done=done or 0,
        )

    def _format_line(self, event: str, fields: dict[str, object]) -> str:
        prefix, action = _prefix_and_action(event, fields)
        clean_fields = {
            key: value
            for key, value in fields.items()
            if key
            not in {
                'batch_index',
                'batch_total',
                'force',
            }
        }
        if event == 'fit.progress':
            action = _fit_action(clean_fields)
            for key in ('done', 'total'):
                clean_fields.pop(key, None)
        parts = [prefix, action]
        for key, value in clean_fields.items():
            if value is None:
                continue
            parts.append(f'{key}={_format_value(key, value)}')
        return ' '.join(parts)


def build_progress_reporter(
    cfg: PhysicalProgressCfg,
    *,
    stream_obj: TextIO | None = None,
) -> ConsoleProgressReporter | NullProgressReporter:
    if not bool(cfg.enabled) or str(cfg.level) == 'none':
        return NullProgressReporter()
    return ConsoleProgressReporter(cfg, stream_obj=stream_obj)


def _stream_from_name(name: str) -> TextIO:
    if str(name) == 'stdout':
        return sys.stdout
    return sys.stderr


def _required_level(event: str) -> str:
    if event.startswith('physics-batch.'):
        return 'batch'
    if event in {'physics.start', 'physics.done'}:
        return 'sgy'
    if event == 'fit.progress':
        return 'fit'
    return 'stage'


def _prefix_and_action(event: str, fields: dict[str, object]) -> tuple[str, str]:
    if event.startswith('physics-batch.'):
        return '[physics-batch]', event.split('.', 1)[1].replace('_', ' ')
    if event.startswith('physics.'):
        index = fields.get('batch_index')
        total = fields.get('batch_total')
        prefix = '[physics]'
        if index is not None and total is not None:
            prefix = f'[physics {index}/{total}]'
        return prefix, event.split('.', 1)[1].replace('_', ' ')
    if event.startswith('physical-center.'):
        return '[physical-center]', event.split('.', 1)[1].replace('_', ' ')
    if event == 'fit.progress':
        return '[fit]', 'progress'
    return '[physics]', event.replace('.', ' ')


def _fit_action(fields: dict[str, object]) -> str:
    done = fields.pop('done', None)
    total = fields.pop('total', None)
    if done is None or total is None:
        return 'progress'
    return f'{done}/{total} done'


def _format_value(key: str, value: object) -> str:
    if isinstance(value, bool):
        return 'on' if value else 'off'
    if isinstance(value, float):
        if key.endswith('_sec') or key in {'elapsed', 'eta'}:
            return f'{value:.2f}s'
        if key == 'rate':
            return f'{value:.2f}/s'
        return f'{value:.3g}'
    return str(value)


def _as_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
