from __future__ import annotations

from pathlib import Path

__all__ = ['BaseTracker', 'NoOpTracker']


class BaseTracker:
    def start_run(
        self,
        *,
        tracking_uri: str,
        experiment: str,
        run_name: str,
        tags: dict,
        params: dict,
        artifacts: dict[str, Path],
    ) -> None:
        raise NotImplementedError

    def log_metrics(self, metrics: dict[str, float], *, step: int) -> None:
        raise NotImplementedError

    def log_best(
        self,
        *,
        ckpt_path: Path,
        vis_epoch_dir: Path | None,
        vis_max_files: int,
    ) -> None:
        raise NotImplementedError

    def log_artifacts(self, artifacts: dict[str, Path]) -> None:
        raise NotImplementedError

    def end_run(self, *, status: str) -> None:
        raise NotImplementedError


class NoOpTracker(BaseTracker):
    def start_run(
        self,
        *,
        tracking_uri: str,
        experiment: str,
        run_name: str,
        tags: dict,
        params: dict,
        artifacts: dict[str, Path],
    ) -> None:
        return None

    def log_metrics(self, metrics: dict[str, float], *, step: int) -> None:
        return None

    def log_best(
        self,
        *,
        ckpt_path: Path,
        vis_epoch_dir: Path | None,
        vis_max_files: int,
    ) -> None:
        return None

    def log_artifacts(self, artifacts: dict[str, Path]) -> None:
        return None

    def end_run(self, *, status: str) -> None:
        return None
