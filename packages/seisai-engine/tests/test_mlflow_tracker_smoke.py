from __future__ import annotations

from pathlib import Path

import pytest

from seisai_engine.tracking.mlflow_tracker import MLflowTracker


def test_mlflow_tracker_smoke(tmp_path: Path) -> None:
    pytest.importorskip('mlflow')

    mlruns = tmp_path / 'mlruns'
    tracking_uri = f'file:{mlruns}'
    artifact_path = tmp_path / 'artifact.txt'
    artifact_path.write_text('ok', encoding='utf-8')
    extra_artifact = tmp_path / 'run_summary.json'
    extra_artifact.write_text('{"ok":true}\n', encoding='utf-8')

    tracker = MLflowTracker()
    tracker.start_run(
        tracking_uri=tracking_uri,
        experiment='seisai/test',
        run_name='smoke',
        tags={'pipeline': 'test'},
        params={'train/epochs': 1},
        artifacts={'artifact.txt': artifact_path},
    )
    tracker.log_artifacts({'run_summary.json': extra_artifact})
    tracker.log_metrics({'train/loss': 1.0}, step=0)
    tracker.end_run(status='FINISHED')
