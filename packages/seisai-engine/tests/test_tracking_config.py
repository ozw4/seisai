from __future__ import annotations

from pathlib import Path

from seisai_engine.tracking.config import load_tracking_config


def test_load_tracking_config_default_disabled(tmp_path: Path) -> None:
    cfg = {'paths': {'out_dir': str(tmp_path)}}
    tracking = load_tracking_config(cfg, tmp_path)
    assert tracking.enabled is False


def test_tracking_uri_resolved_from_base_dir(tmp_path: Path) -> None:
    cfg = {'tracking': {'enabled': True, 'tracking_uri': 'file:./mlruns'}}
    tracking = load_tracking_config(cfg, tmp_path)
    expected = f"file:{(tmp_path / 'mlruns').resolve()}"
    assert tracking.tracking_uri == expected
