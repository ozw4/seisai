from __future__ import annotations

from pathlib import Path

import pytest
from seisai_utils.config import load_config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.mark.e2e
def test_load_config_resolves_psn_paths_independent_of_cwd(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = _repo_root()
    cfg_path = repo_root / 'tests' / 'e2e' / 'config_train_psn.yaml'
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)

    expected_base = cfg_path.parent.resolve()
    expected = {
        'segy_files': [
            str(
                (
                    expected_base / '../../test_data/ridgecrest_das/20200623002546.sgy'
                ).resolve()
            )
        ],
        'phase_pick_files': [
            str(
                (
                    expected_base
                    / '../../test_data/ridgecrest_das/20200623002546_phase_picks.npz'
                ).resolve()
            )
        ],
        'infer_segy_files': [
            str(
                (
                    expected_base / '../../test_data/ridgecrest_das/20200623002546.sgy'
                ).resolve()
            )
        ],
        'infer_phase_pick_files': [
            str(
                (
                    expected_base
                    / '../../test_data/ridgecrest_das/20200623002546_phase_picks.npz'
                ).resolve()
            )
        ],
    }

    cfg1 = load_config(cfg_path)
    for key, expected_value in expected.items():
        assert cfg1['paths'][key] == expected_value

    other_cwd = tmp_path / 'other-cwd'
    other_cwd.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(other_cwd)

    cfg2 = load_config(cfg_path)
    for key, expected_value in expected.items():
        assert cfg2['paths'][key] == expected_value
