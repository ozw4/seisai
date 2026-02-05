from __future__ import annotations

import warnings
from pathlib import Path

import pytest

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


def _repo_root() -> Path:
	return Path(__file__).resolve().parents[2]


@pytest.mark.e2e
def _run_e2e(*, out_dir: Path) -> Path:
	repo_root = _repo_root()
	cfg = repo_root / 'tests' / 'e2e' / 'config_train_psn.yaml'
	if not cfg.is_file():
		raise FileNotFoundError(cfg)

	import examples.example_train_psn as m

	m.main(argv=['--config', str(cfg), '--vis_out_dir', str(out_dir)])

	return out_dir / 'psn_debug_epoch0000.png'


@pytest.mark.integration
def test_e2e_train_psn_one_epoch(tmp_path: Path) -> None:
	out_dir = tmp_path / '_psn_vis'
	png = _run_e2e(out_dir=out_dir)
	assert png.is_file()
	assert png.stat().st_size > 0
