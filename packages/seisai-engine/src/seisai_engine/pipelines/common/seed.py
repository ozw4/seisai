from __future__ import annotations

from seisai_utils.rng import set_seed

__all__ = ['seed_all']


def seed_all(seed: int) -> None:
	set_seed(int(seed))
