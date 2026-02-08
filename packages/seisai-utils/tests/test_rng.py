import os

import numpy as np
import torch
from seisai_utils.rng import get_np_rng, set_seed, worker_init_fn


def test_set_seed_reproducible_for_repo_numpy_rng_and_torch() -> None:
    """Ensure set_seed makes NumPy RNG and torch RNG outputs reproducible within the repo."""
    seed = 123

    set_seed(seed)
    np_output_first = get_np_rng().random(4)
    torch_output_first = torch.rand(4)

    set_seed(seed)
    np_output_second = get_np_rng().random(4)
    torch_output_second = torch.rand(4)

    assert np.array_equal(np_output_first, np_output_second)
    assert torch.equal(torch_output_first, torch_output_second)


def test_set_seed_updates_pythonhashseed() -> None:
    """Ensure set_seed updates the PYTHONHASHSEED environment variable."""
    set_seed(7)
    assert os.environ['PYTHONHASHSEED'] == '7'


def test_worker_init_fn_reproducible_np_rng() -> None:
    """Verify worker_init_fn produces a reproducible NumPy RNG sequence for a fixed torch seed."""
    torch.manual_seed(2024)
    worker_init_fn(3)
    first = get_np_rng().random()

    torch.manual_seed(2024)
    worker_init_fn(3)
    second = get_np_rng().random()

    assert first == second
