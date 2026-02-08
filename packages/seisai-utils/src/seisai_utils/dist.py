"""Distributed training utilities.

This module provides small helper functions for PyTorch distributed setups, such
as initialization, rank/world-size queries, master-only behaviors, and disabling
prints for non-master processes.
"""

import builtins as __builtin__
import os
from typing import Any

import torch
import torch.distributed as dist


def setup_for_distributed(*, is_master: bool) -> None:
    """Override the built-in `print` to suppress output on non-master processes.

    Parameters
    ----------
    is_master : bool
            Whether the current process is the master (rank 0) process; if False,
            prints are suppressed unless `force=True` is passed to `print`.
    """  # noqa: D413
    builtin_print = __builtin__.print

    def distributed_print(*args: Any, **kwargs: Any) -> None:
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = distributed_print


def is_dist_avail_and_initialized() -> bool:
    """Check whether torch.distributed is available and initialized.

    Returns
    -------
    bool
            True if the distributed package is available and the default process group
            has been initialized; otherwise False.

    """
    if not dist.is_available():
        return False
    return dist.is_initialized()


def get_world_size() -> int:
    """Get the total number of processes in the current distributed group.

    Returns
    -------
    int
            The world size if distributed is available and initialized; otherwise 1.

    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Get the rank of the current process.

    Returns
    -------
    int
            The rank if distributed is available and initialized; otherwise 0.

    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    """Check whether the current process is the main (rank 0) process."""
    return get_rank() == 0


def save_on_master(*args: Any, **kwargs: Any) -> None:
    """Save a PyTorch object only on the main (rank 0) process.

    This is a thin wrapper around `torch.save` to avoid multiple processes
    writing the same checkpoint in distributed training.

    Parameters
    ----------
    *args
            Positional arguments forwarded to `torch.save`.
    **kwargs
            Keyword arguments forwarded to `torch.save`.

    """
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args) -> None:
    """Initialize PyTorch distributed training based on environment variables.

    This function populates common distributed attributes on the provided `args`
    object and initializes the default PyTorch distributed process group. If no
    distributed configuration is detected, it disables distributed mode and
    returns.

    Parameters
    ----------
    args
            An argument/namespace object that will be updated in-place. Commonly
            expected attributes include `dist_url` and (optionally) `world_size`.
            The following attributes may be set/updated: `rank`, `world_size`,
            `local_rank`, `dist_backend`, and `distributed`.

    Raises
    ------
    RuntimeError
            If `torch.distributed.init_process_group` fails (e.g., invalid URL,
            backend issues, or networking problems).

    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ and args.world_size > 1:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.local_rank = args.rank % torch.cuda.device_count()
    elif hasattr(args, 'rank'):
        pass
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.local_rank)
    args.dist_backend = 'nccl'
    print(f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    setup_for_distributed(is_master=(args.rank == 0))
