from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import cli.run_fbpick_physics_qc as physics_qc_cli


def _ffid_info(key_to_indices: dict[int, list[int] | np.ndarray]) -> dict[str, object]:
    max_index = max(
        int(np.max(np.asarray(indices, dtype=np.int64)))
        for indices in key_to_indices.values()
        if int(np.asarray(indices).size) > 0
    )
    return {
        'ffid_key_to_indices': {
            key: np.asarray(indices, dtype=np.int64)
            for key, indices in key_to_indices.items()
        },
        'chno_values': np.arange(max_index + 1, dtype=np.int64),
    }


def test_load_vis_cfg_parses_skip_keys_and_max_traces() -> None:
    vis_cfg = physics_qc_cli._load_vis_cfg(
        {
            'vis': {
                'skip_gather_keys': {'ffid': [0], 'cmp': [-1, 2]},
                'max_traces_per_gather': 10000,
            }
        }
    )

    assert vis_cfg['skip_gather_keys'] == {'ffid': {0}, 'cmp': {-1, 2}}
    assert vis_cfg['max_traces_per_gather'] == 10000


def test_load_vis_cfg_allows_null_max_traces() -> None:
    vis_cfg = physics_qc_cli._load_vis_cfg(
        {'vis': {'max_traces_per_gather': None}}
    )

    assert vis_cfg['max_traces_per_gather'] is None


@pytest.mark.parametrize(
    'vis',
    [
        {'skip_gather_keys': None},
        {'skip_gather_keys': {'ffid': [False]}},
        {'skip_gather_keys': {'ffid': (0,)}},
        {'skip_gather_keys': {1: [0]}},
        {'max_traces_per_gather': 0},
        {'max_traces_per_gather': True},
    ],
)
def test_load_vis_cfg_rejects_invalid_new_fields(vis: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        physics_qc_cli._load_vis_cfg({'vis': vis})


def test_iter_vis_gathers_skips_configured_key() -> None:
    info = _ffid_info({0: [0, 1], 1: [2], 2: [3]})

    yielded = list(
        physics_qc_cli._iter_vis_gathers(
            info,
            primary_keys=['ffid'],
            max_gathers=2,
            skip_gather_keys={'ffid': {0}},
            max_traces_per_gather=None,
            segy_path='line.sgy',
        )
    )

    assert [gather_key for _, gather_key, _ in yielded] == [1, 2]


def test_iter_vis_gathers_counts_accepted_gathers_not_skipped() -> None:
    info = _ffid_info({0: [0], 1: [1], 2: [2], 3: [3]})

    yielded = list(
        physics_qc_cli._iter_vis_gathers(
            info,
            primary_keys=['ffid'],
            max_gathers=2,
            skip_gather_keys={'ffid': {0}},
            max_traces_per_gather=None,
            segy_path='line.sgy',
        )
    )

    assert [gather_key for _, gather_key, _ in yielded] == [1, 2]


def test_iter_vis_gathers_skips_oversized_gather(
    capsys: pytest.CaptureFixture[str],
) -> None:
    info = _ffid_info({1: np.arange(5), 2: np.arange(5, 7)})

    yielded = list(
        physics_qc_cli._iter_vis_gathers(
            info,
            primary_keys=['ffid'],
            max_gathers=1,
            skip_gather_keys={},
            max_traces_per_gather=3,
            segy_path='/data/line.sgy',
        )
    )

    assert [gather_key for _, gather_key, _ in yielded] == [2]
    captured = capsys.readouterr()
    assert (
        'skip oversized gather: file=/data/line.sgy '
        'primary=ffid key=1 traces=5 limit=3'
    ) in captured.out


def test_save_vis_pngs_skips_oversized_before_mmap_access(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FailingMmap:
        def __getitem__(self, index: int) -> np.ndarray:
            raise AssertionError(f'mmap should not be read for skipped gather {index}')

    def _unexpected_save(*args, **kwargs) -> Path:
        raise AssertionError('PNG writer should not be called')

    segy_path = tmp_path / 'line' / 'huge.sgy'
    segy_path.parent.mkdir()
    info = _ffid_info({1: np.arange(5)})
    info['mmap'] = FailingMmap()
    runtime = SimpleNamespace(save_fbpick_physics_qc_gather_png=_unexpected_save)

    out_paths = physics_qc_cli._save_vis_pngs(
        info=info,
        segy_path=str(segy_path),
        out_dir=tmp_path / 'out',
        gt_pick_i=np.zeros(5, dtype=np.int64),
        coarse_pick_i=np.zeros(5, dtype=np.int64),
        robust_pick_i=np.zeros(5, dtype=np.int64),
        dataset_cfg={'primary_keys': ['ffid']},
        vis_cfg={
            'max_gathers_per_file': 1,
            'skip_gather_keys': {},
            'max_traces_per_gather': 3,
        },
        runtime=runtime,
    )

    assert out_paths == []
    captured = capsys.readouterr()
    assert 'skip oversized gather:' in captured.out
    assert f'No gather PNGs written for {segy_path}: all candidates were skipped' in (
        captured.out
    )
