from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

import cli.run_pair_infer as m

from ._infer_e2e_helpers import (
    make_dummy_ckpt,
    make_synthetic_segy,
    read_segy_traces,
    read_json,
    segy_text_header_contains,
)

_EPS_STD = 1.0e-8


def _soft_clip_tanh_np(x: np.ndarray, clip_abs: float) -> np.ndarray:
    return np.tanh(x / float(clip_abs)) * float(clip_abs)


def _soft_clipped_input_denorm(
    traces_hw: np.ndarray,
    *,
    clip_abs: float,
    eps_std: float = _EPS_STD,
) -> np.ndarray:
    mean = traces_hw.mean(axis=1, keepdims=True)
    std = traces_hw.std(axis=1, keepdims=True) + float(eps_std)
    standardized = (traces_hw - mean) / std
    clipped = _soft_clip_tanh_np(standardized, clip_abs)
    return clipped * std + mean


def _run_pair_infer(
    *,
    segy_path: Path,
    ckpt_path: Path,
    out_dir: Path,
    unknown_overrides: list[str] | None = None,
) -> tuple[Path, Path]:
    cfg = {
        'paths': {
            'segy_files': [str(segy_path)],
            'out_dir': str(out_dir),
        },
        'infer': {
            'ckpt_path': str(ckpt_path),
            'device': 'cpu',
            'out_suffix': '_pred.sgy',
            'overwrite': True,
            'sort_within': 'chno',
            'ffids': [1],
            'note': 'e2e',
        },
        'tile': {
            'tile_h': 8,
            'overlap_h': 0,
            'tile_w': 64,
            'overlap_w': 0,
            'tiles_per_batch': 4,
            'amp': False,
            'use_tqdm': False,
        },
    }

    cfg_path = out_dir / 'config_infer_pair.yaml'
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')

    argv = ['--config', str(cfg_path)]
    if unknown_overrides is not None:
        argv.extend(list(unknown_overrides))
    m.main(argv=argv)

    out_segy = out_dir / f'{segy_path.stem}.pair_pred.sgy'
    out_meta = out_segy.with_suffix(out_segy.suffix + '.mlmeta.json')
    return out_segy, out_meta


@pytest.mark.e2e
def test_e2e_infer_pair(tmp_path: Path) -> None:
    segy_path = make_synthetic_segy(tmp_path)
    ckpt_path = make_dummy_ckpt(tmp_path, pipeline='pair', in_chans=1, out_chans=1)
    out_dir = tmp_path / '_pair_infer_out'
    out_segy, out_meta = _run_pair_infer(
        segy_path=segy_path,
        ckpt_path=ckpt_path,
        out_dir=out_dir,
        unknown_overrides=['tile.tile_w=32'],
    )

    assert out_segy.is_file()
    assert out_segy.stat().st_size > 0
    assert out_meta.is_file()
    assert out_meta.stat().st_size > 0

    meta = read_json(out_meta)
    assert meta['pipeline'] == 'pair'
    assert meta['tile']['tile_w'] == 32
    assert meta['pair']['residual_learning'] is False
    assert meta['pair']['input_soft_clip_abs'] is None
    assert meta['note'] == 'e2e'

    assert segy_text_header_contains(out_segy, 'ML pipeline=pair')


@pytest.mark.e2e
def test_e2e_infer_pair_residual_learning_reconstructs_input(tmp_path: Path) -> None:
    segy_path = make_synthetic_segy(tmp_path)
    input_traces = read_segy_traces(segy_path)
    clip_abs = 0.5
    ckpt_path = make_dummy_ckpt(
        tmp_path,
        pipeline='pair',
        in_chans=1,
        out_chans=1,
        zero_params=True,
        cfg={'pair': {'residual_learning': False, 'input_soft_clip_abs': None}},
    )

    direct_out_dir = tmp_path / '_pair_infer_out_direct'
    direct_segy, direct_meta_path = _run_pair_infer(
        segy_path=segy_path,
        ckpt_path=ckpt_path,
        out_dir=direct_out_dir,
    )

    residual_out_dir = tmp_path / '_pair_infer_out_residual'
    residual_segy, residual_meta_path = _run_pair_infer(
        segy_path=segy_path,
        ckpt_path=ckpt_path,
        out_dir=residual_out_dir,
        unknown_overrides=['pair.residual_learning=true'],
    )

    clipped_out_dir = tmp_path / '_pair_infer_out_residual_clipped'
    clipped_segy, clipped_meta_path = _run_pair_infer(
        segy_path=segy_path,
        ckpt_path=ckpt_path,
        out_dir=clipped_out_dir,
        unknown_overrides=[
            'pair.residual_learning=true',
            f'pair.input_soft_clip_abs={clip_abs}',
        ],
    )

    direct_traces = read_segy_traces(direct_segy)
    residual_traces = read_segy_traces(residual_segy)
    clipped_traces = read_segy_traces(clipped_segy)
    direct_meta = read_json(direct_meta_path)
    residual_meta = read_json(residual_meta_path)
    clipped_meta = read_json(clipped_meta_path)
    expected_clipped = _soft_clipped_input_denorm(
        input_traces,
        clip_abs=clip_abs,
    )

    assert not np.allclose(direct_traces, input_traces)
    assert np.allclose(residual_traces, input_traces, atol=1.0e-5)
    assert np.allclose(clipped_traces, expected_clipped, atol=1.0e-5)
    assert not np.allclose(clipped_traces, input_traces, atol=1.0e-5)
    assert direct_meta['pair']['residual_learning'] is False
    assert residual_meta['pair']['residual_learning'] is True
    assert residual_meta['pair']['input_soft_clip_abs'] is None
    assert clipped_meta['pair']['residual_learning'] is True
    assert clipped_meta['pair']['input_soft_clip_abs'] == clip_abs
