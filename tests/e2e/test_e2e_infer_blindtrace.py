from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import cli.run_blindtrace_infer as m

from ._infer_e2e_helpers import (
    make_dummy_ckpt,
    make_synthetic_segy,
    read_json,
    segy_text_header_contains,
)


@pytest.mark.e2e
def test_e2e_infer_blindtrace(tmp_path: Path) -> None:
    segy_path = make_synthetic_segy(tmp_path)
    ckpt_path = make_dummy_ckpt(tmp_path, pipeline='blindtrace', in_chans=1, out_chans=1)

    out_dir = tmp_path / '_blindtrace_infer_out'
    out_dir.mkdir(parents=True, exist_ok=True)

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
        # Keep this deterministic/light for CI: no noise.
        'cover': {
            'noise_std': 0.0,
            'offsets': [0],
        },
    }

    cfg_path = tmp_path / 'config_infer_blindtrace.yaml'
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')

    # Exercise unknown override handling.
    m.main(argv=['--config', str(cfg_path), 'tile.tile_w=32'])

    out_segy = out_dir / f'{segy_path.stem}.blindtrace_pred.sgy'
    out_meta = out_segy.with_suffix(out_segy.suffix + '.mlmeta.json')

    assert out_segy.is_file()
    assert out_segy.stat().st_size > 0
    assert out_meta.is_file()
    assert out_meta.stat().st_size > 0

    meta = read_json(out_meta)
    assert meta['pipeline'] == 'blindtrace'
    assert meta['tile']['tile_w'] == 32
    assert meta['note'] == 'e2e'

    assert segy_text_header_contains(out_segy, 'ML pipeline=blindtrace')
