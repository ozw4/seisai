from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import segyio
import yaml
from seisai_engine.pipelines.common import load_checkpoint
from seisai_utils.config import load_config

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _augment_on_cfg() -> dict:
    return {
        'hflip_prob': 1.0,
        'polarity_prob': 1.0,
        'space': {
            'prob': 0.0,
            'factor_range': [0.90, 1.10],
        },
        'time': {
            'prob': 0.0,
            'factor_range': [0.95, 1.05],
        },
        'freq': {
            'prob': 0.0,
            'kinds': ['bandpass', 'lowpass', 'highpass'],
            'band': [0.05, 0.45],
            'width': [0.10, 0.35],
            'roll': 0.02,
            'restandardize': False,
        },
    }


def _assert_ckpt_common(ckpt: Path, *, pipeline: str) -> None:
    assert ckpt.is_file()
    assert ckpt.stat().st_size > 0
    ckpt_dict = load_checkpoint(ckpt)
    assert ckpt_dict['version'] == 1
    assert ckpt_dict['pipeline'] == pipeline
    assert isinstance(ckpt_dict['model_sig'], dict)
    assert isinstance(ckpt_dict['model_state_dict'], dict)
    assert ckpt_dict['epoch'] == 0
    assert ckpt_dict['global_step'] > 0


def _make_fb_file(*, segy_path: Path, fb_path: Path) -> None:
    with segyio.open(segy_path, 'r', ignore_geometry=True) as f:
        tracecount = int(f.tracecount)
    fb = np.full(tracecount, 10, dtype=np.int32)
    np.save(fb_path, fb)


@pytest.mark.e2e
def test_e2e_train_psn_with_augment(tmp_path: Path) -> None:
    repo_root = _repo_root()
    cfg = repo_root / 'tests' / 'e2e' / 'config_train_psn.yaml'
    if not cfg.is_file():
        raise FileNotFoundError(cfg)

    import cli.run_psn_train as m

    cfg_data = load_config(cfg)
    base_dir = cfg.parent

    cfg_data['paths']['segy_files'] = [
        str((base_dir / p).resolve()) for p in cfg_data['paths']['segy_files']
    ]
    cfg_data['paths']['phase_pick_files'] = [
        str((base_dir / p).resolve()) for p in cfg_data['paths']['phase_pick_files']
    ]
    cfg_data['paths']['infer_segy_files'] = [
        str((base_dir / p).resolve()) for p in cfg_data['paths']['infer_segy_files']
    ]
    cfg_data['paths']['infer_phase_pick_files'] = [
        str((base_dir / p).resolve())
        for p in cfg_data['paths']['infer_phase_pick_files']
    ]

    cfg_data['augment'] = _augment_on_cfg()
    cfg_data['train']['samples_per_epoch'] = 2
    cfg_data['infer']['max_batches'] = 1

    out_dir = tmp_path / '_psn_out_aug'
    cfg_data['paths']['out_dir'] = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_tmp = out_dir / 'config_train_psn.yaml'
    cfg_tmp.write_text(yaml.safe_dump(cfg_data, sort_keys=False))

    from seisai_engine.pipelines.psn.build_dataset import (
        build_infer_transform,
        build_train_transform,
    )
    from seisai_transforms import RandomPolarityFlip

    train_tf = build_train_transform(cfg_data)
    infer_tf = build_infer_transform(cfg_data)
    pol_ops = [op for op in train_tf.ops if isinstance(op, RandomPolarityFlip)]
    assert len(pol_ops) == 1
    assert float(pol_ops[0].prob) == 1.0
    assert not any(isinstance(op, RandomPolarityFlip) for op in infer_tf.ops)

    m.main(argv=['--config', str(cfg_tmp)])

    png = out_dir / 'vis' / 'epoch_0000' / 'step_0000.png'
    ckpt = out_dir / 'ckpt' / 'best.pt'
    assert png.is_file()
    assert png.stat().st_size > 0
    _assert_ckpt_common(ckpt, pipeline='psn')


@pytest.mark.e2e
def test_e2e_train_blindtrace_with_augment(tmp_path: Path) -> None:
    repo_root = _repo_root()
    cfg = repo_root / 'tests' / 'e2e' / 'config_train_blindtrace.yaml'
    if not cfg.is_file():
        raise FileNotFoundError(cfg)

    import cli.run_blindtrace_train as m

    cfg_data = load_config(cfg)
    base_dir = cfg.parent
    segy_paths = [
        str((base_dir / p).resolve()) for p in cfg_data['paths']['segy_files']
    ]
    if len(segy_paths) != 1:
        msg = 'expected one segy file in test config'
        raise ValueError(msg)
    segy_path = Path(segy_paths[0])

    out_dir = tmp_path / '_blindtrace_out_aug'
    out_dir.mkdir(parents=True, exist_ok=True)
    fb_path = out_dir / 'fb.npy'
    _make_fb_file(segy_path=segy_path, fb_path=fb_path)

    cfg_data['paths']['segy_files'] = [str(segy_path)]
    cfg_data['paths']['phase_pick_files'] = [str(fb_path)]
    cfg_data['paths']['infer_segy_files'] = [str(segy_path)]
    cfg_data['paths']['infer_phase_pick_files'] = [str(fb_path)]

    cfg_data['augment'] = _augment_on_cfg()
    cfg_data['train']['samples_per_epoch'] = 2
    cfg_data['infer']['max_batches'] = 1

    cfg_data['paths']['out_dir'] = str(out_dir)
    cfg_tmp = out_dir / 'config_train_blindtrace.yaml'
    cfg_tmp.write_text(yaml.safe_dump(cfg_data, sort_keys=False))

    from seisai_engine.pipelines.blindtrace.build_dataset import (
        build_infer_transform,
        build_train_transform,
    )
    from seisai_transforms import RandomPolarityFlip

    time_len = int(cfg_data['transform']['time_len'])
    per_trace_standardize = bool(cfg_data['transform']['per_trace_standardize'])
    train_tf = build_train_transform(
        time_len=time_len,
        per_trace_standardize=per_trace_standardize,
        augment_cfg=cfg_data.get('augment'),
    )
    infer_tf = build_infer_transform(
        time_len=time_len,
        per_trace_standardize=per_trace_standardize,
    )
    pol_ops = [op for op in train_tf.ops if isinstance(op, RandomPolarityFlip)]
    assert len(pol_ops) == 1
    assert float(pol_ops[0].prob) == 1.0
    assert not any(isinstance(op, RandomPolarityFlip) for op in infer_tf.ops)

    m.main(argv=['--config', str(cfg_tmp)])

    png = out_dir / 'vis' / 'epoch_0000' / 'step_0000.png'
    ckpt = out_dir / 'ckpt' / 'best.pt'
    assert png.is_file()
    assert png.stat().st_size > 0
    _assert_ckpt_common(ckpt, pipeline='blindtrace')


@pytest.mark.e2e
def test_e2e_train_pair_with_augment(tmp_path: Path) -> None:
    repo_root = _repo_root()
    cfg = repo_root / 'tests' / 'e2e' / 'config_train_pair.yaml'
    if not cfg.is_file():
        raise FileNotFoundError(cfg)

    import cli.run_pair_train as m

    cfg_data = load_config(cfg)
    base_dir = cfg.parent

    cfg_data['paths']['input_segy_files'] = [
        str((base_dir / p).resolve()) for p in cfg_data['paths']['input_segy_files']
    ]
    cfg_data['paths']['target_segy_files'] = [
        str((base_dir / p).resolve()) for p in cfg_data['paths']['target_segy_files']
    ]
    cfg_data['paths']['infer_input_segy_files'] = [
        str((base_dir / p).resolve())
        for p in cfg_data['paths']['infer_input_segy_files']
    ]
    cfg_data['paths']['infer_target_segy_files'] = [
        str((base_dir / p).resolve())
        for p in cfg_data['paths']['infer_target_segy_files']
    ]

    cfg_data['augment'] = _augment_on_cfg()
    cfg_data['train']['samples_per_epoch'] = 2
    cfg_data['infer']['max_batches'] = 1

    out_dir = tmp_path / '_pair_out_aug'
    cfg_data['paths']['out_dir'] = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_tmp = out_dir / 'config_train_pair.yaml'
    cfg_tmp.write_text(yaml.safe_dump(cfg_data, sort_keys=False))

    from seisai_engine.pipelines.pair.build_dataset import (
        build_infer_transform,
        build_train_transform,
    )
    from seisai_transforms import RandomPolarityFlip

    time_len = int(cfg_data['transform']['time_len'])
    train_tf = build_train_transform(
        time_len=time_len,
        augment_cfg=cfg_data.get('augment'),
    )
    infer_tf = build_infer_transform()
    pol_ops = [op for op in train_tf.ops if isinstance(op, RandomPolarityFlip)]
    assert len(pol_ops) == 1
    assert float(pol_ops[0].prob) == 1.0
    assert not any(isinstance(op, RandomPolarityFlip) for op in infer_tf.ops)

    m.main(argv=['--config', str(cfg_tmp)])

    png = out_dir / 'vis' / 'epoch_0000' / 'step_0000.png'
    ckpt = out_dir / 'ckpt' / 'best.pt'
    assert png.is_file()
    assert png.stat().st_size > 0
    _assert_ckpt_common(ckpt, pipeline='pair')
