from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import yaml
from seisai_engine.pipelines.common import load_checkpoint
from seisai_utils.config import load_config

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_e2e(*, out_dir: Path, waveform_mode: str | None = None) -> tuple[Path, Path]:
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
    if waveform_mode is not None:
        cfg_data.setdefault('dataset', {})
        cfg_data['dataset']['waveform_mode'] = str(waveform_mode)
    cfg_data['paths']['out_dir'] = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_tmp = out_dir / 'config_train_pair.yaml'
    cfg_tmp.write_text(yaml.safe_dump(cfg_data, sort_keys=False))

    m.main(argv=['--config', str(cfg_tmp)])

    png = out_dir / 'vis' / 'epoch_0000' / 'step_0000.png'
    ckpt = out_dir / 'ckpt' / 'best.pt'
    return png, ckpt


@pytest.mark.e2e
def test_e2e_train_pair_one_epoch(tmp_path: Path) -> None:
    out_dir = tmp_path / '_pair_out'
    png, ckpt = _run_e2e(out_dir=out_dir)
    assert png.is_file()
    assert png.stat().st_size > 0
    assert ckpt.is_file()
    assert ckpt.stat().st_size > 0
    ckpt_dict = load_checkpoint(ckpt)
    assert ckpt_dict['version'] == 1
    assert ckpt_dict['pipeline'] == 'pair'
    assert isinstance(ckpt_dict['model_sig'], dict)
    assert isinstance(ckpt_dict['model_state_dict'], dict)
    assert ckpt_dict['epoch'] == 0
    assert ckpt_dict['global_step'] > 0
    model_sig = ckpt_dict['model_sig']
    for key in ('backbone', 'pretrained', 'in_chans', 'out_chans'):
        assert key in model_sig
    assert model_sig['backbone'] == 'resnet18'
    assert model_sig['pretrained'] is False
    assert model_sig['in_chans'] == 1
    assert model_sig['out_chans'] == 1


@pytest.mark.e2e
def test_e2e_train_pair_one_epoch_mmap(tmp_path: Path) -> None:
    out_dir = tmp_path / '_pair_out_mmap'
    png, ckpt = _run_e2e(out_dir=out_dir, waveform_mode='mmap')
    assert png.is_file()
    assert png.stat().st_size > 0
    assert ckpt.is_file()
    assert ckpt.stat().st_size > 0
    ckpt_dict = load_checkpoint(ckpt)
    assert ckpt_dict['version'] == 1
    assert ckpt_dict['pipeline'] == 'pair'
    assert isinstance(ckpt_dict['model_sig'], dict)
    assert isinstance(ckpt_dict['model_state_dict'], dict)
    assert ckpt_dict['epoch'] == 0
    assert ckpt_dict['global_step'] > 0
    model_sig = ckpt_dict['model_sig']
    for key in ('backbone', 'pretrained', 'in_chans', 'out_chans'):
        assert key in model_sig
    assert model_sig['backbone'] == 'resnet18'
    assert model_sig['pretrained'] is False
    assert model_sig['in_chans'] == 1
    assert model_sig['out_chans'] == 1
