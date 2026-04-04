from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cli.run_fbpick_coarse_infer as coarse_infer_cli
import cli.run_fbpick_coarse_train as coarse_train_cli
import cli.run_fbpick_fine_infer as fine_infer_cli
import cli.run_fbpick_fine_train as fine_train_cli
import cli.run_fbpick_physics as physics_cli
import numpy as np
import pytest
import yaml

from seisai_engine.pipelines.common import load_cfg_with_base_dir
from seisai_engine.pipelines.fbpick.coarse import (
    load_coarse_infer_config,
    load_coarse_train_config,
)
from seisai_engine.pipelines.fbpick.fine import (
    load_fine_infer_config,
    load_fine_train_config,
)
from seisai_engine.pipelines.fbpick.fine.infer import main as run_fine_infer_main
from seisai_engine.pipelines.fbpick.physics.config import load_physics_lite_config

REPO_ROOT = Path(__file__).resolve().parents[3]


class _DummyModel:
    def __init__(self) -> None:
        self.loaded_state_dict = None
        self.device = None

    def load_state_dict(self, state_dict) -> None:
        self.loaded_state_dict = state_dict

    def to(self, device):
        self.device = device
        return self


def _load_example_cfg(name: str) -> tuple[dict, Path]:
    return load_cfg_with_base_dir(REPO_ROOT / 'examples' / name)


@pytest.mark.parametrize(
    ('name', 'loader'),
    [
        ('config_train_fbpick_coarse.yaml', load_coarse_train_config),
        ('config_infer_fbpick_coarse.yaml', load_coarse_infer_config),
        ('config_run_fbpick_physics.yaml', load_physics_lite_config),
        ('config_train_fbpick_fine.yaml', load_fine_train_config),
        ('config_infer_fbpick_fine.yaml', load_fine_infer_config),
    ],
)
def test_fbpick_example_yaml_loaders_accept_step6_examples(name: str, loader) -> None:
    cfg, base_dir = _load_example_cfg(name)

    if loader is load_fine_train_config:
        typed = loader(cfg, base_dir=base_dir)
        assert typed.transform.trace_len == 128
        return

    typed = loader(cfg)
    assert typed is not None
    if name == 'config_infer_fbpick_fine.yaml':
        assert typed.viewer.enabled is True
        assert typed.viewer.save_overview_png is True


def test_run_fbpick_coarse_train_cli_is_thin_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_run_train(config_path) -> None:
        captured['config_path'] = config_path

    monkeypatch.setattr(coarse_train_cli, 'run_train', _fake_run_train)

    coarse_train_cli.main(['--config', 'coarse-train.yaml'])

    assert captured['config_path'] == 'coarse-train.yaml'


def test_run_fbpick_coarse_infer_cli_is_thin_wrapper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg_path = tmp_path / 'config_infer_fbpick_coarse.yaml'
    cfg_path.write_text('placeholder: true\n', encoding='utf-8')
    model = _DummyModel()
    out_path = tmp_path / 'coarse_out' / 'synthetic.coarse.npz'
    out_path.parent.mkdir()

    monkeypatch.setattr(
        coarse_infer_cli,
        'load_cfg_with_base_dir',
        lambda path: ({'infer': {'device': 'cpu'}}, tmp_path),
    )
    monkeypatch.setattr(coarse_infer_cli, '_prepare_cfg', lambda cfg, *, base_dir: cfg)
    monkeypatch.setattr(
        coarse_infer_cli,
        'load_coarse_infer_config',
        lambda cfg: SimpleNamespace(
            model_sig={'backbone': 'resnet18', 'in_chans': 3, 'out_chans': 1}
        ),
    )
    monkeypatch.setattr(coarse_infer_cli, '_resolve_ckpt_path', lambda cfg: tmp_path / 'dummy.pt')
    monkeypatch.setattr(coarse_infer_cli, '_resolve_cli_device', lambda cfg: 'cpu')
    monkeypatch.setattr(
        coarse_infer_cli,
        'load_checkpoint',
        lambda path: {
            'pipeline': 'fbpick',
            'model_sig': {'backbone': 'resnet18', 'in_chans': 3, 'out_chans': 1},
        },
    )
    monkeypatch.setattr(
        coarse_infer_cli,
        '_validate_checkpoint_for_infer',
        lambda ckpt, *, model_sig: None,
    )
    monkeypatch.setattr(coarse_infer_cli, 'build_model', lambda model_sig: model)
    monkeypatch.setattr(
        coarse_infer_cli,
        'select_state_dict',
        lambda ckpt: ({'weight': np.array([1.0], dtype=np.float32)}, False),
    )

    captured: dict[str, object] = {}

    def _fake_run_coarse_infer(*, model, cfg, device):
        captured['model'] = model
        captured['cfg'] = cfg
        captured['device'] = device
        return out_path

    monkeypatch.setattr(coarse_infer_cli, 'run_coarse_infer', _fake_run_coarse_infer)

    result = coarse_infer_cli.run_pipeline(cfg_path)

    assert result == out_path
    assert captured['model'] is model
    assert captured['device'] == 'cpu'
    assert captured['cfg'] == {'infer': {'device': 'cpu'}}
    assert isinstance(model.loaded_state_dict, dict)
    np.testing.assert_array_equal(model.loaded_state_dict['weight'], np.array([1.0], dtype=np.float32))
    assert capsys.readouterr().out.strip() == str(out_path)


def test_run_fbpick_physics_cli_is_thin_wrapper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg_path = tmp_path / 'config_run_fbpick_physics.yaml'
    cfg_path.write_text('placeholder: true\n', encoding='utf-8')
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        physics_cli,
        'load_cfg_with_base_dir',
        lambda path: (
            {
                'paths': {
                    'coarse_npz_path': 'input.coarse.npz',
                    'out_path': 'output.robust.npz',
                }
            },
            tmp_path,
        ),
    )
    monkeypatch.setattr(physics_cli, 'resolve_cfg_paths', lambda cfg, base_dir, keys: cfg)

    def _fake_run_physics_lite(coarse_npz_path, *, cfg, out_path):
        captured['coarse_npz_path'] = coarse_npz_path
        captured['cfg'] = cfg
        captured['out_path'] = out_path
        return Path(out_path)

    monkeypatch.setattr(physics_cli, 'run_physics_lite', _fake_run_physics_lite)

    result = physics_cli.run_pipeline(cfg_path)

    assert result == Path('output.robust.npz')
    assert captured['coarse_npz_path'] == 'input.coarse.npz'
    assert captured['cfg']['paths']['coarse_npz_path'] == 'input.coarse.npz'
    assert captured['out_path'] == 'output.robust.npz'
    assert capsys.readouterr().out.strip() == 'output.robust.npz'


def test_run_fbpick_fine_train_cli_is_thin_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_run_train(config_path) -> None:
        captured['config_path'] = config_path

    monkeypatch.setattr(fine_train_cli, 'run_train', _fake_run_train)

    fine_train_cli.main(['--config', 'fine-train.yaml'])

    assert captured['config_path'] == 'fine-train.yaml'


def test_run_fbpick_fine_infer_cli_is_thin_wrapper_with_overview_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, list[str] | None] = {}

    def _fake_pipeline_main(argv: list[str] | None = None) -> None:
        captured['argv'] = argv

    monkeypatch.setattr(fine_infer_cli, 'pipeline_main', _fake_pipeline_main)

    fine_infer_cli.main(['--config', 'fine-infer.yaml', '--save-overview'])
    assert captured['argv'] == [
        '--config',
        'fine-infer.yaml',
        'viewer.enabled=true',
        'viewer.save_overview_png=true',
    ]

    fine_infer_cli.main(['--config', 'fine-infer.yaml', '--no-save-overview'])
    assert captured['argv'] == [
        '--config',
        'fine-infer.yaml',
        'viewer.save_overview_png=false',
    ]


@pytest.mark.parametrize(
    'main_fn',
    [
        coarse_train_cli.main,
        coarse_infer_cli.main,
        physics_cli.main,
        fine_train_cli.main,
        fine_infer_cli.main,
    ],
)
def test_all_fbpick_cli_entrypoints_require_config(main_fn) -> None:
    with pytest.raises(SystemExit) as exc:
        main_fn([])

    assert exc.value.code == 2


def test_fine_infer_main_calls_overview_save_when_viewer_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_dir = tmp_path / 'fine_out'
    out_dir.mkdir()
    cfg = {
        'paths': {
            'segy_files': ['synthetic.sgy'],
            'robust_npz_files': ['synthetic.robust.npz'],
            'out_dir': str(out_dir),
        },
        'dataset': {
            'use_header_cache': False,
            'verbose': False,
            'progress': False,
            'primary_keys': ['ffid'],
            'waveform_mode': 'eager',
            'train_endian': 'big',
            'infer_endian': 'big',
        },
        'transform': {
            'trace_len': 128,
            'time_len': 256,
            'center_index': 128,
            'standardize_eps': 1.0e-8,
        },
        'infer': {
            'ckpt_path': 'synthetic.pt',
            'device': 'cpu',
            'batch_size': 1,
            'num_workers': 0,
            'overlap_h': 96,
            'amp': False,
            'use_tqdm': False,
            'high_conf_threshold': 0.5,
            'source_model_id': None,
            'iter_id': None,
            'allow_unsafe_override': False,
        },
        'viewer': {
            'enabled': True,
            'save_overview_png': True,
            'dpi': 150,
            'clip_percentile': 99.0,
        },
        'model': {
            'backbone': 'resnet18',
            'pretrained': False,
            'in_chans': 1,
            'out_chans': 1,
        },
    }
    cfg_path = tmp_path / 'config_infer_fbpick_fine.yaml'
    cfg_path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    model = _DummyModel()

    coarse_payload = {
        'dt_sec': np.asarray(0.004, dtype=np.float32),
        'n_samples_orig': np.asarray(512, dtype=np.int32),
        'n_traces': np.asarray(3, dtype=np.int32),
        'ffid_values': np.array([1, 1, 1], dtype=np.int32),
        'chno_values': np.array([1, 2, 3], dtype=np.int32),
        'offsets_m': np.array([10.0, 20.0, 30.0], dtype=np.float32),
        'trace_indices': np.array([0, 1, 2], dtype=np.int64),
        'coarse_pick_i': np.array([120, 121, 122], dtype=np.int32),
        'coarse_pick_t_sec': np.array([0.48, 0.484, 0.488], dtype=np.float32),
        'coarse_pmax': np.array([0.9, 0.9, 0.9], dtype=np.float32),
        'coarse_prob_summary': np.array([0.9, 0.9, 0.9], dtype=np.float32),
        'lineage': np.asarray('{"stage":"coarse"}'),
    }
    robust_payload = {
        'dt_sec': np.asarray(0.004, dtype=np.float32),
        'n_samples_orig': np.asarray(512, dtype=np.int32),
        'n_traces': np.asarray(3, dtype=np.int32),
        'ffid_values': np.array([1, 1, 1], dtype=np.int32),
        'chno_values': np.array([1, 2, 3], dtype=np.int32),
        'offsets_m': np.array([10.0, 20.0, 30.0], dtype=np.float32),
        'trace_indices': np.array([0, 1, 2], dtype=np.int64),
        'robust_pick_i': np.array([120, 121, 122], dtype=np.int32),
        'robust_pick_t_sec': np.array([0.48, 0.484, 0.488], dtype=np.float32),
        'robust_conf': np.array([0.9, 0.8, 0.85], dtype=np.float32),
        'robust_source': np.array([0, 0, 0], dtype=np.uint8),
        'used_theoretical_mask': np.array([False, False, False], dtype=np.bool_),
        'reason_mask': np.array([0, 0, 0], dtype=np.uint8),
        'conf_prob1': np.array([0.9, 0.8, 0.85], dtype=np.float32),
        'conf_trend1': np.array([0.9, 0.8, 0.85], dtype=np.float32),
        'conf_rs1': np.array([0.9, 0.8, 0.85], dtype=np.float32),
        'lineage': np.asarray('{"stage":"robust"}'),
    }
    fine_payload = {
        'dt_sec': np.asarray(0.004, dtype=np.float32),
        'n_samples_orig': np.asarray(512, dtype=np.int32),
        'n_traces': np.asarray(3, dtype=np.int32),
        'trace_indices': np.array([0, 1, 2], dtype=np.int64),
        'center_raw_i': np.array([120, 121, 122], dtype=np.int32),
        'fine_pick_local_i': np.array([128, 128, 128], dtype=np.int32),
        'fine_pick_local_f': np.array([128.0, 128.0, 128.0], dtype=np.float32),
        'fine_pmax': np.array([0.95, 0.85, 0.75], dtype=np.float32),
        'final_pick_i': np.array([120, 121, 122], dtype=np.int32),
        'final_pick_f': np.array([120.0, 121.0, 122.0], dtype=np.float32),
        'final_pick_t_sec': np.array([0.48, 0.484, 0.488], dtype=np.float32),
        'final_conf': np.array([0.95, 0.85, 0.75], dtype=np.float32),
        'window_start_i': np.array([-8, -7, -6], dtype=np.int32),
        'window_end_i': np.array([248, 249, 250], dtype=np.int32),
    }
    overview_calls: dict[str, object] = {}

    monkeypatch.setattr(
        'seisai_engine.pipelines.fbpick.fine.infer.build_merged_cfg',
        lambda **kwargs: kwargs['infer_yaml_cfg'],
    )
    monkeypatch.setattr(
        'seisai_engine.pipelines.fbpick.fine.infer.resolve_ckpt_path',
        lambda cfg, base_dir: tmp_path / 'synthetic.pt',
    )
    monkeypatch.setattr(
        'seisai_engine.pipelines.fbpick.fine.infer.load_checkpoint',
        lambda path: {
            'pipeline': 'fbpick',
            'stage': 'fine',
            'model_sig': cfg['model'],
            'output_ids': ['P'],
            'softmax_axis': 'time',
        },
    )
    monkeypatch.setattr(
        'seisai_engine.pipelines.fbpick.fine.infer.select_state_dict',
        lambda ckpt: ({}, False),
    )
    monkeypatch.setattr(
        'seisai_engine.pipelines.fbpick.fine.infer.build_model',
        lambda model_sig: model,
    )

    def _fake_run_fine_local_infer_impl(*, model, typed, device, raw_wave_hw_out=None):
        if raw_wave_hw_out is not None:
            raw_wave_hw_out['raw_wave_hw'] = np.ones((3, 512), dtype=np.float32)
        return fine_payload

    monkeypatch.setattr(
        'seisai_engine.pipelines.fbpick.fine.infer._run_fine_local_infer_impl',
        _fake_run_fine_local_infer_impl,
    )
    monkeypatch.setattr(
        'seisai_engine.pipelines.fbpick.fine.infer.load_coarse_npz',
        lambda path: coarse_payload,
    )
    monkeypatch.setattr(
        'seisai_engine.pipelines.fbpick.fine.infer.load_robust_npz',
        lambda path: robust_payload,
    )
    monkeypatch.setattr(
        'seisai_engine.pipelines.fbpick.fine.infer.build_lineage_payload',
        lambda *args, **kwargs: np.asarray(
            '{"iter_id":"","source_model_id":"fine-model","cfg_hash":"cfg","git_sha":"sha"}'
        ),
    )

    def _fake_save_overview(out_png, *, raw_wave_hw, final_payload, title, dpi, clip_percentile):
        overview_calls['out_png'] = Path(out_png)
        overview_calls['raw_wave_hw'] = raw_wave_hw
        overview_calls['final_payload'] = final_payload
        overview_calls['title'] = title
        overview_calls['dpi'] = dpi
        overview_calls['clip_percentile'] = clip_percentile
        return Path(out_png)

    monkeypatch.setattr(
        'seisai_engine.viewer.fbpick.save_fbpick_overview_png',
        _fake_save_overview,
    )

    run_fine_infer_main(['--config', str(cfg_path)])

    final_path = out_dir / 'synthetic.fbpick_final.npz'
    assert final_path.is_file()
    assert overview_calls['out_png'] == (out_dir / 'synthetic.overview.png').resolve()
    assert overview_calls['raw_wave_hw'].shape == (3, 512)
    assert overview_calls['title'] == 'synthetic'
    assert overview_calls['dpi'] == 150
    assert overview_calls['clip_percentile'] == pytest.approx(99.0)
    assert capsys.readouterr().out.strip() == str(final_path)
