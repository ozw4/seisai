from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CLI_DIR = REPO_ROOT / 'cli'
_MODULE_COUNTER = itertools.count()


class _DummyModel:
    def __init__(self) -> None:
        self.loaded_state_dict = None
        self.device = None

    def load_state_dict(self, state_dict) -> None:
        self.loaded_state_dict = state_dict

    def to(self, device):
        self.device = device
        return self


def _block_heavy_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    for prefix in ('segyio', 'timm'):
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix + '.'):
                monkeypatch.delitem(sys.modules, name, raising=False)
        monkeypatch.setitem(sys.modules, prefix, None)


def _load_module_from_path(path: Path):
    module_name = f'_fbpick_cli_{path.stem}_{next(_MODULE_COUNTER)}'
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'failed to load spec for {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_cli_module(file_name: str, monkeypatch: pytest.MonkeyPatch):
    _block_heavy_modules(monkeypatch)
    return _load_module_from_path(CLI_DIR / file_name)


@pytest.mark.parametrize(
    'file_name',
    [
        'run_fbpick_coarse_train.py',
        'run_fbpick_coarse_infer.py',
        'run_fbpick_physics.py',
        'run_fbpick_physics_batch.py',
        'run_fbpick_fine_train.py',
        'run_fbpick_fine_infer.py',
    ],
)
def test_cli_modules_import_without_segyio_or_timm(
    monkeypatch: pytest.MonkeyPatch,
    file_name: str,
) -> None:
    module = _load_cli_module(file_name, monkeypatch)

    assert callable(module.main)


@pytest.mark.parametrize(
    ('file_name', 'entrypoint_name'),
    [
        ('run_fbpick_coarse_train.py', 'main'),
        ('run_fbpick_coarse_infer.py', 'main'),
        ('run_fbpick_physics.py', 'main'),
        ('run_fbpick_physics_batch.py', 'main'),
        ('run_fbpick_fine_train.py', 'main'),
        ('run_fbpick_fine_infer.py', 'main'),
    ],
)
def test_all_fbpick_cli_entrypoints_require_config(
    monkeypatch: pytest.MonkeyPatch,
    file_name: str,
    entrypoint_name: str,
) -> None:
    module = _load_cli_module(file_name, monkeypatch)
    main_fn = getattr(module, entrypoint_name)

    with pytest.raises(SystemExit) as exc:
        main_fn([])

    assert exc.value.code == 2


def test_run_fbpick_coarse_train_cli_is_thin_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_cli_module('run_fbpick_coarse_train.py', monkeypatch)
    captured: dict[str, object] = {}

    def _fake_run_train(config_path) -> None:
        captured['config_path'] = config_path

    monkeypatch.setattr(module, '_load_run_train', lambda: _fake_run_train)

    module.main(['--config', 'coarse-train.yaml'])

    assert captured['config_path'] == 'coarse-train.yaml'


def test_run_fbpick_coarse_infer_cli_is_thin_wrapper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_cli_module('run_fbpick_coarse_infer.py', monkeypatch)
    cfg_path = tmp_path / 'config_infer_fbpick_coarse.yaml'
    cfg_path.write_text('placeholder: true\n', encoding='utf-8')
    model = _DummyModel()
    out_path = tmp_path / 'coarse_out' / 'synthetic.coarse.npz'
    expected_out_path = tmp_path / 'coarse_out' / 'site54__synthetic.coarse.npz'
    out_path.parent.mkdir()

    def _fake_run_coarse_infer(*, model, cfg, device, ckpt):
        out_path.touch()
        return out_path

    runtime = SimpleNamespace(
        load_cfg_with_base_dir=lambda path: (
            {
                'paths': {'segy_files': ['site54/synthetic.sgy'], 'out_dir': str(out_path.parent)},
                'infer': {'device': 'cpu'},
            },
            tmp_path,
        ),
        expand_cfg_listfiles=lambda cfg, *, keys: None,
        resolve_cfg_paths=lambda cfg, base_dir, *, keys: None,
        load_coarse_infer_config=lambda cfg: SimpleNamespace(
            model_sig={'backbone': 'resnet18', 'in_chans': 3, 'out_chans': 1}
        ),
        load_checkpoint=lambda path: {
            'pipeline': 'fbpick',
            'model_sig': {'backbone': 'resnet18', 'in_chans': 3, 'out_chans': 1},
        },
        resolve_device=lambda device_raw: 'cpu',
        build_model=lambda model_sig: model,
        select_state_dict=lambda ckpt: ({'weight': 1}, False),
        run_coarse_infer=_fake_run_coarse_infer,
    )
    monkeypatch.setattr(module, '_load_runtime', lambda: runtime)
    monkeypatch.setattr(module, '_resolve_ckpt_path', lambda cfg: tmp_path / 'dummy.pt')
    monkeypatch.setattr(
        module,
        '_validate_checkpoint_for_infer',
        lambda ckpt, *, model_sig: None,
    )

    result = module.run_pipeline(cfg_path)

    assert result == expected_out_path
    assert model.loaded_state_dict == {'weight': 1}
    assert model.device == 'cpu'
    assert not out_path.exists()
    assert expected_out_path.exists()
    assert capsys.readouterr().out.strip() == str(expected_out_path)


def test_run_fbpick_coarse_infer_rejects_legacy_ckpt_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_cli_module('run_fbpick_coarse_infer.py', monkeypatch)
    ckpt = {
        'pipeline': 'fbpick',
        'model_sig': {'backbone': 'resnet18', 'in_chans': 3, 'out_chans': 1},
        'output_ids': ['P'],
        'softmax_axis': 'time',
    }

    with pytest.raises(ValueError) as exc:
        module._validate_checkpoint_for_infer(
            ckpt,
            model_sig={'backbone': 'resnet18', 'in_chans': 3, 'out_chans': 1},
        )

    assert "expected coarse_input_mode='global_anchor_resize', got None" in str(
        exc.value
    )
    assert 'legacy tiled coarse pipeline' in str(exc.value)


def test_run_fbpick_coarse_infer_accepts_global_anchor_ckpt_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_cli_module('run_fbpick_coarse_infer.py', monkeypatch)
    ckpt = {
        'pipeline': 'fbpick',
        'model_sig': {'backbone': 'resnet18', 'in_chans': 3, 'out_chans': 1},
        'output_ids': ['P'],
        'softmax_axis': 'time',
        'coarse_input_mode': 'global_anchor_resize',
        'coarse_trace_len': 256,
        'coarse_time_len': 2048,
        'coarse_in_chans': 3,
        'coarse_input_channels': ['waveform', 'offset_ch', 'time_ch'],
    }

    module._validate_checkpoint_for_infer(
        ckpt,
        model_sig={'backbone': 'resnet18', 'in_chans': 3, 'out_chans': 1},
    )


def test_run_fbpick_physics_cli_is_thin_wrapper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_cli_module('run_fbpick_physics.py', monkeypatch)
    cfg_path = tmp_path / 'config_run_fbpick_physics.yaml'
    cfg_path.write_text('placeholder: true\n', encoding='utf-8')
    captured: dict[str, object] = {}

    def _fake_run_physics_lite(coarse_npz_path, *, cfg, out_path):
        captured['coarse_npz_path'] = coarse_npz_path
        captured['cfg'] = cfg
        captured['out_path'] = out_path
        return Path(out_path)

    runtime = SimpleNamespace(
        load_cfg_with_base_dir=lambda path: (
            {
                'paths': {
                    'coarse_npz_path': 'input.coarse.npz',
                    'out_path': 'output.robust.npz',
                }
            },
            tmp_path,
        ),
        resolve_cfg_paths=lambda cfg, base_dir, *, keys: None,
        run_physics_lite=_fake_run_physics_lite,
    )
    monkeypatch.setattr(module, '_load_runtime', lambda: runtime)

    result = module.run_pipeline(cfg_path)

    assert result == Path('output.robust.npz')
    assert captured['coarse_npz_path'] == 'input.coarse.npz'
    assert captured['cfg']['paths']['coarse_npz_path'] == 'input.coarse.npz'
    assert captured['out_path'] == 'output.robust.npz'
    assert capsys.readouterr().out.strip() == 'output.robust.npz'


def test_run_fbpick_physics_batch_cli_loops_over_multiple_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_cli_module('run_fbpick_physics_batch.py', monkeypatch)
    cfg_path = tmp_path / 'config_run_fbpick_physics_batch.yaml'
    cfg_path.write_text('placeholder: true\n', encoding='utf-8')
    coarse_dir = tmp_path / 'coarse_out'
    out_dir = tmp_path / 'physics_out'
    coarse_dir.mkdir()
    captured_calls: list[dict[str, object]] = []
    coarse_paths = [
        coarse_dir / 'site54__survey_a.coarse.npz',
        coarse_dir / 'site55__survey_b.coarse.npz',
    ]
    out_paths = [
        out_dir / 'site54__survey_a.robust.npz',
        out_dir / 'site55__survey_b.robust.npz',
    ]
    for coarse_path in coarse_paths:
        coarse_path.touch()

    def _fake_run_physics_lite(coarse_npz_path, *, cfg, out_path):
        captured_calls.append(
            {
                'coarse_npz_path': coarse_npz_path,
                'cfg': cfg,
                'out_path': out_path,
            }
        )
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    runtime = SimpleNamespace(
        load_cfg_with_base_dir=lambda path: (
            {
                'paths': {
                    'segy_files': ['site54/survey_a.sgy', 'site55/survey_b.sgy'],
                    'coarse_npz_dir': str(coarse_dir),
                    'out_dir': str(out_dir),
                }
            },
            tmp_path,
        ),
        expand_cfg_listfiles=lambda cfg, *, keys: None,
        resolve_cfg_paths=lambda cfg, base_dir, *, keys: None,
        run_physics_lite=_fake_run_physics_lite,
    )
    monkeypatch.setattr(module, '_load_runtime', lambda: runtime)

    result = module.run_pipeline(cfg_path)

    assert result == out_paths[-1]
    assert [call['coarse_npz_path'] for call in captured_calls] == [
        str(coarse_paths[0]),
        str(coarse_paths[1]),
    ]
    assert [call['out_path'] for call in captured_calls] == [
        str(out_paths[0]),
        str(out_paths[1]),
    ]
    assert [call['cfg']['paths']['segy_files'] for call in captured_calls] == [
        ['site54/survey_a.sgy'],
        ['site55/survey_b.sgy'],
    ]
    assert [call['cfg']['paths']['coarse_npz_path'] for call in captured_calls] == [
        str(coarse_paths[0]),
        str(coarse_paths[1]),
    ]
    assert [call['cfg']['paths']['out_path'] for call in captured_calls] == [
        str(out_paths[0]),
        str(out_paths[1]),
    ]
    assert capsys.readouterr().out.strip().splitlines() == [
        str(out_paths[0]),
        str(out_paths[1]),
    ]


def test_run_fbpick_fine_train_cli_is_thin_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_cli_module('run_fbpick_fine_train.py', monkeypatch)
    captured: dict[str, object] = {}

    def _fake_run_train(config_path) -> None:
        captured['config_path'] = config_path

    monkeypatch.setattr(module, '_load_run_train', lambda: _fake_run_train)

    module.main(['--config', 'fine-train.yaml'])

    assert captured['config_path'] == 'fine-train.yaml'


def test_run_fbpick_fine_infer_cli_is_thin_wrapper_with_overview_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_cli_module('run_fbpick_fine_infer.py', monkeypatch)
    captured: dict[str, list[str] | None] = {}

    def _fake_pipeline_main(argv: list[str] | None = None) -> None:
        captured['argv'] = argv

    monkeypatch.setattr(module, '_load_pipeline_main', lambda: _fake_pipeline_main)

    module.main(['--config', 'fine-infer.yaml', '--save-overview'])
    assert captured['argv'] == [
        '--config',
        'fine-infer.yaml',
        'viewer.enabled=true',
        'viewer.save_overview_png=true',
    ]

    module.main(['--config', 'fine-infer.yaml', '--no-save-overview'])
    assert captured['argv'] == [
        '--config',
        'fine-infer.yaml',
        'viewer.save_overview_png=false',
    ]


def test_run_fbpick_coarse_infer_cli_loops_over_multiple_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_cli_module('run_fbpick_coarse_infer.py', monkeypatch)
    cfg_path = tmp_path / 'config_infer_fbpick_coarse_multi.yaml'
    cfg_path.write_text('placeholder: true\n', encoding='utf-8')
    model = _DummyModel()
    captured_cfgs: list[dict[str, object]] = []
    out_paths = [
        tmp_path / 'coarse_out' / 'survey_a.coarse.npz',
        tmp_path / 'coarse_out' / 'survey_b.coarse.npz',
    ]
    expected_out_paths = [
        tmp_path / 'coarse_out' / 'site54__survey_a.coarse.npz',
        tmp_path / 'coarse_out' / 'site55__survey_b.coarse.npz',
    ]
    out_paths[0].parent.mkdir()

    def _fake_run_coarse_infer(*, model, cfg, device, ckpt):
        captured_cfgs.append(cfg)
        out_path = out_paths[len(captured_cfgs) - 1]
        out_path.touch()
        return out_path

    runtime = SimpleNamespace(
        load_cfg_with_base_dir=lambda path: (
            {
                'paths': {
                    'segy_files': ['site54/survey_a.sgy', 'site55/survey_b.sgy'],
                    'out_dir': str(out_paths[0].parent),
                },
                'infer': {'device': 'cpu'},
            },
            tmp_path,
        ),
        expand_cfg_listfiles=lambda cfg, *, keys: None,
        resolve_cfg_paths=lambda cfg, base_dir, *, keys: None,
        load_coarse_infer_config=lambda cfg: SimpleNamespace(
            model_sig={'backbone': 'resnet18', 'in_chans': 3, 'out_chans': 1}
        ),
        load_checkpoint=lambda path: {
            'pipeline': 'fbpick',
            'model_sig': {'backbone': 'resnet18', 'in_chans': 3, 'out_chans': 1},
        },
        resolve_device=lambda device_raw: 'cpu',
        build_model=lambda model_sig: model,
        select_state_dict=lambda ckpt: ({'weight': 1}, False),
        run_coarse_infer=_fake_run_coarse_infer,
    )
    monkeypatch.setattr(module, '_load_runtime', lambda: runtime)
    monkeypatch.setattr(module, '_resolve_ckpt_path', lambda cfg: tmp_path / 'dummy.pt')
    monkeypatch.setattr(
        module,
        '_validate_checkpoint_for_infer',
        lambda ckpt, *, model_sig: None,
    )

    result = module.run_pipeline(cfg_path)

    assert result == expected_out_paths[-1]
    assert model.loaded_state_dict == {'weight': 1}
    assert model.device == 'cpu'
    assert [cfg['paths']['segy_files'] for cfg in captured_cfgs] == [
        ['site54/survey_a.sgy'],
        ['site55/survey_b.sgy'],
    ]
    assert not out_paths[0].exists()
    assert not out_paths[1].exists()
    assert expected_out_paths[0].exists()
    assert expected_out_paths[1].exists()
    assert capsys.readouterr().out.strip().splitlines() == [
        str(expected_out_paths[0]),
        str(expected_out_paths[1]),
    ]
