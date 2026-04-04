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
    out_path.parent.mkdir()

    runtime = SimpleNamespace(
        load_cfg_with_base_dir=lambda path: ({'infer': {'device': 'cpu'}}, tmp_path),
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
        run_coarse_infer=lambda *, model, cfg, device: out_path,
    )
    monkeypatch.setattr(module, '_load_runtime', lambda: runtime)
    monkeypatch.setattr(module, '_resolve_ckpt_path', lambda cfg: tmp_path / 'dummy.pt')
    monkeypatch.setattr(
        module,
        '_validate_checkpoint_for_infer',
        lambda ckpt, *, model_sig: None,
    )

    result = module.run_pipeline(cfg_path)

    assert result == out_path
    assert model.loaded_state_dict == {'weight': 1}
    assert model.device == 'cpu'
    assert capsys.readouterr().out.strip() == str(out_path)


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
