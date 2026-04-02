from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TRAIN_CASES = [
    (
        'cli.run_fbpick_coarse_train',
        'seisai_engine.pipelines.fbpick.coarse.train',
        REPO_ROOT / 'examples' / 'config_train_fbpick_coarse.yaml',
    ),
    (
        'cli.run_fbpick_fine_train',
        'seisai_engine.pipelines.fbpick.fine.train',
        REPO_ROOT / 'examples' / 'config_train_fbpick_fine.yaml',
    ),
]

INFER_CASES = [
    (
        'cli.run_fbpick_coarse_infer',
        'seisai_engine.pipelines.fbpick.coarse.infer_segy2npz',
    ),
    (
        'cli.run_fbpick_fine_infer',
        'seisai_engine.pipelines.fbpick.fine.infer_from_coarse',
    ),
]


def _install_module_chain(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
) -> types.ModuleType:
    parts = module_name.split('.')
    for idx in range(1, len(parts) + 1):
        name = '.'.join(parts[:idx])
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            if idx != len(parts):
                module.__path__ = []
            monkeypatch.setitem(sys.modules, name, module)
        if idx > 1:
            parent = sys.modules['.'.join(parts[: idx - 1])]
            setattr(parent, parts[idx - 1], module)
    return sys.modules[module_name]


def _import_cli_module(
    monkeypatch: pytest.MonkeyPatch,
    cli_module_name: str,
    pipeline_module_name: str,
):
    monkeypatch.delitem(sys.modules, cli_module_name, raising=False)
    fake_pipeline_module = _install_module_chain(monkeypatch, pipeline_module_name)

    def _unexpected_pipeline_main(*, argv: list[str] | None = None) -> None:
        raise AssertionError(f'unexpected downstream call: {argv!r}')

    fake_pipeline_module.main = _unexpected_pipeline_main
    return importlib.import_module(cli_module_name)


@pytest.mark.parametrize(
    ('module_name', 'pipeline_module_name', 'expected_default_config'),
    TRAIN_CASES,
)
def test_train_entrypoint_delegates_to_shared_helper(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    pipeline_module_name: str,
    expected_default_config: Path,
) -> None:
    module = _import_cli_module(monkeypatch, module_name, pipeline_module_name)
    calls: dict[str, object] = {}
    argv = ['--config', 'custom.yaml', 'train.max_epochs=2']

    def _fake_helper(
        *,
        default_config_path: Path,
        pipeline_main,
        argv: list[str] | None = None,
    ) -> None:
        calls['default_config_path'] = default_config_path
        calls['pipeline_main'] = pipeline_main
        calls['argv'] = argv

    monkeypatch.setattr(module, 'run_pipeline_train_entrypoint', _fake_helper)

    module.main(argv=argv)

    assert calls['default_config_path'] == expected_default_config
    assert calls['pipeline_main'] is module.pipeline_main
    assert calls['argv'] is argv


@pytest.mark.parametrize(
    ('module_name', 'pipeline_module_name'),
    [(case[0], case[1]) for case in TRAIN_CASES],
)
def test_train_entrypoint_forwards_explicit_config_and_unknown_args(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    pipeline_module_name: str,
) -> None:
    module = _import_cli_module(monkeypatch, module_name, pipeline_module_name)
    calls: dict[str, object] = {}

    def _fake_pipeline_main(*, argv: list[str] | None = None) -> None:
        calls['argv'] = argv

    monkeypatch.setattr(module, 'pipeline_main', _fake_pipeline_main)

    module.main(
        argv=[
            '--config',
            'custom.yaml',
            'train.max_epochs=2',
            '--trainer.fast_dev_run',
            '1',
        ]
    )

    assert calls['argv'] == [
        '--config',
        'custom.yaml',
        'train.max_epochs=2',
        '--trainer.fast_dev_run',
        '1',
    ]


@pytest.mark.parametrize(
    ('module_name', 'pipeline_module_name', 'expected_default_config'),
    TRAIN_CASES,
)
def test_train_entrypoint_uses_default_config_path_when_config_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    pipeline_module_name: str,
    expected_default_config: Path,
) -> None:
    module = _import_cli_module(monkeypatch, module_name, pipeline_module_name)
    calls: dict[str, object] = {}

    def _fake_pipeline_main(*, argv: list[str] | None = None) -> None:
        calls['argv'] = argv

    monkeypatch.setattr(module, 'pipeline_main', _fake_pipeline_main)

    module.main(argv=['train.max_epochs=2'])

    assert calls['argv'] == [
        '--config',
        str(expected_default_config),
        'train.max_epochs=2',
    ]


@pytest.mark.parametrize(('module_name', 'pipeline_module_name'), INFER_CASES)
def test_infer_entrypoint_passes_argv_through_without_parsing(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    pipeline_module_name: str,
) -> None:
    module = _import_cli_module(monkeypatch, module_name, pipeline_module_name)
    calls: dict[str, object] = {}
    argv = [
        '--config',
        'custom.yaml',
        '--custom-flag',
        'value',
        'fine.window=9',
    ]

    def _fake_pipeline_main(*, argv: list[str] | None = None) -> None:
        calls['argv'] = argv

    monkeypatch.setattr(module, 'pipeline_main', _fake_pipeline_main)

    module.main(argv=argv)

    assert calls['argv'] is argv
