from __future__ import annotations

import importlib
from pathlib import Path

import pytest


PACKAGE_ROOTS = [
    'seisai_utils',
    'seisai_transforms',
    'seisai_pick',
    'seisai_dataset',
    'seisai_models',
    'seisai_engine',
]

# Optional dependency, GPU-heavy module, or intentionally side-effectful module only.
# Each skipped module must have a reason in the value.
SKIP_MODULES: dict[str, str] = {
    'seisai_pick.detection.example': 'example modules are outside import smoke scope',
    'seisai_transforms.signal_ops.example': 'example modules are outside import smoke scope',
}


def skip_reason_for(module_name: str) -> str | None:
    parts = module_name.split('.')
    for i in range(len(parts), 0, -1):
        skipped_name = '.'.join(parts[:i])
        if skipped_name in SKIP_MODULES:
            return SKIP_MODULES[skipped_name]
    return None


def module_name_from_path(package_name: str, package_path: Path, py_file: Path) -> str:
    if py_file.name == '__init__.py':
        relative = py_file.parent.relative_to(package_path)
    else:
        relative = py_file.relative_to(package_path).with_suffix('')

    if not relative.parts:
        return package_name

    return '.'.join((package_name, *relative.parts))


def collect_package_modules(package_name: str, package_paths) -> list[str]:
    modules: list[str] = []

    for package_path_str in package_paths:
        package_path = Path(package_path_str)
        for py_file in package_path.rglob('*.py'):
            modules.append(module_name_from_path(package_name, package_path, py_file))

    return modules


def collect_modules() -> list[str]:
    modules: list[str] = []

    for package_name in PACKAGE_ROOTS:
        modules.append(package_name)

        if skip_reason_for(package_name) is not None:
            continue

        package = importlib.import_module(package_name)
        if hasattr(package, '__path__'):
            modules.extend(collect_package_modules(package_name, package.__path__))

    return sorted(set(modules))


def test_import_smoke_collects_namespace_subpackage_modules() -> None:
    modules = set(collect_modules())

    assert 'seisai_models.ops.channels' in modules
    assert 'seisai_models.nn.blocks' in modules
    assert 'seisai_models.models.encdec2d' in modules
    assert 'seisai_transforms.signal_ops.scaling.standardize' in modules
    assert 'seisai_pick.trend.trend_fit' in modules
    assert 'seisai_engine.postprocess.trend_prior_op' in modules


@pytest.mark.parametrize('module_name', collect_modules())
def test_package_module_import_smoke(module_name: str) -> None:
    skip_reason = skip_reason_for(module_name)
    if skip_reason is not None:
        pytest.skip(skip_reason)

    importlib.import_module(module_name)
