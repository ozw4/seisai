from __future__ import annotations

import importlib
import pkgutil

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
SKIP_MODULES: dict[str, str] = {}


def skip_reason_for(module_name: str) -> str | None:
    parts = module_name.split('.')
    for i in range(len(parts), 0, -1):
        skipped_name = '.'.join(parts[:i])
        if skipped_name in SKIP_MODULES:
            return SKIP_MODULES[skipped_name]
    return None


def collect_package_modules(package_name: str, package_path) -> list[str]:
    modules: list[str] = []

    for module_info in pkgutil.iter_modules(package_path, prefix=f'{package_name}.'):
        modules.append(module_info.name)

        if not module_info.ispkg or skip_reason_for(module_info.name) is not None:
            continue

        package = importlib.import_module(module_info.name)
        if hasattr(package, '__path__'):
            modules.extend(collect_package_modules(module_info.name, package.__path__))

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

    return modules


@pytest.mark.parametrize('module_name', collect_modules())
def test_package_module_import_smoke(module_name: str) -> None:
    skip_reason = skip_reason_for(module_name)
    if skip_reason is not None:
        pytest.skip(skip_reason)

    importlib.import_module(module_name)
