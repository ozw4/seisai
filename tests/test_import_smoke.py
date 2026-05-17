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


def collect_modules() -> list[str]:
	modules: list[str] = []

	for package_name in PACKAGE_ROOTS:
		package = importlib.import_module(package_name)
		modules.append(package_name)

		if not hasattr(package, '__path__'):
			continue

		for module_info in pkgutil.walk_packages(
			package.__path__,
			prefix=f'{package_name}.',
		):
			modules.append(module_info.name)

	return modules


@pytest.mark.parametrize('module_name', collect_modules())
def test_package_module_import_smoke(module_name: str) -> None:
	if module_name in SKIP_MODULES:
		pytest.skip(SKIP_MODULES[module_name])

	importlib.import_module(module_name)
