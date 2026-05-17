import importlib


def test_kernels_all_symbols_exist() -> None:
    mod = importlib.import_module('seisai_transforms.kernels')
    missing = [name for name in mod.__all__ if not hasattr(mod, name)]
    assert missing == []


def test_kernels_star_import_does_not_fail() -> None:
    namespace: dict[str, object] = {}
    exec('from seisai_transforms.kernels import *', namespace)

