from __future__ import annotations

import builtins
import importlib
import sys

import pytest

pytest.importorskip('segyio')
pytest.importorskip('seisai_dataset')
pytest.importorskip('seisai_engine')
pytest.importorskip('seisai_models')
pytest.importorskip('seisai_pick')
pytest.importorskip('seisai_utils')


def _block_viz_backend_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ):
        if name == 'matplotlib' or name.startswith('matplotlib.'):
            raise AssertionError(f'matplotlib import is forbidden during module import: {name}')
        if name == 'seisai_utils.viz_wiggle' or name.startswith(
            'seisai_utils.viz_wiggle.'
        ):
            raise AssertionError(
                f'seisai_utils.viz_wiggle import is forbidden during module import: {name}'
            )
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, '__import__', guarded_import)


@pytest.mark.parametrize(
    'module_name',
    (
        'stage1.process_one',
        'stage4.process_one',
        'stage1_fbp_infer_raw',
        'stage4_psn512_infer_to_raw',
    ),
)
def test_target_modules_import_without_viz_backends(
    monkeypatch: pytest.MonkeyPatch, module_name: str
) -> None:
    _block_viz_backend_imports(monkeypatch)

    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
        return
    importlib.import_module(module_name)


def test_stage_process_one_modules_importable() -> None:
    st1p = importlib.import_module('stage1.process_one')
    st2p = importlib.import_module('stage2.process_one')
    st4p = importlib.import_module('stage4.process_one')

    assert hasattr(st1p, 'process_one_segy')
    assert hasattr(st2p, 'process_one_segy')
    assert hasattr(st4p, 'process_one_pair')


def test_script_wrappers_point_to_new_modules() -> None:
    st1 = importlib.import_module('stage1_fbp_infer_raw')
    st2 = importlib.import_module('stage2_make_psn512_windows')
    st4 = importlib.import_module('stage4_psn512_infer_to_raw')

    st1p = importlib.import_module('stage1.process_one')
    st2p = importlib.import_module('stage2.process_one')
    st4p = importlib.import_module('stage4.process_one')

    assert callable(st1.process_one_segy)
    assert callable(st2.process_one_segy)
    assert callable(st4.process_one_pair)

    assert st1._process_one_segy is st1p.process_one_segy
    assert st2._process_one_segy is st2p.process_one_segy
    assert st4._process_one_pair is st4p.process_one_pair
