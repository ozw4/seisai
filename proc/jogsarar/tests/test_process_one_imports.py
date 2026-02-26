from __future__ import annotations

import importlib

import pytest

pytest.importorskip('segyio')
pytest.importorskip('seisai_dataset')
pytest.importorskip('seisai_engine')
pytest.importorskip('seisai_models')
pytest.importorskip('seisai_pick')
pytest.importorskip('seisai_utils')


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
