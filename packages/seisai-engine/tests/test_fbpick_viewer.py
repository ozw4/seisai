from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch


def _block_heavy_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    for prefix in ('segyio', 'timm'):
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix + '.'):
                monkeypatch.delitem(sys.modules, name, raising=False)
        monkeypatch.setitem(sys.modules, prefix, None)


def _clear_viewer_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(sys.modules):
        if name == 'seisai_engine.viewer' or name.startswith('seisai_engine.viewer.'):
            monkeypatch.delitem(sys.modules, name, raising=False)


def _import_viewer_package(monkeypatch: pytest.MonkeyPatch):
    _block_heavy_modules(monkeypatch)
    _clear_viewer_modules(monkeypatch)
    return importlib.import_module('seisai_engine.viewer')


def _import_viewer_fbpick(monkeypatch: pytest.MonkeyPatch):
    _block_heavy_modules(monkeypatch)
    _clear_viewer_modules(monkeypatch)
    return importlib.import_module('seisai_engine.viewer.fbpick')


def _make_final_payload(*, n_traces: int) -> dict[str, np.ndarray]:
    coarse_pick_i = np.array([10, 20, 30, 40][:n_traces], dtype=np.int32)
    robust_pick_i = coarse_pick_i + np.array([1, -1, 2, -2][:n_traces], dtype=np.int32)
    window_start_i = robust_pick_i - 128
    window_end_i = window_start_i + 255
    final_pick_i = robust_pick_i + np.array([0, 1, 0, 1][:n_traces], dtype=np.int32)
    high_conf_mask = np.array([True, False, True, False][:n_traces], dtype=np.bool_)
    return {
        'coarse_pick_i': coarse_pick_i,
        'robust_pick_i': robust_pick_i,
        'window_start_i': window_start_i.astype(np.int32),
        'window_end_i': window_end_i.astype(np.int32),
        'final_pick_i': final_pick_i.astype(np.int32),
        'high_conf_mask': high_conf_mask,
    }


def test_viewer_package_import_is_safe_without_segyio_or_timm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_viewer_package(monkeypatch)

    assert module.__name__ == 'seisai_engine.viewer'
    assert 'seisai_engine.viewer.denoise' not in sys.modules
    assert 'seisai_engine.viewer.fbpick' not in sys.modules


def test_viewer_fbpick_import_is_safe_without_segyio_or_timm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_viewer_fbpick(monkeypatch)

    assert callable(module.render_fbpick_overview)
    assert callable(module.save_fbpick_overview_png)


def test_render_fbpick_overview_creates_figure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_viewer_fbpick(monkeypatch)
    raw_wave_hw = np.linspace(-1.0, 1.0, 4 * 256, dtype=np.float32).reshape(4, 256)
    final_payload = _make_final_payload(n_traces=4)

    fig, ax = module.render_fbpick_overview(
        raw_wave_hw,
        final_payload,
        title='synthetic',
        clip_percentile=99.0,
    )
    try:
        assert fig.axes[0] is ax
        assert len(ax.lines) >= 5
        assert len(ax.collections) >= 1
    finally:
        plt.close(fig)


def test_make_debug_title_includes_identifiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_viewer_fbpick(monkeypatch)
    batch = {
        'secondary_key': ['chno'],
        'meta': {
            'file_path': ['/tmp/data/AIRAS_TRCTAB_shinjo22_wolmo.sgy'],
            'key_name': ['ffid'],
            'primary_unique': [5401],
            'trace_valid': torch.ones((1, 4), dtype=torch.bool),
            'fb_idx_view': torch.ones((1, 4), dtype=torch.int64),
        },
    }

    title = module._make_debug_title(batch, b=0)

    assert title == 'AIRAS_TRCTAB_shinjo22_wolmo.sgy\nffid=5401 | secondary=chno'


def test_save_fbpick_overview_png_writes_png(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _import_viewer_fbpick(monkeypatch)
    raw_wave_hw = np.linspace(-1.0, 1.0, 4 * 256, dtype=np.float32).reshape(4, 256)
    out_path = tmp_path / 'synthetic.overview.png'

    saved = module.save_fbpick_overview_png(
        out_path,
        raw_wave_hw=raw_wave_hw,
        final_payload=_make_final_payload(n_traces=4),
        title='synthetic',
        dpi=120,
        clip_percentile=99.0,
    )

    assert saved == out_path.resolve()
    assert out_path.is_file()
    assert out_path.stat().st_size > 0


def test_render_fbpick_overview_draws_inclusive_window_end_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_viewer_fbpick(monkeypatch)
    raw_wave_hw = np.linspace(-1.0, 1.0, 4 * 256, dtype=np.float32).reshape(4, 256)
    final_payload = _make_final_payload(n_traces=4)

    fig, ax = module.render_fbpick_overview(
        raw_wave_hw,
        final_payload,
        title='inclusive-end',
        clip_percentile=99.0,
    )
    try:
        line_by_label = {line.get_label(): line for line in ax.lines}
        assert 'window_end_i' in line_by_label
        np.testing.assert_array_equal(
            line_by_label['window_end_i'].get_ydata(),
            final_payload['window_end_i'],
        )
        np.testing.assert_array_equal(
            final_payload['window_end_i'] - final_payload['window_start_i'],
            np.full((4,), 255, dtype=np.int32),
        )
        assert len(ax.collections) >= 1
    finally:
        plt.close(fig)
