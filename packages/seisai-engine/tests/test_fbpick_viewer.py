from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from seisai_engine.viewer.fbpick import (
    render_fbpick_overview,
    save_fbpick_overview_png,
)


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


def test_save_fbpick_overview_png_writes_png(tmp_path: Path) -> None:
    raw_wave_hw = np.linspace(-1.0, 1.0, 4 * 256, dtype=np.float32).reshape(4, 256)
    out_path = tmp_path / 'synthetic.overview.png'

    saved = save_fbpick_overview_png(
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


def test_render_fbpick_overview_draws_inclusive_window_end_contract() -> None:
    raw_wave_hw = np.linspace(-1.0, 1.0, 4 * 256, dtype=np.float32).reshape(4, 256)
    final_payload = _make_final_payload(n_traces=4)

    fig, ax = render_fbpick_overview(
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
    finally:
        plt.close(fig)
