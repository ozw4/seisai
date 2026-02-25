from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from seisai_engine.infer.ffid_segy2segy import run_ffid_gather_infer_core_chw


@dataclass(frozen=True)
class _DummyGather:
    trace_indices: np.ndarray
    x_hw: np.ndarray
    ffid: int


class _DummyIterator:
    def __init__(
        self,
        *,
        file_infos: list[dict[str, object]],
        gathers_by_file: dict[int, list[_DummyGather]],
    ) -> None:
        self.file_infos = file_infos
        self._gathers_by_file = gathers_by_file

    def iter_gathers(
        self,
        *,
        file_indices: list[int],
        ffids: list[int] | None = None,
    ):
        ffid_set = None if ffids is None else set(int(v) for v in ffids)
        for file_index in file_indices:
            for gather in self._gathers_by_file.get(int(file_index), []):
                if ffid_set is not None and int(gather.ffid) not in ffid_set:
                    continue
                yield gather


def _make_gather(*, idx: list[int], n_samples: int, ffid: int) -> _DummyGather:
    trace_indices = np.asarray(idx, dtype=np.int64)
    x_hw = np.zeros((int(trace_indices.shape[0]), int(n_samples)), dtype=np.float32)
    return _DummyGather(
        trace_indices=trace_indices,
        x_hw=x_hw,
        ffid=int(ffid),
    )


def test_run_ffid_gather_infer_core_chw_fills_all_traces() -> None:
    iterator = _DummyIterator(
        file_infos=[{'n_traces': 4, 'n_samples': 5, 'path': 'dummy.sgy'}],
        gathers_by_file={
            0: [
                _make_gather(idx=[0, 2], n_samples=5, ffid=10),
                _make_gather(idx=[1, 3], n_samples=5, ffid=20),
            ]
        },
    )

    def infer_one_gather(gather: _DummyGather) -> np.ndarray:
        h = int(gather.trace_indices.shape[0])
        w = int(gather.x_hw.shape[1])
        out = np.empty((3, h, w), dtype=np.float32)
        trace_base = gather.trace_indices.astype(np.float32)[:, None]
        time_base = np.arange(w, dtype=np.float32)[None, :]
        for chan in range(3):
            out[chan] = trace_base * 10.0 + time_base + float(chan * 100)
        return out

    out = run_ffid_gather_infer_core_chw(
        iterator=iterator,
        file_index=0,
        out_chans=3,
        infer_one_gather_fn=infer_one_gather,
    )

    assert out.dtype == np.float32
    assert out.shape == (3, 4, 5)
    assert float(out[0, 3, 4]) == pytest.approx(34.0)
    assert float(out[2, 1, 2]) == pytest.approx(212.0)


def test_run_ffid_gather_infer_core_chw_rejects_shape_mismatch() -> None:
    iterator = _DummyIterator(
        file_infos=[{'n_traces': 2, 'n_samples': 4, 'path': 'dummy.sgy'}],
        gathers_by_file={0: [_make_gather(idx=[0, 1], n_samples=4, ffid=11)]},
    )

    with pytest.raises(ValueError, match='output must be'):
        run_ffid_gather_infer_core_chw(
            iterator=iterator,
            file_index=0,
            out_chans=3,
            infer_one_gather_fn=lambda gather: np.zeros(
                (2, int(gather.trace_indices.shape[0]), int(gather.x_hw.shape[1])),
                dtype=np.float32,
            ),
        )


def test_run_ffid_gather_infer_core_chw_rejects_unfilled_trace() -> None:
    iterator = _DummyIterator(
        file_infos=[{'n_traces': 3, 'n_samples': 4, 'path': 'dummy.sgy'}],
        gathers_by_file={0: [_make_gather(idx=[0, 2], n_samples=4, ffid=15)]},
    )

    with pytest.raises(ValueError, match='some traces were not filled'):
        run_ffid_gather_infer_core_chw(
            iterator=iterator,
            file_index=0,
            out_chans=2,
            infer_one_gather_fn=lambda gather: np.zeros(
                (2, int(gather.trace_indices.shape[0]), int(gather.x_hw.shape[1])),
                dtype=np.float32,
            ),
        )
