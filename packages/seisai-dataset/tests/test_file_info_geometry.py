from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import segyio
from seisai_dataset.file_info import build_file_info, build_file_info_dataclass
from seisai_dataset.geometry_headers import (
    GEOMETRY_ARRAY_KEYS,
    apply_source_group_scalar,
)


def write_geometry_segy(
    path: str,
    *,
    source_x: np.ndarray,
    source_y: np.ndarray,
    receiver_x: np.ndarray,
    receiver_y: np.ndarray,
    scalar: np.ndarray,
    offsets: np.ndarray,
    dt_us: int = 2000,
) -> np.ndarray:
    sx = np.asarray(source_x, dtype=np.int32)
    sy = np.asarray(source_y, dtype=np.int32)
    gx = np.asarray(receiver_x, dtype=np.int32)
    gy = np.asarray(receiver_y, dtype=np.int32)
    scal = np.asarray(scalar, dtype=np.int32)
    off = np.asarray(offsets, dtype=np.int32)
    n_traces = int(sx.size)
    if not all(arr.shape == (n_traces,) for arr in (sy, gx, gy, scal, off)):
        msg = 'all header arrays must have shape (n_traces,)'
        raise ValueError(msg)
    n_samples = 16
    traces = np.stack(
        [np.linspace(0.0, 1.0, n_samples, dtype=np.float32) + i for i in range(n_traces)],
        axis=0,
    )

    spec = segyio.spec()
    spec.iline = 189
    spec.xline = 193
    spec.format = 5
    spec.sorting = 2
    spec.samples = np.arange(n_samples, dtype=np.int32)
    spec.tracecount = n_traces

    with segyio.create(path, spec) as f:
        f.bin[segyio.BinField.Interval] = int(dt_us)
        for i in range(n_traces):
            f.header[i] = {
                segyio.TraceField.FieldRecord: 1,
                segyio.TraceField.TraceNumber: int(i + 1),
                segyio.TraceField.CDP: 1,
                segyio.TraceField.offset: int(off[i]),
                segyio.TraceField.SourceX: int(sx[i]),
                segyio.TraceField.SourceY: int(sy[i]),
                segyio.TraceField.GroupX: int(gx[i]),
                segyio.TraceField.GroupY: int(gy[i]),
                segyio.TraceField.SourceGroupScalar: int(scal[i]),
            }
            f.trace[i] = traces[i]
    return traces


def close_info(info: dict) -> None:
    segy_obj = info.get('segy_obj')
    if segy_obj is not None:
        segy_obj.close()


def test_apply_source_group_scalar_uses_segy_convention() -> None:
    values = np.asarray([10.0, 20.0, 30.0], dtype=np.float32)
    scalar = np.asarray([2.0, -2.0, 0.0], dtype=np.float32)

    out = apply_source_group_scalar(values, scalar)

    np.testing.assert_allclose(out, np.asarray([20.0, 10.0, 30.0]))


def test_build_file_info_includes_geometry_arrays_and_preserves_header_offsets(
    tmp_path: Path,
) -> None:
    segy_path = str(tmp_path / 'geometry.sgy')
    write_geometry_segy(
        segy_path,
        source_x=np.asarray([0, 0, 10], dtype=np.int32),
        source_y=np.asarray([0, 0, 20], dtype=np.int32),
        receiver_x=np.asarray([3, 6, 10], dtype=np.int32),
        receiver_y=np.asarray([4, 8, 30], dtype=np.int32),
        scalar=np.asarray([1, 1, 1], dtype=np.int32),
        offsets=np.asarray([101, 102, 103], dtype=np.int32),
    )

    info = build_file_info(
        segy_path,
        ffid_byte=segyio.TraceField.FieldRecord,
        chno_byte=segyio.TraceField.TraceNumber,
        cmp_byte=segyio.TraceField.CDP,
        use_header_cache=False,
        include_geometry_arrays=True,
    )
    try:
        np.testing.assert_array_equal(
            info['offsets'],
            np.asarray([101, 102, 103], dtype=np.float32),
        )
        np.testing.assert_allclose(
            info['offset_abs_geom_m'],
            np.asarray([5.0, 10.0, 10.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            info['geometry_valid_mask'],
            np.asarray([True, True, True], dtype=np.bool_),
        )
        assert info['source_x_m'].dtype == np.float32
        assert info['geometry_valid_mask'].dtype == np.bool_
    finally:
        close_info(info)


def test_build_file_info_dataclass_geometry_fields_default_to_none(
    tmp_path: Path,
) -> None:
    segy_path = str(tmp_path / 'default_geometry.sgy')
    write_geometry_segy(
        segy_path,
        source_x=np.asarray([0, 0], dtype=np.int32),
        source_y=np.asarray([0, 0], dtype=np.int32),
        receiver_x=np.asarray([10, 20], dtype=np.int32),
        receiver_y=np.asarray([0, 0], dtype=np.int32),
        scalar=np.asarray([1, 1], dtype=np.int32),
        offsets=np.asarray([10, 20], dtype=np.int32),
    )

    info = build_file_info_dataclass(
        segy_path,
        ffid_byte=segyio.TraceField.FieldRecord,
        chno_byte=segyio.TraceField.TraceNumber,
        cmp_byte=segyio.TraceField.CDP,
        use_header_cache=False,
    )
    try:
        assert info.source_x_m is None
        assert info.offset_abs_geom_m is None
        assert info.geometry_valid_mask is None
    finally:
        if info.segy_obj is not None:
            info.segy_obj.close()


def test_build_file_info_geometry_failure_returns_nan_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from seisai_dataset import file_info as file_info_mod

    segy_path = str(tmp_path / 'geometry_failure.sgy')
    write_geometry_segy(
        segy_path,
        source_x=np.asarray([0, 0], dtype=np.int32),
        source_y=np.asarray([0, 0], dtype=np.int32),
        receiver_x=np.asarray([10, 20], dtype=np.int32),
        receiver_y=np.asarray([0, 0], dtype=np.int32),
        scalar=np.asarray([1, 1], dtype=np.int32),
        offsets=np.asarray([10, 20], dtype=np.int32),
    )

    def fail_geometry(*args, **kwargs):
        raise RuntimeError('synthetic geometry failure')

    monkeypatch.setattr(file_info_mod, 'read_geometry_arrays_from_segy', fail_geometry)
    info = file_info_mod.build_file_info(
        segy_path,
        ffid_byte=segyio.TraceField.FieldRecord,
        chno_byte=segyio.TraceField.TraceNumber,
        cmp_byte=segyio.TraceField.CDP,
        use_header_cache=False,
        include_geometry_arrays=True,
    )
    try:
        assert not np.any(info['geometry_valid_mask'])
        for key in GEOMETRY_ARRAY_KEYS:
            if key != 'geometry_valid_mask':
                assert np.all(np.isnan(info[key]))
    finally:
        close_info(info)


def test_include_geometry_false_keeps_existing_dict_shape(tmp_path: Path) -> None:
    segy_path = str(tmp_path / 'no_geometry_keys.sgy')
    write_geometry_segy(
        segy_path,
        source_x=np.asarray([0], dtype=np.int32),
        source_y=np.asarray([0], dtype=np.int32),
        receiver_x=np.asarray([10], dtype=np.int32),
        receiver_y=np.asarray([0], dtype=np.int32),
        scalar=np.asarray([1], dtype=np.int32),
        offsets=np.asarray([10], dtype=np.int32),
    )

    info = build_file_info(
        segy_path,
        ffid_byte=segyio.TraceField.FieldRecord,
        chno_byte=segyio.TraceField.TraceNumber,
        cmp_byte=segyio.TraceField.CDP,
        use_header_cache=False,
        include_geometry_arrays=False,
    )
    try:
        for key in GEOMETRY_ARRAY_KEYS:
            assert key not in info
    finally:
        close_info(info)


def test_geometry_request_rebuilds_cache_missing_geometry_keys(tmp_path: Path) -> None:
    segy_path = str(tmp_path / 'geometry_cache.sgy')
    cache_dir = tmp_path / 'cache'
    cache_dir.mkdir()
    write_geometry_segy(
        segy_path,
        source_x=np.asarray([0, 0], dtype=np.int32),
        source_y=np.asarray([0, 0], dtype=np.int32),
        receiver_x=np.asarray([30, 60], dtype=np.int32),
        receiver_y=np.asarray([40, 80], dtype=np.int32),
        scalar=np.asarray([1, 1], dtype=np.int32),
        offsets=np.asarray([7, 8], dtype=np.int32),
    )

    info = build_file_info(
        segy_path,
        ffid_byte=segyio.TraceField.FieldRecord,
        chno_byte=segyio.TraceField.TraceNumber,
        cmp_byte=segyio.TraceField.CDP,
        header_cache_dir=str(cache_dir),
        use_header_cache=True,
        include_geometry_arrays=False,
    )
    close_info(info)

    cache_path = cache_dir / (Path(segy_path).name + '.headers.big.npz')
    with np.load(cache_path, allow_pickle=False) as z:
        assert 'source_x_m' not in z.files

    info = build_file_info(
        segy_path,
        ffid_byte=segyio.TraceField.FieldRecord,
        chno_byte=segyio.TraceField.TraceNumber,
        cmp_byte=segyio.TraceField.CDP,
        header_cache_dir=str(cache_dir),
        use_header_cache=True,
        include_geometry_arrays=True,
    )
    try:
        np.testing.assert_allclose(
            info['offset_abs_geom_m'],
            np.asarray([50.0, 100.0], dtype=np.float32),
        )
    finally:
        close_info(info)

    with np.load(cache_path, allow_pickle=False) as z:
        for key in GEOMETRY_ARRAY_KEYS:
            assert key in z.files

    info = build_file_info(
        segy_path,
        ffid_byte=segyio.TraceField.FieldRecord,
        chno_byte=segyio.TraceField.TraceNumber,
        cmp_byte=segyio.TraceField.CDP,
        header_cache_dir=str(cache_dir),
        use_header_cache=True,
        include_geometry_arrays=True,
        coord_unit_scale_to_m=0.5,
    )
    try:
        np.testing.assert_allclose(
            info['offset_abs_geom_m'],
            np.asarray([25.0, 50.0], dtype=np.float32),
        )
    finally:
        close_info(info)
