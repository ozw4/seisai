from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import segyio

from seisai_dataset.local_window_dataset import LocalWindowDataset, LocalWindowDatasetConfig
from seisai_engine.pipelines.fbpick.common.artifacts import (
    ARTIFACT_SPECS,
    ARTIFACT_VERSION,
    ArtifactSpec,
    get_artifact_spec,
)
from seisai_engine.pipelines.fbpick.common.config import (
    FbpickCommonConfig,
    FbpickPathsCfg,
    FbpickThresholdsCfg,
)
from seisai_engine.pipelines.fbpick.common.io import (
    load_artifact,
    load_coarse_artifact_from_paths,
    save_artifact,
    save_coarse_artifact,
    save_fine_artifact,
)
from seisai_engine.pipelines.fbpick.global_qc.build_candidates import build_global_qc_candidates
from seisai_engine.pipelines.fbpick.global_qc.config import (
    GlobalQcArrivalBandCfg,
    GlobalQcArtifactCfg,
    GlobalQcBackendCfg,
    GlobalQcConfidenceCfg,
    GlobalQcConfig,
    GlobalQcConsistencyCfg,
    GlobalQcExportCfg,
    GlobalQcGeometryCfg,
    GlobalQcRejectPolicyCfg,
)


def _paths_cfg(tmp_path: Path, *, survey_id: str = 'survey') -> FbpickPathsCfg:
    return FbpickPathsCfg(
        out_dir=str(tmp_path / 'artifacts'),
        survey_id=survey_id,
    )


def _shape_from_spec(spec: ArtifactSpec, dims: dict[str, int], *, key: str) -> tuple[int, ...]:
    field = next(field for field in spec.fields if field.key == key)
    return tuple(int(dims[name]) for name in field.shape)


def _dims_from_spec(spec: ArtifactSpec) -> dict[str, int]:
    dims = {'n_traces': 3, 'n_samples': 8, 'local_window_len': 4}
    return {
        dim_name: dims[dim_name]
        for field in spec.fields
        for dim_name in field.shape
    }


def _make_stage_arrays(spec: ArtifactSpec) -> dict[str, np.ndarray]:
    dims = _dims_from_spec(spec)
    arrays: dict[str, np.ndarray] = {}
    for field in spec.fields:
        shape = _shape_from_spec(spec, dims, key=field.key)
        size = int(np.prod(shape, dtype=np.int64))
        dtype = np.dtype(field.dtype_name)
        if dtype == np.dtype(bool):
            value = (np.arange(size, dtype=np.int64) % 2) == 0
        elif np.issubdtype(dtype, np.floating):
            value = np.arange(1, size + 1, dtype=np.float32)
        else:
            value = np.arange(size, dtype=np.int64)
        arrays[field.key] = np.asarray(value, dtype=dtype).reshape(shape)
    return arrays


def _expected_dimensions(spec: ArtifactSpec, arrays: dict[str, np.ndarray]) -> dict[str, int]:
    dims: dict[str, int] = {}
    for field in spec.fields:
        arr = arrays[field.key]
        for axis, dim_name in enumerate(field.shape):
            dims[dim_name] = int(arr.shape[axis])
    return dims


def _valid_coarse_arrays(*, n_traces: int = 3, n_samples: int = 8) -> dict[str, np.ndarray]:
    prob = np.tile(
        np.linspace(1.0, float(n_samples), num=n_samples, dtype=np.float32),
        (n_traces, 1),
    )
    return {
        'prob': prob,
        'pick_idx': np.asarray([2, -1, 5], dtype=np.int32),
        'confidence': np.asarray([0.9, 0.0, 0.8], dtype=np.float32),
        'trace_valid': np.asarray([True, False, True], dtype=bool),
        'raw_trace_idx': np.arange(n_traces, dtype=np.int64),
        'offsets': np.asarray([10.0, 20.0, 30.0], dtype=np.float32),
        'time_axis': np.arange(n_samples, dtype=np.float32) * np.float32(0.002),
    }


def _write_unstructured_segy(path: Path, traces: np.ndarray, *, dt_us: int) -> None:
    arr = np.asarray(traces, dtype=np.float32)
    n_traces, n_samples = arr.shape

    spec = segyio.spec()
    spec.iline = 189
    spec.xline = 193
    spec.format = 5
    spec.sorting = 2
    spec.samples = np.arange(n_samples, dtype=np.int32)
    spec.tracecount = int(n_traces)

    with segyio.create(str(path), spec) as f:
        f.bin[segyio.BinField.Interval] = int(dt_us)
        for i in range(n_traces):
            f.header[i] = {
                segyio.TraceField.FieldRecord: 1,
                segyio.TraceField.TraceNumber: int(i + 1),
                segyio.TraceField.CDP: 1,
                segyio.TraceField.offset: int((i + 1) * 10),
                segyio.TraceField.SourceX: 100,
                segyio.TraceField.SourceY: 200,
                segyio.TraceField.GroupX: int(1000 + i * 10),
                segyio.TraceField.GroupY: 200,
                segyio.TraceField.SourceGroupScalar: 1,
            }
            f.trace[i] = arr[i]


def _global_qc_cfg(
    tmp_path: Path,
    *,
    survey_id: str,
    coarse_npz_path: Path,
    coarse_meta_path: Path,
    fine_npz_path: Path,
    fine_meta_path: Path,
) -> GlobalQcConfig:
    return GlobalQcConfig(
        fbpick=FbpickCommonConfig(
            paths=FbpickPathsCfg(out_dir=str(tmp_path / 'out'), survey_id=survey_id),
            thresholds=FbpickThresholdsCfg(
                confidence_min=0.0,
                trace_valid_min_fraction=0.0,
                qc_reject_confidence_below=0.0,
            ),
        ),
        coarse_artifact=GlobalQcArtifactCfg(
            artifact_npz_path=str(coarse_npz_path),
            artifact_meta_path=str(coarse_meta_path),
        ),
        fine_artifact=GlobalQcArtifactCfg(
            artifact_npz_path=str(fine_npz_path),
            artifact_meta_path=str(fine_meta_path),
        ),
        geometry=GlobalQcGeometryCfg(
            format='inline',
            path=None,
            inline={},
            normalization=None,
        ),
        backend=GlobalQcBackendCfg(name='trend', source_path=None),
        arrival_band=GlobalQcArrivalBandCfg(
            use_hard_mask=False,
            band_half_width_idx=1,
            uncertainty_scale=1.0,
            band_radius_sigma=1.0,
            min_half_width_idx=0,
            prior_floor=0.0,
            prior_power=1.0,
        ),
        consistency=GlobalQcConsistencyCfg(
            adjacent_radius=1,
            adjacent_sigma_idx=1.0,
            adjacent_min_count=1,
            trend_sigma_idx=1.0,
            outlier_radius=1,
            outlier_z_scale=1.0,
            outlier_min_count=1,
            outlier_mad_floor_idx=1.0,
            adjacent_weight=1.0,
            trend_weight=0.0,
            outlier_weight=0.0,
        ),
        confidence=GlobalQcConfidenceCfg(
            probability_weight=1.0,
            band_weight=0.0,
            trend_weight=0.0,
            consistency_weight=0.0,
            entropy_floor=0.0,
            entropy_power=1.0,
            trend_sigma_idx=1.0,
        ),
        reject_policy=GlobalQcRejectPolicyCfg(
            min_confidence=0.0,
            allow_zero_mass_failure=False,
            invalid_handling='reject_to_minus_one',
        ),
        export=GlobalQcExportCfg(write_csv=False, csv_path=None),
    )


@pytest.mark.parametrize('stage', sorted(ARTIFACT_SPECS))
def test_artifact_round_trip_matches_canonical_spec(stage: str, tmp_path: Path) -> None:
    spec = get_artifact_spec(stage)
    paths_cfg = _paths_cfg(tmp_path, survey_id=f'{stage}_survey')
    arrays = _make_stage_arrays(spec)

    save_artifact(
        paths_cfg=paths_cfg,
        stage=stage,
        arrays=arrays,
        source_refs={'upstream': 'synthetic'},
    )
    loaded = load_artifact(paths_cfg=paths_cfg, stage=stage)

    assert loaded.meta.artifact_version == ARTIFACT_VERSION
    assert loaded.meta.stage == spec.stage
    assert loaded.meta.dimensions == _expected_dimensions(spec, arrays)
    for field in spec.fields:
        np.testing.assert_array_equal(loaded.arrays[field.key], arrays[field.key])


def test_load_coarse_artifact_rejects_missing_key(tmp_path: Path) -> None:
    paths_cfg = _paths_cfg(tmp_path)
    saved = save_coarse_artifact(paths_cfg=paths_cfg, arrays=_valid_coarse_arrays())

    with np.load(saved.npz_path, allow_pickle=False) as z:
        payload = {key: z[key] for key in z.files if key != 'time_axis'}
    np.savez_compressed(saved.npz_path, **payload)

    with pytest.raises(ValueError, match='missing required keys'):
        load_coarse_artifact_from_paths(
            npz_path=saved.npz_path,
            meta_path=saved.meta_path,
            survey_id=paths_cfg.survey_id,
        )


def test_load_coarse_artifact_rejects_wrong_dtype(tmp_path: Path) -> None:
    paths_cfg = _paths_cfg(tmp_path)
    saved = save_coarse_artifact(paths_cfg=paths_cfg, arrays=_valid_coarse_arrays())

    with np.load(saved.npz_path, allow_pickle=False) as z:
        payload = {key: z[key] for key in z.files}
    payload['pick_idx'] = payload['pick_idx'].astype(np.int64)
    np.savez_compressed(saved.npz_path, **payload)

    with pytest.raises(TypeError, match='pick_idx'):
        load_coarse_artifact_from_paths(
            npz_path=saved.npz_path,
            meta_path=saved.meta_path,
            survey_id=paths_cfg.survey_id,
        )


def test_local_window_dataset_infer_accepts_canonical_coarse_artifact(tmp_path: Path) -> None:
    segy_path = tmp_path / 'synthetic.sgy'
    traces = np.stack(
        [np.arange(8, dtype=np.float32) + np.float32(i * 10.0) for i in range(3)],
        axis=0,
    )
    _write_unstructured_segy(segy_path, traces, dt_us=2000)

    paths_cfg = _paths_cfg(tmp_path, survey_id='local_window_survey')
    saved = save_coarse_artifact(paths_cfg=paths_cfg, arrays=_valid_coarse_arrays())

    ds = LocalWindowDataset(
        [str(segy_path)],
        cfg=LocalWindowDatasetConfig(local_window_len=4, mode='infer'),
        coarse_artifact_npz_path=saved.npz_path,
        coarse_artifact_meta_path=saved.meta_path,
        use_header_cache=False,
    )
    try:
        assert len(ds) == 2

        first = ds[0]
        second = ds[1]
    finally:
        ds.close()

    np.testing.assert_array_equal(first['raw_trace_idx'], np.asarray([0], dtype=np.int64))
    np.testing.assert_array_equal(second['raw_trace_idx'], np.asarray([2], dtype=np.int64))
    np.testing.assert_array_equal(
        first['local_window_start_idx'],
        np.asarray([0], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        second['local_window_start_idx'],
        np.asarray([3], dtype=np.int64),
    )
    np.testing.assert_array_equal(first['label_valid'], np.asarray([False], dtype=np.bool_))
    np.testing.assert_array_equal(second['label_valid'], np.asarray([False], dtype=np.bool_))


def test_build_global_qc_candidates_rejects_trace_alignment_mismatch(tmp_path: Path) -> None:
    survey_id = 'global_qc_survey'
    paths_cfg = _paths_cfg(tmp_path, survey_id=survey_id)
    coarse_saved = save_coarse_artifact(paths_cfg=paths_cfg, arrays=_valid_coarse_arrays())
    fine_saved = save_fine_artifact(
        paths_cfg=paths_cfg,
        arrays={
            'local_prob': np.asarray(
                [
                    [0.1, 0.7, 0.1, 0.1],
                    [0.1, 0.1, 0.7, 0.1],
                ],
                dtype=np.float32,
            ),
            'local_pick_idx': np.asarray([1, 2], dtype=np.int32),
            'raw_pick_idx': np.asarray([2, 6], dtype=np.int32),
            'local_window_start_idx': np.asarray([1, 4], dtype=np.int64),
            'local_window_end_idx': np.asarray([5, 8], dtype=np.int64),
            'raw_trace_idx': np.asarray([0, 1], dtype=np.int64),
            'confidence': np.asarray([0.9, 0.8], dtype=np.float32),
        },
    )
    cfg = _global_qc_cfg(
        tmp_path,
        survey_id=survey_id,
        coarse_npz_path=coarse_saved.npz_path,
        coarse_meta_path=coarse_saved.meta_path,
        fine_npz_path=fine_saved.npz_path,
        fine_meta_path=fine_saved.meta_path,
    )

    with pytest.raises(ValueError, match='coarse/fine artifacts are not 1:1 aligned'):
        build_global_qc_candidates(cfg)
