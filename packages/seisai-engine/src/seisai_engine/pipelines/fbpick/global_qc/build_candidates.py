from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from seisai_engine.pipelines.fbpick.common.artifacts import (
    COARSE_ARTIFACT_SPEC,
    ArtifactSpec,
    FINE_ARTIFACT_SPEC,
)
from seisai_engine.pipelines.fbpick.common.io import LoadedArtifact, load_artifact_from_paths

from .config import GlobalQcConfig

__all__ = [
    'GlobalQcCandidates',
    'build_global_qc_candidates',
]


def _validate_unique_non_negative_raw_trace_idx(
    raw_trace_idx: np.ndarray,
    *,
    label: str,
) -> None:
    if np.any(raw_trace_idx < 0):
        msg = f'{label} must contain only non-negative values'
        raise ValueError(msg)
    if np.unique(raw_trace_idx).shape[0] != int(raw_trace_idx.shape[0]):
        msg = f'{label} must be unique'
        raise ValueError(msg)


def _validate_uniform_time_axis(time_axis: np.ndarray) -> float:
    if int(time_axis.shape[0]) < 2:
        msg = 'coarse time_axis must have at least 2 samples to derive sample_interval_sec'
        raise ValueError(msg)
    diffs = np.diff(time_axis.astype(np.float64, copy=False))
    if np.any(diffs <= 0.0):
        msg = 'coarse time_axis must be strictly increasing'
        raise ValueError(msg)
    if not np.allclose(diffs, diffs[0], rtol=0.0, atol=1.0e-9):
        msg = 'coarse time_axis must be uniformly sampled'
        raise ValueError(msg)
    return float(diffs[0])


def _require_row_positive_mass(prob: np.ndarray, *, label: str) -> None:
    row_sum = np.sum(prob, axis=1, dtype=np.float64)
    if np.any(row_sum <= 0.0):
        bad_rows = np.nonzero(row_sum <= 0.0)[0].tolist()
        msg = f'{label} must have positive row mass for all rows, got bad rows {bad_rows}'
        raise ValueError(msg)


def _normalize_prob_rows(prob: np.ndarray, *, label: str) -> np.ndarray:
    _require_row_positive_mass(prob, label=label)
    row_sum = np.sum(prob, axis=1, keepdims=True, dtype=np.float64)
    return (prob / row_sum).astype(np.float32, copy=False)


def _artifact_arrays(
    artifact: LoadedArtifact,
    *,
    spec: ArtifactSpec,
) -> dict[str, np.ndarray]:
    return {field.key: artifact.arrays[field.key] for field in spec.fields}


@dataclass(frozen=True)
class GlobalQcCandidates:
    """Normalized inputs for engine-side global QC orchestration.

    Contracts
    ---------
    - `raw_trace_idx`: `(n_traces,)` int64, unique raw survey trace indices.
    - `trace_valid`: `(n_traces,)` bool, engine-visible valid mask before geometry/backend QC.
    - `offsets`: `(n_traces,)` float32, aligned with `raw_trace_idx`.
    - `time_axis`: `(n_samples,)` float32, raw sample axis shared by coarse/global QC.
    - `coarse_prob`: `(n_traces, n_samples)` float32, normalized per-trace coarse probability.
    - `coarse_pick_idx`: `(n_traces,)` int32, raw-axis pick; invalid traces must use `-1`.
    - `coarse_confidence`: `(n_traces,)` float32.
    - `fine_local_prob`: `(n_valid_traces, local_window_len)` float32 from the fine artifact.
    - `fine_local_pick_idx`: `(n_valid_traces,)` int32, local-window pick.
    - `fine_local_window_start_idx` / `fine_local_window_end_idx`: `(n_valid_traces,)` int64.
    - `fine_prob_raw`: `(n_traces, n_samples)` float32, fine probability scattered onto the raw axis.
    - `fine_raw_pick_idx`: `(n_traces,)` int32, raw-axis fine pick; invalid traces use `-1`.
    - `fine_confidence`: `(n_traces,)` float32.
    - `base_prob`: `(n_traces, n_samples)` float32, raw-axis probability passed to Phase 7 kernels.
    - `base_pick_idx`: `(n_traces,)` int32, fine-first pick used for inversion/repick; invalid is `-1`.
    - `base_confidence`: `(n_traces,)` float32.

    Notes
    -----
    - Index `0` is a valid sample index. Only `-1` denotes invalid.
    - `base_prob` is built from fine local probability only; coarse probability is retained as an
      explicit fallback candidate but is not used implicitly by the engine.
    """

    survey_id: str
    raw_trace_idx: np.ndarray
    trace_valid: np.ndarray
    offsets: np.ndarray
    time_axis: np.ndarray
    sample_interval_sec: float
    coarse_prob: np.ndarray
    coarse_pick_idx: np.ndarray
    coarse_confidence: np.ndarray
    fine_raw_trace_idx: np.ndarray
    fine_local_prob: np.ndarray
    fine_local_pick_idx: np.ndarray
    fine_local_window_start_idx: np.ndarray
    fine_local_window_end_idx: np.ndarray
    fine_prob_raw: np.ndarray
    fine_raw_pick_idx: np.ndarray
    fine_confidence: np.ndarray
    base_prob: np.ndarray
    base_pick_idx: np.ndarray
    base_confidence: np.ndarray

    @property
    def n_traces(self) -> int:
        return int(self.raw_trace_idx.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.time_axis.shape[0])


def build_global_qc_candidates(cfg: GlobalQcConfig) -> GlobalQcCandidates:
    if not isinstance(cfg, GlobalQcConfig):
        msg = 'cfg must be GlobalQcConfig'
        raise TypeError(msg)

    coarse = load_artifact_from_paths(
        stage='coarse',
        npz_path=cfg.coarse_artifact.artifact_npz_path,
        meta_path=cfg.coarse_artifact.artifact_meta_path,
        survey_id=cfg.fbpick.paths.survey_id,
    )
    fine = load_artifact_from_paths(
        stage='fine',
        npz_path=cfg.fine_artifact.artifact_npz_path,
        meta_path=cfg.fine_artifact.artifact_meta_path,
        survey_id=cfg.fbpick.paths.survey_id,
    )

    coarse_arrays = _artifact_arrays(coarse, spec=COARSE_ARTIFACT_SPEC)
    fine_arrays = _artifact_arrays(fine, spec=FINE_ARTIFACT_SPEC)

    coarse_prob = coarse_arrays['prob']
    coarse_pick_idx = coarse_arrays['pick_idx']
    coarse_confidence = coarse_arrays['confidence']
    trace_valid = coarse_arrays['trace_valid']
    raw_trace_idx = coarse_arrays['raw_trace_idx']
    offsets = coarse_arrays['offsets']
    time_axis = coarse_arrays['time_axis']

    fine_local_prob = fine_arrays['local_prob']
    fine_local_pick_idx = fine_arrays['local_pick_idx']
    fine_raw_pick_idx_valid = fine_arrays['raw_pick_idx']
    fine_window_start = fine_arrays['local_window_start_idx']
    fine_window_end = fine_arrays['local_window_end_idx']
    fine_raw_trace_idx = fine_arrays['raw_trace_idx']
    fine_confidence_valid = fine_arrays['confidence']

    _validate_unique_non_negative_raw_trace_idx(raw_trace_idx, label='coarse.raw_trace_idx')
    _validate_unique_non_negative_raw_trace_idx(fine_raw_trace_idx, label='fine.raw_trace_idx')

    sample_interval_sec = _validate_uniform_time_axis(time_axis)
    coarse_prob = _normalize_prob_rows(coarse_prob, label='coarse.prob')
    _require_row_positive_mass(fine_local_prob, label='fine.local_prob')

    if np.any(trace_valid & (coarse_pick_idx < 0)):
        bad_rows = np.nonzero(trace_valid & (coarse_pick_idx < 0))[0].tolist()
        msg = f'coarse valid traces must have non-negative pick_idx, got bad rows {bad_rows}'
        raise ValueError(msg)
    if np.any((~trace_valid) & (coarse_pick_idx != -1)):
        bad_rows = np.nonzero((~trace_valid) & (coarse_pick_idx != -1))[0].tolist()
        msg = f'coarse invalid traces must use pick_idx=-1, got bad rows {bad_rows}'
        raise ValueError(msg)
    if np.any(trace_valid & (coarse_pick_idx >= int(time_axis.shape[0]))):
        bad_rows = np.nonzero(trace_valid & (coarse_pick_idx >= int(time_axis.shape[0])))[0].tolist()
        msg = f'coarse pick_idx must be < n_samples for valid traces, got bad rows {bad_rows}'
        raise ValueError(msg)

    coarse_valid_raw_trace_idx = raw_trace_idx[trace_valid]
    if fine_raw_trace_idx.shape != coarse_valid_raw_trace_idx.shape:
        msg = (
            'fine artifact trace count must match coarse valid trace count exactly: '
            f'fine={int(fine_raw_trace_idx.shape[0])}, '
            f'coarse_valid={int(coarse_valid_raw_trace_idx.shape[0])}'
        )
        raise ValueError(msg)
    if not np.array_equal(fine_raw_trace_idx, coarse_valid_raw_trace_idx):
        msg = (
            'fine.raw_trace_idx must match coarse.raw_trace_idx[coarse.trace_valid] exactly; '
            'coarse/fine artifacts are not 1:1 aligned'
        )
        raise ValueError(msg)

    local_window_len = int(fine_local_prob.shape[1])
    fine_prob_raw = np.zeros_like(coarse_prob, dtype=np.float32)
    fine_raw_pick_idx = np.full(raw_trace_idx.shape, -1, dtype=np.int32)
    fine_confidence = np.zeros(raw_trace_idx.shape, dtype=np.float32)
    valid_positions = np.nonzero(trace_valid)[0]

    for local_row_idx, raw_pos in enumerate(valid_positions.tolist()):
        start_idx = int(fine_window_start[local_row_idx])
        end_idx = int(fine_window_end[local_row_idx])
        local_pick_idx = int(fine_local_pick_idx[local_row_idx])
        raw_pick_idx = int(fine_raw_pick_idx_valid[local_row_idx])
        if start_idx < 0 or end_idx <= start_idx:
            msg = (
                'fine local window must satisfy 0 <= start < end, got '
                f'start={start_idx}, end={end_idx}, row={local_row_idx}'
            )
            raise ValueError(msg)
        if end_idx > int(time_axis.shape[0]):
            msg = (
                'fine local window end must stay within coarse raw axis, got '
                f'end={end_idx}, n_samples={int(time_axis.shape[0])}, row={local_row_idx}'
            )
            raise ValueError(msg)

        raw_window_len = end_idx - start_idx
        if raw_window_len > local_window_len:
            msg = (
                'fine local window raw coverage exceeds local_prob width, got '
                f'raw_window_len={raw_window_len}, local_window_len={local_window_len}, '
                f'row={local_row_idx}'
            )
            raise ValueError(msg)
        if local_pick_idx < 0 or local_pick_idx >= local_window_len:
            msg = (
                'fine local_pick_idx must satisfy 0 <= idx < local_window_len, got '
                f'idx={local_pick_idx}, local_window_len={local_window_len}, row={local_row_idx}'
            )
            raise ValueError(msg)
        if local_pick_idx >= raw_window_len:
            msg = (
                'fine local_pick_idx must stay inside raw-covered local samples, got '
                f'idx={local_pick_idx}, raw_window_len={raw_window_len}, row={local_row_idx}'
            )
            raise ValueError(msg)
        if raw_pick_idx < 0:
            msg = f'fine raw_pick_idx must be >= 0 for valid traces, got row={local_row_idx}'
            raise ValueError(msg)
        if raw_pick_idx != start_idx + local_pick_idx:
            msg = (
                'fine raw_pick_idx must map from local_window_start_idx + local_pick_idx, got '
                f'raw_pick_idx={raw_pick_idx}, start={start_idx}, local_pick_idx={local_pick_idx}, '
                f'row={local_row_idx}'
            )
            raise ValueError(msg)
        if raw_pick_idx >= int(time_axis.shape[0]):
            msg = (
                'fine raw_pick_idx must be < coarse n_samples, got '
                f'raw_pick_idx={raw_pick_idx}, n_samples={int(time_axis.shape[0])}, '
                f'row={local_row_idx}'
            )
            raise ValueError(msg)
        row_prob = fine_local_prob[local_row_idx]
        row_argmax = int(np.argmax(row_prob))
        if row_argmax != local_pick_idx:
            msg = (
                'fine local_pick_idx must match argmax(local_prob), got '
                f'argmax={row_argmax}, local_pick_idx={local_pick_idx}, row={local_row_idx}'
            )
            raise ValueError(msg)

        raw_slice_prob = row_prob[:raw_window_len].astype(np.float32, copy=False)
        raw_slice_mass = float(np.sum(raw_slice_prob, dtype=np.float64))
        if raw_slice_mass <= 0.0:
            msg = (
                'fine local_prob has zero feasible mass on raw-covered samples, got '
                f'row={local_row_idx}, raw_window_len={raw_window_len}'
            )
            raise ValueError(msg)
        raw_slice_prob = (raw_slice_prob / raw_slice_mass).astype(np.float32, copy=False)
        fine_prob_raw[raw_pos, start_idx:end_idx] = raw_slice_prob
        fine_raw_pick_idx[raw_pos] = np.int32(raw_pick_idx)
        fine_confidence[raw_pos] = np.float32(fine_confidence_valid[local_row_idx])

    base_prob = fine_prob_raw.astype(np.float32, copy=False)
    base_pick_idx = fine_raw_pick_idx.astype(np.int32, copy=False)
    base_confidence = fine_confidence.astype(np.float32, copy=False)

    if np.any(trace_valid & (base_pick_idx < 0)):
        bad_rows = np.nonzero(trace_valid & (base_pick_idx < 0))[0].tolist()
        msg = f'fine-first base_pick_idx must be valid on coarse-valid traces, got bad rows {bad_rows}'
        raise ValueError(msg)
    if np.any(trace_valid & (base_pick_idx >= int(time_axis.shape[0]))):
        bad_rows = np.nonzero(trace_valid & (base_pick_idx >= int(time_axis.shape[0])))[0].tolist()
        msg = f'fine-first base_pick_idx must stay on raw axis, got bad rows {bad_rows}'
        raise ValueError(msg)

    if np.any(np.sum(base_prob[trace_valid], axis=1, dtype=np.float64) <= 0.0):
        bad_rows = np.nonzero(np.sum(base_prob[trace_valid], axis=1, dtype=np.float64) <= 0.0)[0]
        msg = f'fine-first base_prob must have positive row mass on valid traces, got bad rows {bad_rows.tolist()}'
        raise ValueError(msg)

    return GlobalQcCandidates(
        survey_id=coarse.meta.survey_id,
        raw_trace_idx=raw_trace_idx,
        trace_valid=trace_valid,
        offsets=offsets,
        time_axis=time_axis,
        sample_interval_sec=float(sample_interval_sec),
        coarse_prob=coarse_prob,
        coarse_pick_idx=coarse_pick_idx,
        coarse_confidence=coarse_confidence,
        fine_raw_trace_idx=fine_raw_trace_idx,
        fine_local_prob=fine_local_prob,
        fine_local_pick_idx=fine_local_pick_idx,
        fine_local_window_start_idx=fine_window_start,
        fine_local_window_end_idx=fine_window_end,
        fine_prob_raw=fine_prob_raw,
        fine_raw_pick_idx=fine_raw_pick_idx,
        fine_confidence=fine_confidence,
        base_prob=base_prob,
        base_pick_idx=base_pick_idx,
        base_confidence=base_confidence,
    )
