# Physical Center Refactor Guardrails

Date: 2026-05-17

These guardrails apply before splitting
`packages/seisai-engine/src/seisai_engine/pipelines/fbpick/physics/physical_center.py`.
The split must preserve behavior through the public facade:

```python
seisai_engine.pipelines.fbpick.physics.physical_center
```

## Public Import Contract

Keep these names importable from the facade module:

```python
from seisai_engine.pipelines.fbpick.physics.physical_center import (
    PHYSICAL_MODEL_FAILURE_FIT_FAILED,
    PHYSICAL_MODEL_FAILURE_GEOMETRY_INVALID,
    PHYSICAL_MODEL_FAILURE_INSUFFICIENT_OBSERVATIONS,
    PHYSICAL_MODEL_FAILURE_LABELS,
    PHYSICAL_MODEL_FAILURE_NONE,
    PHYSICAL_MODEL_FAILURE_PHYSICAL_DISABLED,
    PHYSICAL_MODEL_FAILURE_PREDICTION_INVALID,
    PHYSICAL_MODEL_STATUS_FALLBACK_EXISTING_TREND,
    PHYSICAL_MODEL_STATUS_FALLBACK_FEASIBLE_CLIP,
    PHYSICAL_MODEL_STATUS_FALLBACK_RELAXED_SEGMENT,
    PHYSICAL_MODEL_STATUS_FALLBACK_ROBUST,
    PHYSICAL_MODEL_STATUS_FIT_FAILED,
    PHYSICAL_MODEL_STATUS_GEOMETRY_INVALID,
    PHYSICAL_MODEL_STATUS_INSUFFICIENT_OBSERVATIONS,
    PHYSICAL_MODEL_STATUS_LABELS,
    PHYSICAL_MODEL_STATUS_PHYSICAL_DISABLED,
    PHYSICAL_MODEL_STATUS_TWO_PIECE_OK,
    PHYSICAL_OFFSET_SOURCE_GEOMETRY,
    PHYSICAL_OFFSET_SOURCE_HEADER,
    PHYSICAL_OFFSET_SOURCE_LABELS,
    PHYSICAL_OFFSET_SOURCE_NONE,
    PHYSICAL_RUNTIME_FIT_SOURCE_ADAPTIVE_REFIT,
    PHYSICAL_RUNTIME_FIT_SOURCE_ANCHOR_FIT,
    PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_EXISTING_TREND,
    PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_FULL_FIT_NO_COMPATIBLE_ANCHOR,
    PHYSICAL_RUNTIME_FIT_SOURCE_FALLBACK_ROBUST,
    PHYSICAL_RUNTIME_FIT_SOURCE_FULL_FIT,
    PHYSICAL_RUNTIME_FIT_SOURCE_LABELS,
    PHYSICAL_RUNTIME_FIT_SOURCE_NEAREST_ANCHOR_REUSE,
    PhysicalCenterFallbackPreflight,
    PhysicalCenterResult,
    build_geometry_two_piece_physical_center,
    preflight_geometry_two_piece_fallback,
)
```

The integer codes and label dictionaries for the `PHYSICAL_*` constants are
part of the test contract in
`test_physical_center_public_import_contract_is_stable`.

## Result Array Contract

Every `PhysicalCenterResult` field is a one-dimensional, save-friendly
`np.ndarray` with shape `(n_traces,)` and no object dtype.

| field | dtype |
|---|---|
| `physical_center_i` | `np.int32` |
| `physical_center_t_sec` | `np.float32` |
| `fine_center_i` | `np.int32` |
| `fine_center_t_sec` | `np.float32` |
| `physical_model_status` | `np.uint8` |
| `physical_model_failure_reason` | `np.uint8` |
| `physical_offset_source` | `np.uint8` |
| `physical_model_break_offset_m` | `np.float32` |
| `physical_model_slope_near_s_per_m` | `np.float32` |
| `physical_model_slope_far_s_per_m` | `np.float32` |
| `physical_model_velocity_near_m_s` | `np.float32` |
| `physical_model_velocity_far_m_s` | `np.float32` |
| `physical_model_neighbor_count` | `np.int32` |
| `physical_prefilter_valid_count` | `np.int32` |
| `physical_model_segment_id` | `np.int32` |
| `physical_model_side` | `np.int8` |
| `physical_model_resid_p50_ms` | `np.float32` |
| `physical_model_resid_p90_ms` | `np.float32` |
| `physical_anchor_group_id` | `np.int32` |
| `physical_anchor_is_anchor` | `np.bool_` |
| `physical_anchor_nearest_anchor_group_id` | `np.int32` |
| `physical_anchor_source_distance_m` | `np.float32` |
| `physical_runtime_t0_shift_ms` | `np.float32` |
| `physical_runtime_reuse_resid_p50_ms` | `np.float32` |
| `physical_runtime_reuse_resid_p90_ms` | `np.float32` |
| `physical_runtime_reuse_valid_count` | `np.int32` |
| `physical_runtime_refit_mask` | `np.bool_` |
| `physical_runtime_fit_source` | `np.uint8` |

Use `np.testing.assert_array_equal` for integer and boolean result arrays.
Use `np.testing.assert_allclose(..., equal_nan=True)` for floating result
arrays when comparing before and after a refactor.

## Characterization Coverage

`packages/seisai-engine/tests/test_fbpick_physical_center.py` is the primary
guardrail suite. Representative scenarios are pinned by these tests:

| scenario | test |
|---|---|
| disabled physical trend | `test_physical_disabled_returns_existing_trend_center` |
| geometry missing | `test_geometry_missing_falls_back_to_existing_trend_without_crashing` |
| header offset fit | `test_physical_center_uses_header_offsets_when_geometry_offset_disabled` |
| header offset fallback | `test_constant_header_offsets_fall_back_when_geometry_offset_disabled` |
| full two-piece fit | `test_synthetic_two_piece_trend_predicts_physical_centers` |
| observation sampling | `test_observation_sampling_limits_observations_before_ransac` |
| anchor reuse | `test_anchor_source_xy_reuses_nearest_anchor_without_non_anchor_fit` |
| t0 shift reuse | `test_anchor_source_xy_t0_shift_estimates_constant_target_shift` |
| adaptive refit | `test_anchor_source_xy_adaptive_refit_reduces_bad_reuse_tail` |
| runtime diagnostics and progress keys | `test_physical_center_runtime_and_progress_keys_are_stable` |
| result array schema | `test_physical_center_diagnostic_arrays_are_save_friendly` |

The split is acceptable only if the representative result arrays remain equal
or allclose under the same test inputs.

## Private Helper References

Tests currently reference private helpers through `physical_center_mod._...`.
During the split, move helpers to owner modules by concern. For any split PR
that has not migrated a test yet, keep a direct facade alias in
`physical_center.py`; remove that alias only in the same PR that moves the last
test/import to the owner module.

| helper | split policy |
|---|---|
| `_FitContextWorkItem` | move with fit work item orchestration |
| `_FitTaskResult` | move with fit task execution |
| `_GroupObservationContext` | move with observation context planning |
| `_ObservationPlan` | move with observation plan selection |
| `_ObservationPlanCache` | move with observation plan caching |
| `_SideObservationContext` | move with side/segment context planning |
| `_allocate_result_arrays` | move with result array construction |
| `_assign_fallback_all` | move with fallback assignment |
| `_assign_model_prediction` | move with prediction assignment |
| `_assign_model_prediction_batch` | move with prediction assignment |
| `_build_group_observation_contexts` | move with observation context planning |
| `_build_observation_plan` | move with observation plan selection |
| `_build_side_observation_context` | move with side/segment context planning |
| `_cache_entry_from_fit_task_result` | move with fit cache/task conversion |
| `_concat_group_traces` | move with source group utilities |
| `_fallback_center_for_trace` | move with fallback assignment |
| `_fit_cache_key` | move with fit cache/key utilities |
| `_fit_key_for_obs` | move with fit cache/key utilities |
| `_fit_model_for_plan` | move with model fitting |
| `_obs_with_target_gap_segment` | move with gap segment observation filtering |
| `_obs_with_target_signed_offset_side` | move with signed-side observation filtering |
| `_record_cached_context_hits` | move with runtime diagnostics accounting |
| `_record_new_fit_task_diagnostics` | move with runtime diagnostics accounting |
| `_sample_observation_indices_for_fit` | move with observation sampling |
| `_select_group_ids` | move with source group utilities |
| `_signed_offset_side_labels` | move with signed-side observation filtering |
| `_stable_unique` | move with small array utilities |

## Verification Commands

Run these before merging a physical-center split:

```bash
python -m pytest -q packages/seisai-engine/tests/test_fbpick_physical_center.py
python -m pytest -q packages/seisai-engine/tests/test_fbpick_physics.py packages/seisai-engine/tests/test_fbpick_physics_runtime_benchmark.py tests/test_compare_fbpick_physics_runtime.py
python -m pytest -q -m "not e2e and not integration"
```
