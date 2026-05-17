# Physical Center Refactor Guardrails

Date: 2026-05-17

These guardrails apply to the split
`packages/seisai-engine/src/seisai_engine/pipelines/fbpick/physics/physical_center*.py`
modules. The split must preserve behavior through the public facade:

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

## Module Responsibility Map

Keep public imports routed through
`seisai_engine.pipelines.fbpick.physics.physical_center`. Add implementation
code to the owner module below instead of growing the facade.

| module | responsibility |
|---|---|
| `physical_center.py` | public facade, preflight wrapper, and fit-policy dispatch only |
| `physical_center_types.py` | public result dataclasses, status/failure/source constants, and label maps |
| `physical_center_context.py` | shared input, geometry/build, and workspace dataclasses |
| `physical_center_setup.py` | table validation, fallback preflight, geometry/source-group setup, prefiltering, workspace allocation |
| `physical_center_geometry.py` | saved geometry loading, offset source selection, source-group construction, signed-side label context |
| `physical_center_observation.py` | group observation contexts, side/gap filtering, observation plan cache and plan selection |
| `physical_center_fit.py` | fit strategy selection, observation sampling, fit cache keys, fit task execution, fit diagnostics |
| `physical_center_context_fit.py` | fit-context work item preparation, cache accounting, and assignment orchestration |
| `physical_center_prediction.py` | single-trace and batched model prediction assignment |
| `physical_center_fallback.py` | result array allocation, scalar/vector fallback assignment, disabled/invalid fallback finalization |
| `physical_center_full_fit.py` | `fit_policy='full'` orchestration |
| `physical_center_anchor.py` | anchor selection diagnostics and anchor model context construction |
| `physical_center_anchor_policy.py` | `fit_policy='anchor_source_xy'` top-level orchestration |
| `physical_center_anchor_reuse.py` | nearest-anchor reuse, t0-shift reuse, adaptive refit, and no-compatible-anchor fallback |

Contributor rule of thumb: add a helper next to the data it owns. If the
change affects observations, start in `physical_center_observation.py`; if it
affects fitting or sampling, start in `physical_center_fit.py`; if it only
writes result arrays, start in `physical_center_prediction.py` or
`physical_center_fallback.py`. Add a new public export only when external code
must import it from `physical_center.py`.

## Maintainability Metrics

Measured on 2026-05-17 after the cleanup split:

| file | lines |
|---|---:|
| `physical_center.py` | 165 |
| `physical_center_anchor.py` | 346 |
| `physical_center_anchor_policy.py` | 159 |
| `physical_center_anchor_reuse.py` | 902 |
| `physical_center_context.py` | 116 |
| `physical_center_context_fit.py` | 741 |
| `physical_center_fallback.py` | 807 |
| `physical_center_fit.py` | 771 |
| `physical_center_full_fit.py` | 117 |
| `physical_center_geometry.py` | 429 |
| `physical_center_observation.py` | 754 |
| `physical_center_prediction.py` | 388 |
| `physical_center_setup.py` | 520 |
| `physical_center_types.py` | 154 |

`build_geometry_two_piece_physical_center` is 55 lines. All split modules are
below 1,000 lines; `physical_center_anchor_reuse.py` is the largest module and
is the first candidate for another split if anchor reuse grows.

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

The guardrail suite is split by responsibility:

| file | coverage |
|---|---|
| `test_fbpick_physical_center.py` | public facade contract, result schema, runtime/progress diagnostic keys |
| `test_fbpick_physical_center_observation.py` | group/side/gap observation planning and cached-plan equivalence |
| `test_fbpick_physical_center_fit.py` | disabled/full-fit/fallback/prediction/sampling/header-offset behavior |
| `test_fbpick_physical_center_anchor.py` | anchor-source-xy reuse, t0 shift, adaptive refit, and full-fit fallback |

Representative scenarios are pinned by these tests:

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

`physical_center.py` must not carry private compatibility aliases. Tests that
need private helper characterization import the owner module directly, for
example `physical_center_observation as observation_mod` or
`physical_center_fit as fit_mod`. If a future split temporarily needs a private
facade alias, document the alias in this section and remove it in the same
change that moves the last test/import to the owner module.

The cleanup removed the previous `_PHYSICAL_CENTER_PRIVATE_COMPAT` tuple and
the facade `_build_observation_plan` alias. The current owner mapping is:

| helper | owner module |
|---|---|
| `_FitContextWorkItem` | `physical_center_context_fit.py` |
| `_FitTaskResult` | `physical_center_fit.py` |
| `_GroupObservationContext` | `physical_center_observation.py` |
| `_ObservationPlan` | `physical_center_observation.py` |
| `_ObservationPlanCache` | `physical_center_observation.py` |
| `_SideObservationContext` | `physical_center_observation.py` |
| `_allocate_result_arrays` | `physical_center_fallback.py` |
| `_assign_fallback_all` | `physical_center_fallback.py` |
| `_assign_model_prediction` | `physical_center_prediction.py` |
| `_assign_model_prediction_batch` | `physical_center_prediction.py` |
| `_build_group_observation_contexts` | `physical_center_observation.py` |
| `_build_observation_plan` | `physical_center_observation.py`; use `physical_center_context_fit.py` only for the default-min-fit wrapper |
| `_build_side_observation_context` | `physical_center_observation.py` |
| `_cache_entry_from_fit_task_result` | `physical_center_context_fit.py` |
| `_concat_group_traces` | `physical_center_observation.py` |
| `_fallback_center_for_trace` | `physical_center_fallback.py` |
| `_fit_cache_key` | `physical_center_fit.py` |
| `_fit_key_for_obs` | `physical_center_fit.py` |
| `_fit_model_for_plan` | `physical_center_fit.py` |
| `_obs_with_target_gap_segment` | `physical_center_observation.py` |
| `_obs_with_target_signed_offset_side` | `physical_center_observation.py` |
| `_record_cached_context_hits` | `physical_center_context_fit.py` |
| `_record_new_fit_task_diagnostics` | `physical_center_context_fit.py` |
| `_sample_observation_indices_for_fit` | `physical_center_fit.py` |
| `_select_group_ids` | `physical_center_observation.py` |
| `_signed_offset_side_labels` | `physical_center_geometry.py` |
| `_stable_unique` | `physical_center_observation.py` |

## Verification Commands

Run these before merging a physical-center split:

```bash
python -m pytest -q packages/seisai-engine/tests/test_fbpick_physical_center.py packages/seisai-engine/tests/test_fbpick_physical_center_observation.py packages/seisai-engine/tests/test_fbpick_physical_center_fit.py packages/seisai-engine/tests/test_fbpick_physical_center_anchor.py
python -m pytest -q packages/seisai-engine/tests/test_fbpick_physics.py packages/seisai-engine/tests/test_fbpick_physics_runtime_benchmark.py tests/test_compare_fbpick_physics_runtime.py
python -m pytest -q -m "not e2e and not integration"
```
