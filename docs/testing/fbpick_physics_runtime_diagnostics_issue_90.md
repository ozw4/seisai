# FBPick Physics Runtime Diagnostics Check

This records the Issue #90 Arakawa one-file runtime check for the physical
center diagnostics change.

Date: 2026-05-12

Runner:

```bash
python -m cli.run_arakawa_fbpick_physical_export --config <runtime-check-config>
```

The check used the same Arakawa SEG-Y file as
`proc/arakawa/configs/run_coarse_physics_export_minimal.yaml`:

```text
/home/dcuser/data/ActiveSeisField/Arakawa2026/fdata_hset_ARA26_Vib.sgy
```

Temporary runtime-check configs set separate `paths.work_dir` values so A0 and
A1 artifacts could be compared directly:

```text
.work/codex/arakawa_runtime_A0.yaml
.work/codex/arakawa_runtime_A1.yaml
```

Generated outputs:

```text
proc/arakawa/runtime_runs/A0_full/
proc/arakawa/runtime_runs/A1_diagnostics_only/
```

## Results

```text
A0 physics_total_sec: 1626.761
A1 physics_total_sec: 1621.2469229309354
A1 runner wall sec: 1624.836
A1/A0 overhead ratio: 0.9988166669842712
n_fit_calls: 1757
cache_hit_rate: 0.9943209182205759
```

`physical_runtime.diagnostics_enabled` was disabled for A0, so the A0
`physics_total_sec` value is the external physics-only runner wall measurement.
The A1 internal summary was written to:

```text
proc/arakawa/runtime_runs/A1_diagnostics_only/robust/Arakawa2026__fdata_hset_ARA26_Vib.physics_runtime_summary.json
```

## Exact-Match Checks

Compared:

```text
proc/arakawa/runtime_runs/A0_full/robust/Arakawa2026__fdata_hset_ARA26_Vib.robust.npz
proc/arakawa/runtime_runs/A1_diagnostics_only/robust/Arakawa2026__fdata_hset_ARA26_Vib.robust.npz
```

Results:

```text
physical_center_i_exact_match=True
fine_center_i_exact_match=True
physical_model_status_exact_match=True
physical_model_failure_reason_exact_match=True
```

## A1 Runtime Summary

```text
physics_total_sec=1621.2469229309354
physical_center_total_sec=1552.9374448850285
ransac_fit_total_sec=212.45718785421923
n_fit_calls=1757
n_cache_hits=307624
n_cache_misses=1757
cache_hit_rate=0.9943209182205759
n_source_groups=281
n_unique_fit_contexts=1757
ransac_fit_time_p50_sec=0.12846213998273015
ransac_fit_time_p90_sec=0.21333527509123087
ransac_fit_time_p99_sec=0.3894252842292191
obs_count_for_fit_p50=845.0
obs_count_for_fit_p90=5189.8
obs_count_for_fit_p99=5470.88
```

## Verification

```text
pytest -q \
  packages/seisai-engine/tests/test_fbpick_physics.py::test_physical_runtime_diagnostics_initializes_with_zero_counts \
  packages/seisai-engine/tests/test_fbpick_physics.py::test_physical_runtime_diagnostics_fit_timer_increments_counts \
  packages/seisai-engine/tests/test_fbpick_physics.py::test_run_physics_lite_end_to_end_outputs_full_covering_robust_picks \
  packages/seisai-engine/tests/test_fbpick_physics.py::test_run_physics_lite_allows_disabling_runtime_diagnostics \
  packages/seisai-engine/tests/test_fbpick_physics.py::test_runtime_diagnostics_do_not_change_physical_center_outputs \
  packages/seisai-engine/tests/test_fbpick_physical_center.py::test_physical_center_calls_existing_two_piece_ransac
```

Result:

```text
6 passed in 2.63s
```
