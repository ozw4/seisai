# FBPick physics runtime benchmark

- manifest: `/workspace/proc/arakawa/experiments/runtime_speedup/benchmark_manifest.yaml`
- tag: `Arakawa2026__fdata_hset_ARA26_Vib`
- artifacts_only: `False`

## Baseline

| name | config | robust_npz | export_npz | runtime_summary |
|---|---|---|---|---|
| A0_full | `/workspace/proc/arakawa/experiments/runtime_speedup/configs/A0_full.yaml` | `/workspace/proc/arakawa/outputs/runtime_runs/A0_full/robust/Arakawa2026__fdata_hset_ARA26_Vib.robust.npz` | `/workspace/proc/arakawa/outputs/runtime_runs/A0_full/grstat/Arakawa2026__fdata_hset_ARA26_Vib.physical_center.snap_peak.ltcor2.npz` | `/workspace/proc/arakawa/outputs/runtime_runs/A0_full/robust/Arakawa2026__fdata_hset_ARA26_Vib.physics_runtime_summary.json` |

## Candidate Artifacts

| candidate | config | robust_npz | export_npz | runtime_summary | comparison_json | comparison_csv |
|---|---|---|---|---|---|---|
| A0D_downsample_only | `/workspace/proc/arakawa/experiments/runtime_speedup/configs/A0D_downsample_only.yaml` | `/workspace/proc/arakawa/outputs/runtime_runs/A0D_downsample_only/robust/Arakawa2026__fdata_hset_ARA26_Vib.robust.npz` | `/workspace/proc/arakawa/outputs/runtime_runs/A0D_downsample_only/grstat/Arakawa2026__fdata_hset_ARA26_Vib.physical_center.snap_peak.ltcor2.npz` | `/workspace/proc/arakawa/outputs/runtime_runs/A0D_downsample_only/robust/Arakawa2026__fdata_hset_ARA26_Vib.physics_runtime_summary.json` | `/workspace/proc/arakawa/outputs/runtime_benchmark/B101/comparisons/A0D_downsample_only_vs_A0_full.json` | `/workspace/proc/arakawa/outputs/runtime_benchmark/B101/comparisons/A0D_downsample_only_vs_A0_full.csv` |
| A1_diagnostics_only | `/workspace/proc/arakawa/experiments/runtime_speedup/configs/A1_diagnostics_only.yaml` | `/workspace/proc/arakawa/outputs/runtime_runs/A1_diagnostics_only/robust/Arakawa2026__fdata_hset_ARA26_Vib.robust.npz` | `/workspace/proc/arakawa/outputs/runtime_runs/A1_diagnostics_only/grstat/Arakawa2026__fdata_hset_ARA26_Vib.physical_center.snap_peak.ltcor2.npz` | `/workspace/proc/arakawa/outputs/runtime_runs/A1_diagnostics_only/robust/Arakawa2026__fdata_hset_ARA26_Vib.physics_runtime_summary.json` | `/workspace/proc/arakawa/outputs/runtime_benchmark/B101/comparisons/A1_diagnostics_only_vs_A0_full.json` | `/workspace/proc/arakawa/outputs/runtime_benchmark/B101/comparisons/A1_diagnostics_only_vs_A0_full.csv` |
| A2_anchor_selection_dry_run | `/workspace/proc/arakawa/experiments/runtime_speedup/configs/A2_anchor_selection_dry_run.yaml` | `/workspace/proc/arakawa/outputs/runtime_runs/A2_anchor_selection_dry_run/robust/Arakawa2026__fdata_hset_ARA26_Vib.robust.npz` | `/workspace/proc/arakawa/outputs/runtime_runs/A2_anchor_selection_dry_run/grstat/Arakawa2026__fdata_hset_ARA26_Vib.physical_center.snap_peak.ltcor2.npz` | `/workspace/proc/arakawa/outputs/runtime_runs/A2_anchor_selection_dry_run/robust/Arakawa2026__fdata_hset_ARA26_Vib.physics_runtime_summary.json` | `/workspace/proc/arakawa/outputs/runtime_benchmark/B101/comparisons/A2_anchor_selection_dry_run_vs_A0_full.json` | `/workspace/proc/arakawa/outputs/runtime_benchmark/B101/comparisons/A2_anchor_selection_dry_run_vs_A0_full.csv` |
| A3_anchor_stride5_nearest_anchor | `/workspace/proc/arakawa/experiments/runtime_speedup/configs/A3_anchor_stride5_nearest_anchor.yaml` | `/workspace/proc/arakawa/outputs/runtime_runs/A3_anchor_stride5_nearest_anchor/robust/Arakawa2026__fdata_hset_ARA26_Vib.robust.npz` | `/workspace/proc/arakawa/outputs/runtime_runs/A3_anchor_stride5_nearest_anchor/grstat/Arakawa2026__fdata_hset_ARA26_Vib.physical_center.snap_peak.ltcor2.npz` | `/workspace/proc/arakawa/outputs/runtime_runs/A3_anchor_stride5_nearest_anchor/robust/Arakawa2026__fdata_hset_ARA26_Vib.physics_runtime_summary.json` | `/workspace/proc/arakawa/outputs/runtime_benchmark/B101/comparisons/A3_anchor_stride5_nearest_anchor_vs_A0_full.json` | `/workspace/proc/arakawa/outputs/runtime_benchmark/B101/comparisons/A3_anchor_stride5_nearest_anchor_vs_A0_full.csv` |
| A4_anchor_stride5_t0_shift | `/workspace/proc/arakawa/experiments/runtime_speedup/configs/A4_anchor_stride5_t0_shift.yaml` | `/workspace/proc/arakawa/outputs/runtime_runs/A4_anchor_stride5_t0_shift/robust/Arakawa2026__fdata_hset_ARA26_Vib.robust.npz` | `/workspace/proc/arakawa/outputs/runtime_runs/A4_anchor_stride5_t0_shift/grstat/Arakawa2026__fdata_hset_ARA26_Vib.physical_center.snap_peak.ltcor2.npz` | `/workspace/proc/arakawa/outputs/runtime_runs/A4_anchor_stride5_t0_shift/robust/Arakawa2026__fdata_hset_ARA26_Vib.physics_runtime_summary.json` | `/workspace/proc/arakawa/outputs/runtime_benchmark/B101/comparisons/A4_anchor_stride5_t0_shift_vs_A0_full.json` | `/workspace/proc/arakawa/outputs/runtime_benchmark/B101/comparisons/A4_anchor_stride5_t0_shift_vs_A0_full.csv` |
| A5_anchor_stride5_t0_shift_adaptive_refit | `/workspace/proc/arakawa/experiments/runtime_speedup/configs/A5_anchor_stride5_t0_shift_adaptive_refit.yaml` | `/workspace/proc/arakawa/outputs/runtime_runs/A5_anchor_stride5_t0_shift_adaptive_refit/robust/Arakawa2026__fdata_hset_ARA26_Vib.robust.npz` | `/workspace/proc/arakawa/outputs/runtime_runs/A5_anchor_stride5_t0_shift_adaptive_refit/grstat/Arakawa2026__fdata_hset_ARA26_Vib.physical_center.snap_peak.ltcor2.npz` | `/workspace/proc/arakawa/outputs/runtime_runs/A5_anchor_stride5_t0_shift_adaptive_refit/robust/Arakawa2026__fdata_hset_ARA26_Vib.physics_runtime_summary.json` | `/workspace/proc/arakawa/outputs/runtime_benchmark/B101/comparisons/A5_anchor_stride5_t0_shift_adaptive_refit_vs_A0_full.json` | `/workspace/proc/arakawa/outputs/runtime_benchmark/B101/comparisons/A5_anchor_stride5_t0_shift_adaptive_refit_vs_A0_full.csv` |
| A6_A5_obs_downsample256 | `/workspace/proc/arakawa/experiments/runtime_speedup/configs/A6_A5_obs_downsample256.yaml` | `/workspace/proc/arakawa/outputs/runtime_runs/A6_A5_obs_downsample256/robust/Arakawa2026__fdata_hset_ARA26_Vib.robust.npz` | `/workspace/proc/arakawa/outputs/runtime_runs/A6_A5_obs_downsample256/grstat/Arakawa2026__fdata_hset_ARA26_Vib.physical_center.snap_peak.ltcor2.npz` | `/workspace/proc/arakawa/outputs/runtime_runs/A6_A5_obs_downsample256/robust/Arakawa2026__fdata_hset_ARA26_Vib.physics_runtime_summary.json` | `/workspace/proc/arakawa/outputs/runtime_benchmark/B101/comparisons/A6_A5_obs_downsample256_vs_A0_full.json` | `/workspace/proc/arakawa/outputs/runtime_benchmark/B101/comparisons/A6_A5_obs_downsample256_vs_A0_full.csv` |

## Runtime Summary

| candidate | gates | physics_total_sec | speedup_physics_total | ransac_fit_total_sec | missing_runtime_keys |
|---|---|---:|---:|---:|---|
| A0D_downsample_only | pass | 1463.46 | 1.05146 | 58.6179 |  |
| A1_diagnostics_only | pass | 1538.51 | 1.00017 | 231.178 |  |
| A2_anchor_selection_dry_run | pass | 1515.46 | 1.01538 | 213.73 |  |
| A3_anchor_stride5_nearest_anchor | fail | 1382.56 | 1.11299 | 98.5543 |  |
| A4_anchor_stride5_t0_shift | pass | 1385.85 | 1.11034 | 93.1566 |  |
| A5_anchor_stride5_t0_shift_adaptive_refit | pass | 1463.36 | 1.05153 | 149.965 |  |
| A6_A5_obs_downsample256 | pass | 1455.38 | 1.0573 | 43.4683 |  |

## Detailed Timing

| candidate | physics_total_sec | physical_center_total_sec | non_ransac_total_sec | ransac_fit_total_sec | neighbor_plan_sec | side_segment_build_sec | prediction_sec | assignment_sec |
|---|---|---|---|---|---|---|---|---|
| A0D_downsample_only | 1463.46 | 1393.49 | 1334.88 | 58.6179 | 0.456547 | 1125.53 | 0.238075 | 0.206034 |
| A1_diagnostics_only | 1538.51 | 1469.21 | 1238.03 | 231.178 | 0.433489 | 1149.66 | 0.315365 | 0.243518 |
| A2_anchor_selection_dry_run | 1515.46 | 1445.13 | 1231.4 | 213.73 | 0.434906 | 1152.51 | 0.33366 | 0.274278 |
| A3_anchor_stride5_nearest_anchor | 1382.56 | 1312.29 | 1213.74 | 98.5543 | 0.512721 | 1147.51 | 0.303204 | 0.204481 |
| A4_anchor_stride5_t0_shift | 1385.85 | 1315.63 | 1222.47 | 93.1566 | 0.507157 | 1146.39 | 0.263819 | 0.193469 |
| A5_anchor_stride5_t0_shift_adaptive_refit | 1463.36 | 1393.24 | 1243.28 | 149.965 | 0.560401 | 1158.18 | 0.344916 | 0.253999 |
| A6_A5_obs_downsample256 | 1455.38 | 1386.43 | 1342.96 | 43.4683 | 0.574055 | 1149.55 | 0.27583 | 0.205762 |

## Exact Match Summary

| candidate | key | available | shape_match | arrays_match | missing_baseline | missing_candidate |
|---|---|---|---|---|---|---|
| A0D_downsample_only | physical_model_status | pass | pass | fail | no | no |
| A0D_downsample_only | physical_model_failure_reason | pass | pass | fail | no | no |
| A0D_downsample_only | physical_center_i | pass | pass | fail | no | no |
| A0D_downsample_only | fine_center_i | pass | pass | fail | no | no |
| A1_diagnostics_only | physical_model_status | pass | pass | pass | no | no |
| A1_diagnostics_only | physical_model_failure_reason | pass | pass | pass | no | no |
| A1_diagnostics_only | physical_center_i | pass | pass | pass | no | no |
| A1_diagnostics_only | fine_center_i | pass | pass | pass | no | no |
| A2_anchor_selection_dry_run | physical_model_status | pass | pass | pass | no | no |
| A2_anchor_selection_dry_run | physical_model_failure_reason | pass | pass | pass | no | no |
| A2_anchor_selection_dry_run | physical_center_i | pass | pass | pass | no | no |
| A2_anchor_selection_dry_run | fine_center_i | pass | pass | pass | no | no |
| A3_anchor_stride5_nearest_anchor | physical_model_status | pass | pass | fail | no | no |
| A3_anchor_stride5_nearest_anchor | physical_model_failure_reason | pass | pass | fail | no | no |
| A3_anchor_stride5_nearest_anchor | physical_center_i | pass | pass | fail | no | no |
| A3_anchor_stride5_nearest_anchor | fine_center_i | pass | pass | fail | no | no |
| A4_anchor_stride5_t0_shift | physical_model_status | pass | pass | fail | no | no |
| A4_anchor_stride5_t0_shift | physical_model_failure_reason | pass | pass | fail | no | no |
| A4_anchor_stride5_t0_shift | physical_center_i | pass | pass | fail | no | no |
| A4_anchor_stride5_t0_shift | fine_center_i | pass | pass | fail | no | no |
| A5_anchor_stride5_t0_shift_adaptive_refit | physical_model_status | pass | pass | fail | no | no |
| A5_anchor_stride5_t0_shift_adaptive_refit | physical_model_failure_reason | pass | pass | fail | no | no |
| A5_anchor_stride5_t0_shift_adaptive_refit | physical_center_i | pass | pass | fail | no | no |
| A5_anchor_stride5_t0_shift_adaptive_refit | fine_center_i | pass | pass | fail | no | no |
| A6_A5_obs_downsample256 | physical_model_status | pass | pass | fail | no | no |
| A6_A5_obs_downsample256 | physical_model_failure_reason | pass | pass | fail | no | no |
| A6_A5_obs_downsample256 | physical_center_i | pass | pass | fail | no | no |
| A6_A5_obs_downsample256 | fine_center_i | pass | pass | fail | no | no |

## Diff Summary

| candidate | key | valid_both | one_sided_missing | p90 | p99 | max | within_16 |
|---|---|---:|---:|---:|---:|---:|---:|
| A0D_downsample_only | physical_center_i | 309381 | 0 | 2 | 12 | 153 | 0.995003 |
| A0D_downsample_only | fine_center_i | 309381 | 0 | 2 | 12 | 153 | 0.995003 |
| A0D_downsample_only | physical_model_break_offset_m | 298666 | 7718 | 317.328 | 950.942 | 1305.47 | 0.417851 |
| A0D_downsample_only | physical_model_slope_near_s_per_m | 298666 | 7718 | 0.000113573 | 0.00075991 | 0.00114127 | 0.974809 |
| A1_diagnostics_only | physical_center_i | 309381 | 0 | 0 | 0 | 0 | 1 |
| A1_diagnostics_only | fine_center_i | 309381 | 0 | 0 | 0 | 0 | 1 |
| A1_diagnostics_only | physical_model_break_offset_m | 301030 | 0 | 0 | 0 | 0 | 1 |
| A1_diagnostics_only | physical_model_slope_near_s_per_m | 301030 | 0 | 0 | 0 | 0 | 1 |
| A2_anchor_selection_dry_run | physical_center_i | 309381 | 0 | 0 | 0 | 0 | 1 |
| A2_anchor_selection_dry_run | fine_center_i | 309381 | 0 | 0 | 0 | 0 | 1 |
| A2_anchor_selection_dry_run | physical_model_break_offset_m | 301030 | 0 | 0 | 0 | 0 | 1 |
| A2_anchor_selection_dry_run | physical_model_slope_near_s_per_m | 301030 | 0 | 0 | 0 | 0 | 1 |
| A3_anchor_stride5_nearest_anchor | physical_center_i | 309381 | 0 | 4 | 76 | 222 | 0.963288 |
| A3_anchor_stride5_nearest_anchor | fine_center_i | 309381 | 0 | 4 | 76 | 222 | 0.963288 |
| A3_anchor_stride5_nearest_anchor | physical_model_break_offset_m | 301030 | 1284 | 244.435 | 614.943 | 958.137 | 0.741279 |
| A3_anchor_stride5_nearest_anchor | physical_model_slope_near_s_per_m | 301030 | 1284 | 0.000124275 | 0.00116042 | 0.00538147 | 0.995753 |
| A4_anchor_stride5_t0_shift | physical_center_i | 309381 | 0 | 3 | 47 | 200 | 0.978169 |
| A4_anchor_stride5_t0_shift | fine_center_i | 309381 | 0 | 3 | 47 | 200 | 0.978169 |
| A4_anchor_stride5_t0_shift | physical_model_break_offset_m | 301030 | 1284 | 244.435 | 614.943 | 958.137 | 0.741279 |
| A4_anchor_stride5_t0_shift | physical_model_slope_near_s_per_m | 301030 | 1284 | 0.000124275 | 0.00116042 | 0.00538147 | 0.995753 |
| A5_anchor_stride5_t0_shift_adaptive_refit | physical_center_i | 309381 | 0 | 1 | 7 | 53 | 0.999402 |
| A5_anchor_stride5_t0_shift_adaptive_refit | fine_center_i | 309381 | 0 | 1 | 7 | 53 | 0.999402 |
| A5_anchor_stride5_t0_shift_adaptive_refit | physical_model_break_offset_m | 301030 | 891 | 34.1722 | 493.624 | 888.112 | 0.874345 |
| A5_anchor_stride5_t0_shift_adaptive_refit | physical_model_slope_near_s_per_m | 301030 | 891 | 2.60889e-05 | 0.00113646 | 0.00217216 | 0.997049 |
| A6_A5_obs_downsample256 | physical_center_i | 309381 | 0 | 3 | 13 | 153 | 0.993704 |
| A6_A5_obs_downsample256 | fine_center_i | 309381 | 0 | 3 | 13 | 153 | 0.993704 |
| A6_A5_obs_downsample256 | physical_model_break_offset_m | 299106 | 7766 | 388.595 | 942.385 | 1305.47 | 0.385099 |
| A6_A5_obs_downsample256 | physical_model_slope_near_s_per_m | 299106 | 7766 | 0.000192279 | 0.00124416 | 0.00217216 | 0.974693 |

## Status Count Diff

| candidate | key | available | counts_match | arrays_match | baseline_counts | candidate_counts |
|---|---|---|---|---|---|---|
| A0D_downsample_only | physical_model_status | pass | fail | fail | two_piece_ok=276142; relaxed_segment_ok=24888; fallback_existing_trend=8351; fallback_feasible_clip=0; fallback_robust=0; geometry_invalid=0; insufficient_observations=0; fit_failed=0; physical_disabled=0 | two_piece_ok=279132; relaxed_segment_ok=24888; fallback_existing_trend=5361; fallback_feasible_clip=0; fallback_robust=0; geometry_invalid=0; insufficient_observations=0; fit_failed=0; physical_disabled=0 |
| A0D_downsample_only | physical_model_failure_reason | pass | fail | fail | none=301030; physical_disabled=0; geometry_invalid=0; insufficient_observations=0; fit_failed=8351; prediction_invalid=0 | none=304020; physical_disabled=0; geometry_invalid=0; insufficient_observations=0; fit_failed=5361; prediction_invalid=0 |
| A1_diagnostics_only | physical_model_status | pass | pass | pass | two_piece_ok=276142; relaxed_segment_ok=24888; fallback_existing_trend=8351; fallback_feasible_clip=0; fallback_robust=0; geometry_invalid=0; insufficient_observations=0; fit_failed=0; physical_disabled=0 | two_piece_ok=276142; relaxed_segment_ok=24888; fallback_existing_trend=8351; fallback_feasible_clip=0; fallback_robust=0; geometry_invalid=0; insufficient_observations=0; fit_failed=0; physical_disabled=0 |
| A1_diagnostics_only | physical_model_failure_reason | pass | pass | pass | none=301030; physical_disabled=0; geometry_invalid=0; insufficient_observations=0; fit_failed=8351; prediction_invalid=0 | none=301030; physical_disabled=0; geometry_invalid=0; insufficient_observations=0; fit_failed=8351; prediction_invalid=0 |
| A2_anchor_selection_dry_run | physical_model_status | pass | pass | pass | two_piece_ok=276142; relaxed_segment_ok=24888; fallback_existing_trend=8351; fallback_feasible_clip=0; fallback_robust=0; geometry_invalid=0; insufficient_observations=0; fit_failed=0; physical_disabled=0 | two_piece_ok=276142; relaxed_segment_ok=24888; fallback_existing_trend=8351; fallback_feasible_clip=0; fallback_robust=0; geometry_invalid=0; insufficient_observations=0; fit_failed=0; physical_disabled=0 |
| A2_anchor_selection_dry_run | physical_model_failure_reason | pass | pass | pass | none=301030; physical_disabled=0; geometry_invalid=0; insufficient_observations=0; fit_failed=8351; prediction_invalid=0 | none=301030; physical_disabled=0; geometry_invalid=0; insufficient_observations=0; fit_failed=8351; prediction_invalid=0 |
| A3_anchor_stride5_nearest_anchor | physical_model_status | pass | fail | fail | two_piece_ok=276142; relaxed_segment_ok=24888; fallback_existing_trend=8351; fallback_feasible_clip=0; fallback_robust=0; geometry_invalid=0; insufficient_observations=0; fit_failed=0; physical_disabled=0 | two_piece_ok=277417; relaxed_segment_ok=24897; fallback_existing_trend=7067; fallback_feasible_clip=0; fallback_robust=0; geometry_invalid=0; insufficient_observations=0; fit_failed=0; physical_disabled=0 |
| A3_anchor_stride5_nearest_anchor | physical_model_failure_reason | pass | fail | fail | none=301030; physical_disabled=0; geometry_invalid=0; insufficient_observations=0; fit_failed=8351; prediction_invalid=0 | none=302314; physical_disabled=0; geometry_invalid=0; insufficient_observations=0; fit_failed=7067; prediction_invalid=0 |
| A4_anchor_stride5_t0_shift | physical_model_status | pass | fail | fail | two_piece_ok=276142; relaxed_segment_ok=24888; fallback_existing_trend=8351; fallback_feasible_clip=0; fallback_robust=0; geometry_invalid=0; insufficient_observations=0; fit_failed=0; physical_disabled=0 | two_piece_ok=277417; relaxed_segment_ok=24897; fallback_existing_trend=7067; fallback_feasible_clip=0; fallback_robust=0; geometry_invalid=0; insufficient_observations=0; fit_failed=0; physical_disabled=0 |
| A4_anchor_stride5_t0_shift | physical_model_failure_reason | pass | fail | fail | none=301030; physical_disabled=0; geometry_invalid=0; insufficient_observations=0; fit_failed=8351; prediction_invalid=0 | none=302314; physical_disabled=0; geometry_invalid=0; insufficient_observations=0; fit_failed=7067; prediction_invalid=0 |
| A5_anchor_stride5_t0_shift_adaptive_refit | physical_model_status | pass | fail | fail | two_piece_ok=276142; relaxed_segment_ok=24888; fallback_existing_trend=8351; fallback_feasible_clip=0; fallback_robust=0; geometry_invalid=0; insufficient_observations=0; fit_failed=0; physical_disabled=0 | two_piece_ok=277033; relaxed_segment_ok=24888; fallback_existing_trend=7460; fallback_feasible_clip=0; fallback_robust=0; geometry_invalid=0; insufficient_observations=0; fit_failed=0; physical_disabled=0 |
| A5_anchor_stride5_t0_shift_adaptive_refit | physical_model_failure_reason | pass | fail | fail | none=301030; physical_disabled=0; geometry_invalid=0; insufficient_observations=0; fit_failed=8351; prediction_invalid=0 | none=301921; physical_disabled=0; geometry_invalid=0; insufficient_observations=0; fit_failed=7460; prediction_invalid=0 |
| A6_A5_obs_downsample256 | physical_model_status | pass | fail | fail | two_piece_ok=276142; relaxed_segment_ok=24888; fallback_existing_trend=8351; fallback_feasible_clip=0; fallback_robust=0; geometry_invalid=0; insufficient_observations=0; fit_failed=0; physical_disabled=0 | two_piece_ok=280060; relaxed_segment_ok=24888; fallback_existing_trend=4433; fallback_feasible_clip=0; fallback_robust=0; geometry_invalid=0; insufficient_observations=0; fit_failed=0; physical_disabled=0 |
| A6_A5_obs_downsample256 | physical_model_failure_reason | pass | fail | fail | none=301030; physical_disabled=0; geometry_invalid=0; insufficient_observations=0; fit_failed=8351; prediction_invalid=0 | none=304948; physical_disabled=0; geometry_invalid=0; insufficient_observations=0; fit_failed=4433; prediction_invalid=0 |

## Gate Summary

| candidate | gate | key | value | threshold | result |
|---|---|---|---:|---:|---|
| A0D_downsample_only | max_p90_abs_diff_samples | physical_center_i | 2 | 8 | pass |
| A0D_downsample_only | max_p90_abs_diff_samples | fine_center_i | 2 | 8 | pass |
| A0D_downsample_only | min_within_16_sample_rate | physical_center_i | 0.995003 | 0.97 | pass |
| A0D_downsample_only | min_within_16_sample_rate | fine_center_i | 0.995003 | 0.97 | pass |
| A0D_downsample_only | min_speedup_physics_total |  | 1.05146 | 1 | pass |
| A1_diagnostics_only | exact_match_required | physical_center_i | pass | pass | pass |
| A1_diagnostics_only | exact_match_required | fine_center_i | pass | pass | pass |
| A1_diagnostics_only | exact_match_required | physical_model_status | pass | pass | pass |
| A1_diagnostics_only | exact_match_required | physical_model_failure_reason | pass | pass | pass |
| A1_diagnostics_only | status_counts_match | physical_model_status | pass | pass | pass |
| A1_diagnostics_only | status_counts_match | physical_model_failure_reason | pass | pass | pass |
| A2_anchor_selection_dry_run | exact_match_required | physical_center_i | pass | pass | pass |
| A2_anchor_selection_dry_run | exact_match_required | fine_center_i | pass | pass | pass |
| A2_anchor_selection_dry_run | exact_match_required | physical_model_status | pass | pass | pass |
| A2_anchor_selection_dry_run | exact_match_required | physical_model_failure_reason | pass | pass | pass |
| A2_anchor_selection_dry_run | status_counts_match | physical_model_status | pass | pass | pass |
| A2_anchor_selection_dry_run | status_counts_match | physical_model_failure_reason | pass | pass | pass |
| A3_anchor_stride5_nearest_anchor | max_p90_abs_diff_samples | physical_center_i | 4 | 8 | pass |
| A3_anchor_stride5_nearest_anchor | max_p90_abs_diff_samples | fine_center_i | 4 | 8 | pass |
| A3_anchor_stride5_nearest_anchor | min_within_16_sample_rate | physical_center_i | 0.963288 | 0.97 | fail |
| A3_anchor_stride5_nearest_anchor | min_within_16_sample_rate | fine_center_i | 0.963288 | 0.97 | fail |
| A3_anchor_stride5_nearest_anchor | min_speedup_physics_total |  | 1.11299 | 1 | pass |
| A4_anchor_stride5_t0_shift | max_p90_abs_diff_samples | physical_center_i | 3 | 8 | pass |
| A4_anchor_stride5_t0_shift | max_p90_abs_diff_samples | fine_center_i | 3 | 8 | pass |
| A4_anchor_stride5_t0_shift | min_within_16_sample_rate | physical_center_i | 0.978169 | 0.97 | pass |
| A4_anchor_stride5_t0_shift | min_within_16_sample_rate | fine_center_i | 0.978169 | 0.97 | pass |
| A4_anchor_stride5_t0_shift | min_speedup_physics_total |  | 1.11034 | 1 | pass |
| A5_anchor_stride5_t0_shift_adaptive_refit | max_p90_abs_diff_samples | physical_center_i | 1 | 8 | pass |
| A5_anchor_stride5_t0_shift_adaptive_refit | max_p90_abs_diff_samples | fine_center_i | 1 | 8 | pass |
| A5_anchor_stride5_t0_shift_adaptive_refit | min_within_16_sample_rate | physical_center_i | 0.999402 | 0.97 | pass |
| A5_anchor_stride5_t0_shift_adaptive_refit | min_within_16_sample_rate | fine_center_i | 0.999402 | 0.97 | pass |
| A5_anchor_stride5_t0_shift_adaptive_refit | min_speedup_physics_total |  | 1.05153 | 1 | pass |
| A6_A5_obs_downsample256 | max_p90_abs_diff_samples | physical_center_i | 3 | 8 | pass |
| A6_A5_obs_downsample256 | max_p90_abs_diff_samples | fine_center_i | 3 | 8 | pass |
| A6_A5_obs_downsample256 | min_within_16_sample_rate | physical_center_i | 0.993704 | 0.97 | pass |
| A6_A5_obs_downsample256 | min_within_16_sample_rate | fine_center_i | 0.993704 | 0.97 | pass |
| A6_A5_obs_downsample256 | min_speedup_physics_total |  | 1.0573 | 1 | pass |
