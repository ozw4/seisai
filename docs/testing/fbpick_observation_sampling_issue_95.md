# FBPick Observation Sampling Runtime Check

This records the Issue #95 Arakawa one-file runtime check for offset-bin
observation downsampling before RANSAC fits.

Date: 2026-05-12

The check reused the existing Arakawa coarse artifact:

```text
proc/arakawa/coarse/Arakawa2026__fdata_hset_ARA26_Vib.coarse.npz
```

Temporary configs:

```text
.work/codex/arakawa_issue95_A5.yaml
.work/codex/arakawa_issue95_A6.yaml
```

Commands:

```bash
python -m cli.run_fbpick_physics --config .work/codex/arakawa_issue95_A5.yaml
python -m cli.run_fbpick_physics --config .work/codex/arakawa_issue95_A6.yaml
```

A0 output was taken from the existing full baseline artifact:

```text
proc/arakawa/runtime_runs/A0_full/robust/Arakawa2026__fdata_hset_ARA26_Vib.robust.npz
```

Because A0 was run with runtime diagnostics disabled, the A1 diagnostics-only
full run summary is used for A0 runtime numbers. Issue #90 records that A1
matched A0 exactly for physical center, fine center, status, and failure reason.

## Outputs

```text
proc/arakawa/runtime_runs/A5_anchor_stride5_t0_shift_adaptive_refit/robust/Arakawa2026__fdata_hset_ARA26_Vib.robust.npz
proc/arakawa/runtime_runs/A5_anchor_stride5_t0_shift_adaptive_refit/robust/Arakawa2026__fdata_hset_ARA26_Vib.physics_runtime_summary.json
proc/arakawa/runtime_runs/A6_A5_obs_downsample256/robust/Arakawa2026__fdata_hset_ARA26_Vib.robust.npz
proc/arakawa/runtime_runs/A6_A5_obs_downsample256/robust/Arakawa2026__fdata_hset_ARA26_Vib.physics_runtime_summary.json
```

Comparison artifacts:

```text
.work/codex/compare_issue95_A5_vs_A0.json
.work/codex/compare_issue95_A6_vs_A5.json
.work/codex/compare_issue95_A6_vs_A0.json
```

## Runtime Summary

| run | n_fit_calls | n_adaptive_refit_calls | obs_before p50/p90/p99 | obs_after p50/p90/p99 | ransac_fit p50/p90 sec | ransac_fit_total_sec | physics_total_sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0_full (A1 diagnostics runtime) | 1757 | n/a | n/a | 845.0 / 5189.8 / 5470.88 | 0.128462 / 0.213335 | 212.457188 | 1621.246923 |
| A5_anchor_stride5_t0_shift_adaptive_refit | 1344 | 105 | 652.0 / 4970.4 / 5475.14 | 652.0 / 4970.4 / 5475.14 | 0.111399 / 0.253548 | 162.635240 | 2612.649938 |
| A6_A5_obs_downsample256 | 1274 | 94 | 559.5 / 4950.8 / 5469.27 | 64.0 / 119.4 / 236.35 | 0.033237 / 0.042373 | 45.394405 | 2585.422297 |

Shell wall times:

```text
A5: 43m36.515s
A6: 43m08.977s
```

A6 vs A5:

```text
obs_downsample_rate_p50: 0.8856098698543928
obs_downsample_rate_p90: 0.987072794543054
ransac_fit_total speedup: 3.582715553648784x
physics_total speedup: 1.0105312164182796x
fit_failed count: 7460 -> 4433
insufficient_observations count: 0 -> 0
```

## Center Diffs

Physical center diff:

| comparison | p50 | p90 | p95 | p99 | max | within4 | within8 | within16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A5 vs A0 | 3 | 7 | 10 | 29 | 174 | 0.773111 | 0.934220 | 0.977284 |
| A6 vs A5 | 0 | 2 | 4 | 12 | 153 | 0.955744 | 0.982142 | 0.994188 |
| A6 vs A0 | 3 | 7 | 10 | 35 | 179 | 0.777404 | 0.937453 | 0.973832 |

`fine_center_i` matched the same diff percentiles in these comparisons.

## Status Counts

A6 vs A5 `physical_model_status`:

```text
A5 fallback_existing_trend: 7460
A6 fallback_existing_trend: 4433
A5 relaxed_segment_ok: 24888
A6 relaxed_segment_ok: 24888
A5 two_piece_ok: 277033
A6 two_piece_ok: 280060
```

A6 vs A5 `physical_model_failure_reason`:

```text
A5 fit_failed: 7460
A6 fit_failed: 4433
A5 insufficient_observations: 0
A6 insufficient_observations: 0
A5 none: 301921
A6 none: 304948
```

## Notes

A6 reduces observations passed to RANSAC and lowers RANSAC fit time
substantially. End-to-end `physics_total_sec` improves only slightly versus A5
on this run, and both A5/A6 are slower than the existing full A0/A1 runtime
summary because adaptive refit and non-RANSAC overhead dominate this Arakawa
configuration.
