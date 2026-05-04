# FBPick Coarse Coverage Evaluation

`fbpick-coarse` is successful when the true first break lands inside the
downstream fine-stage search window. The coarse pick is a search center, not the
final sample-accurate answer, so MAE alone is not enough.

The primary metric is:

```text
coverage@K = mean(abs(coarse_pick_i - fb_i) <= K)
```

For the fine stage, use `K = eval.fine_window_half_samples`:

```text
coverage_fine_window = mean(abs(coarse_pick_i - fb_i) <= fine_window_half_samples)
```

## CLI

Use a config file:

```bash
python cli/run_fbpick_coarse_eval.py \
  --config examples/config_eval_fbpick_coarse.yaml
```

Minimal config:

```yaml
paths:
  coarse_files: /path/to/coarse_files.txt
  fb_files: /path/to/fb_files.txt
  out_dir: /path/to/eval_out

eval:
  fine_window_half_samples: 128
  coverage_thresholds_samples: [32, 64, 128, 256]
  coverage_thresholds_ms: [10, 20, 50, 100]
  gap_neighborhood_traces: 10
  confidence_bins: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
  make_figures: true
```

`paths.coarse_files` and `paths.fb_files` can be lists or listfiles. Listfile
entries are paired by line number.

## Inputs

Each `.coarse.npz` must contain restored original-coordinate fields:

- `coarse_pick_i`
- `coarse_pick_t_sec`
- `coarse_pmax`
- `dt_sec`
- `n_samples_orig`
- `n_traces`
- `trace_indices`
- `offsets_m`
- `ffid_values`
- `chno_values`

The evaluator checks shape, finite confidence/time values, sample-index bounds,
`trace_indices == np.arange(n_traces)`, and
`coarse_pick_t_sec ~= coarse_pick_i * dt_sec`.

Ground-truth first-break files are loaded as 1D numpy arrays. Invalid labels are
excluded from metrics and counted in reports:

- `fb_i < 0`
- `NaN` / `inf`
- `fb_i >= n_samples_orig`

## Outputs

The evaluator writes:

- `per_gather.csv`: one row per coarse/FB pair
- `summary.csv` and `summary.json`: all valid traces aggregated globally
- `confidence_bins.csv`: error and coverage grouped by `coarse_pmax`
- `gap_neighborhood.csv`: gap-boundary traces vs non-gap traces
- `per_segment.csv`: written only when segment start/stop metadata is present
- `figures/*.png`: optional plots when `eval.make_figures: true`

Important columns:

- `mae_samples`, `p50_abs_samples`, `p90_abs_samples`, `p95_abs_samples`,
  `p99_abs_samples`, `max_abs_samples`, `bias_samples`
- `mae_ms`, `p50_abs_ms`, `p90_abs_ms`, `p95_abs_ms`, `p99_abs_ms`,
  `max_abs_ms`, `bias_ms`
- `coverage_32`, `coverage_64`, `coverage_128`, `coverage_256`
- `coverage_fine_window`
- `coverage_ms_10`, `coverage_ms_20`, `coverage_ms_50`, `coverage_ms_100`
- `mean_confidence`, `median_confidence`
- `failure_rate_fine_window`

`bias_samples` and `bias_ms` are signed means of `coarse_pick_i - fb_i`; positive
values mean the coarse pick is later than the label on average.

## Gap Neighborhood

If segment metadata is not present in the `.coarse.npz`, the evaluator re-detects
offset gaps from `offsets_m` using `eval.gap_ratio` and `eval.min_gap_m`. These
defaults match the coarse example configs:

```yaml
gap_ratio: 5.0
min_gap_m: null
```

For a boundary between trace `b - 1` and `b`, `gap_neighborhood_traces: N`
includes:

```text
[b - N - 1, b + N]
```

clipped to the gather range. With a boundary between traces 9 and 10 and `N = 2`,
the gap neighborhood is traces 7, 8, 9, 10, 11, and 12.

## Segment Metrics

`per_segment.csv` is written when `.coarse.npz` includes segment start/stop
arrays. The preferred keys are:

- `segment_start_pos`
- `segment_stop_pos`
- optional `segment_ids`

When segment metadata is absent, `per_segment.csv` is skipped and
`summary.json` records that status.
