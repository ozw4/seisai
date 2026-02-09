# MLflow Tracking Spec (seisai-engine)

## Purpose
Define the minimal integration points and required information for MLflow tracking
in seisai-engine so future implementation is consistent and low-risk.

## Hook Point (Minimal Integration)
- Single entry point: `packages/seisai-engine/src/seisai_engine/pipelines/common/train_skeleton.py`
- Integrate inside `run_train_skeleton`:
  - Start tracking once per process (1 run = 1 process invocation)
  - Log params/tags/artifacts at start
  - Log metrics per epoch
  - Log best artifacts only when best improves
  - End run as FINISHED or FAILED

## Available Information (from TrainSkeletonSpec)
- `pipeline`: `spec.pipeline`
- `cfg` (full config dict): `spec.cfg`
- `base_dir`: `spec.base_dir` (YAML base_dir; directory of the config file)
- `out_dir`: `spec.out_dir`
- Seeds/epochs/batch/etc: `spec.seed_train`, `spec.seed_infer`, `spec.epochs`,
  `spec.train_batch_size`, `spec.samples_per_epoch`, `spec.infer_batch_size`,
  `spec.infer_max_batches`
- Model signature: `spec.model_sig` (e.g., backbone/in_chans/out_chans)
- Optimizer: `spec.optimizer` (lr, weight_decay from param_groups)
- Best checkpoint path: `out_dir/ckpt/best.pt`
- Vis epoch dir: `out_dir/<vis_subdir>/epoch_xxxx/`

## Run Definition
- 1 process invocation = 1 MLflow run
- Resume always creates a new run
  - Resume info is captured in tags (`resume_from_path`, `parent_run_id` if known)

## Experiment / Run Naming
- experiment = `seisai/<pipeline>`
- run_name = `{timestamp}__{exp_name}__s{seed}`
  - No collision avoidance

## Tracking URI
- Default: `file:./mlruns`
- Relative paths must be resolved against `spec.base_dir` (YAML base_dir), then passed
  as absolute `file:/abs/path` URI to MLflow
- `TrainSkeletonSpec` must include `base_dir: Path`, and each `*/train.py` passes
  the `base_dir` returned by `load_cfg_with_base_dir`

## Tracking Enablement
- Tracking is controlled by `cfg['tracking']`.


## Tags (small, searchable)
- `pipeline`
- `git_sha`, `dirty` (true/false)
- `user`
- `data_id` (short SHA256 of normalized manifest string)
- `data_id_human` (e.g., `nfiles=<N>`)
- `data_nfiles`
- `resume_from_path` (if present)
- `parent_run_id` (if known)

## Params (minimal common set)
- `train/batch_size`, `train/epochs`, `train/samples_per_epoch`
- `optimizer/lr`, `optimizer/weight_decay`
  - Read from `optimizer.param_groups[0]`
  - If multiple param_groups have differing values, do not log these params
- `model/backbone`, `model/in_chans`, `model/out_chans`
- `seeds/seed_train`, `seeds/seed_infer`

## Metrics (per epoch, step=epoch)
- `train/loss`
- `infer/loss`
- `lr` (if available)

## Artifacts (reproducibility)
- `config.original.yaml`
- `config.resolved.yaml` (if available)
- `git.txt`, `env.txt`, `stdout.log` (optional)
- `data_manifest.json`
- `ckpt/best.pt` (only when best updates)
- `vis/epoch_xxxx/` (only when best updates, max 50 files)
- Overlong values storage:
  - Save overlong tag/param values to `out_dir/tracking/overlong_values.json`
    and log it as an MLflow artifact.

## Best Definition
- Best is defined by minimum `infer/loss`
- If `infer/loss` is absent, do not compute best (no fallback)
- Best update detection (minimal change): compute
  `did_update = (best_infer_loss is None) or (infer_loss < best_infer_loss)`
  before calling `maybe_save_best_min`, and only log MLflow best artifacts when
  `did_update` is true

## Visualization Limits
- Only log vis artifacts when best updates
- Max 50 files; extra files remain in `out_dir` but are not logged to MLflow
- Selection rule: recursively list regular files under `vis_epoch_dir`, sort by path, and
  log only the first `vis_max_files` entries

## Data ID / Manifest
- Collect `cfg['paths']` entries where key matches `*_files` and value is `list[str]`
  (ignore `str` values and non-`*_files` keys such as `out_dir`)
- Manifest must include the key name (e.g., `segy_files`) for each collected list
- Sort each list (string sort) and build a normalized manifest
- Save `data_manifest.json` to artifacts
- `data_id` = first 12 hex chars of sha256(normalized_manifest_string)
- `data_id_human` can be `nfiles=<N>`

## Failure Handling
- On exception, close MLflow run with status FAILED
- Re-raise exceptions (no swallowing)

## Sanitize Rules
- MLflow keys allow only `a-zA-Z0-9_./-` and `/` separators
- Invalid characters replaced with `_`
- Overlong values should be skipped in tags/params and stored in artifacts
- Value length policy (bytes):
  - tag value: >256 bytes -> artifact
  - param value: >1024 bytes -> artifact
  - If value contains newlines or looks like JSON/YAML, store as artifact regardless
