# site54 OOF CV runbook

`/workspace/proc/fbpick/site54/oof` is the canonical root for the site54 54-survey 6-fold OOF CV experiment. Keep new fold lists, configs, logs, stage outputs, manifests, and aggregate reports under this root so coarse, physics, fine, and eval artifacts can be traced as one CV run.

## Canonical paths

- CV root: `/workspace/proc/fbpick/site54/oof`
- Fold list root: `/workspace/proc/fbpick/site54/oof/fold_lists`
- Run root: `/workspace/proc/fbpick/site54/oof/runs/<run_id>`
- Config root: `/workspace/proc/fbpick/site54/oof/runs/<run_id>/configs`

The canonical fold list layout is:

```text
fold_lists/
  README.md
  lists/
    all_sgy.txt
    all_fb.txt
  folds/
    fold00/
      train_sgy.txt
      train_fb.txt
      inner_valid_sgy.txt
      inner_valid_fb.txt
      heldout_sgy.txt
      heldout_fb.txt
      train_all_nonheldout_sgy.txt
      train_all_nonheldout_fb.txt
      *_names.txt
    fold01/
    ...
```

`fold_lists/fold_summary.csv`, `fold_lists/fold_assignments.csv`, `fold_lists/site54_manifest.csv`, and per-fold `fold_meta.json` are tracked metadata for this split.

## Legacy paths

The following paths are deprecated compatibility paths retained only for existing configs and historical outputs. Do not use them for new runs, new configs, or new scripts.

| Old path | New path |
| --- | --- |
| `oof/folds` | `oof/fold_lists/folds` |
| `oof/site54_oof_6fold_lists/lists` | `oof/fold_lists/lists` |
| `oof/coarse_foldXX_train_out` | `oof/runs/<run_id>/foldXX/01_coarse_train` |
| `oof/coarse_oof/foldXX` | `oof/runs/<run_id>/foldXX/02_coarse_infer` |
| `oof/robust_oof/foldXX` | `oof/runs/<run_id>/foldXX/03_physics` |
| `oof/physics_qc/foldXX` | `oof/runs/<run_id>/foldXX/04_physics_qc` |
| `site54/fbpick_fine_train_oof_foldXX_out` | `oof/runs/<run_id>/foldXX/06_fine_train` |
| `oof/fine_infer/foldXX` | `oof/runs/<run_id>/foldXX/07_fine_infer` |
| `oof/fine_eval` | `oof/runs/<run_id>/aggregate/08_eval` |

Deprecated for new runs:

- `proc/fbpick/site54/oof/coarse_oof`
- `proc/fbpick/site54/oof/robust_oof`
- `proc/fbpick/site54/oof/fine_infer`
- `proc/fbpick/site54/oof/fine_eval`
- `proc/fbpick/site54/oof/configs`
- `proc/fbpick/site54/oof/lists`
- `proc/fbpick/site54/oof/logs`
- `proc/fbpick/site54/fbpick_fine_train_oof_foldXX_out`

New fold-list references must use `/workspace/proc/fbpick/site54/oof/fold_lists`.

## Clean Generated Artifacts

Before a clean rerun, inspect generated legacy artifacts without deleting them:

```bash
python proc/fbpick/site54/oof/scripts/clean_generated_artifacts.py \
  --cv-root /workspace/proc/fbpick/site54/oof \
  --legacy-only \
  --dry-run
```

Delete only the selected legacy artifacts after inspection:

```bash
python proc/fbpick/site54/oof/scripts/clean_generated_artifacts.py \
  --cv-root /workspace/proc/fbpick/site54/oof \
  --legacy-only \
  --yes
```

`--legacy-only` targets deprecated generated paths such as `oof/configs`,
`oof/lists`, `oof/logs`, old fold-list mirrors, old stage output roots, and
the old site54 fine train/infer output roots. It does not target `README.md`,
`fold_lists/`, or `scripts/`.

Run-scoped outputs are deleted only when explicitly selected:

```bash
python proc/fbpick/site54/oof/scripts/clean_generated_artifacts.py \
  --cv-root /workspace/proc/fbpick/site54/oof \
  --run-id baseline_physical_center \
  --dry-run
```

Use `--all-runs` to select every direct child under `oof/runs/`.

## Stage order

Use these stage names for manifests, logs, and run directories:

```text
01_coarse_train
02_coarse_infer
03_physics
04_physics_qc
05_collect_oof_lists
06_fine_train
07_fine_infer
08_eval
```

## Stage details

| Stage | Purpose | Inputs | Outputs | Representative command | Smoke vs full | Heldout use |
| --- | --- | --- | --- | --- | --- | --- |
| `01_coarse_train` | Train one coarse model per fold. | `fold_lists/folds/foldXX/train_sgy.txt`, `train_fb.txt`, `inner_valid_sgy.txt`, `inner_valid_fb.txt`; `runs/<run_id>/configs/foldXX/01_coarse_train*.yaml`. | Full: `runs/<run_id>/foldXX/01_coarse_train/`; smoke: `runs/<run_id>/foldXX/01_coarse_train_smoke/`. | `RUN_ID=baseline_physical_center proc/fbpick/site54/oof/scripts/run_coarse_train_fold.sh fold00 0 full` | Smoke uses `01_coarse_train_smoke.yaml` and short training in its own output directory. Full uses `01_coarse_train.yaml`. | Heldout lists must not be used for training, validation, model selection, or early stopping. |
| `02_coarse_infer` | Produce coarse predictions for each fold's heldout surveys. | `fold_lists/folds/foldXX/heldout_sgy.txt`; checkpoint from `runs/<run_id>/foldXX/01_coarse_train/ckpt/best.pt`. | `runs/<run_id>/foldXX/02_coarse_infer/`. | `RUN_ID=baseline_physical_center proc/fbpick/site54/oof/scripts/run_coarse_infer_fold.sh fold00 0` | Smoke should run on the smoke checkpoint and a temporary run directory. Full runs every heldout SGY for the fold. | Heldout SGY may be used only as inference input. Heldout FB is still not used. |
| `03_physics` | Convert heldout coarse predictions to physical-center robust NPZ files. | `fold_lists/folds/foldXX/heldout_sgy.txt`; heldout coarse NPZ from `runs/<run_id>/foldXX/02_coarse_infer`; `runs/<run_id>/configs/foldXX/03_physics.yaml`. | `runs/<run_id>/foldXX/03_physics/`. | `RUN_ID=baseline_physical_center proc/fbpick/site54/oof/scripts/run_physics_fold.sh fold00` | Smoke should target a temporary run directory and a small heldout subset. Full runs all heldout predictions. | Heldout FB is not used. Heldout SGY and coarse NPZ are inference inputs only. |
| `04_physics_qc` | Check robust outputs against heldout SGY/FB lists and summarize missing or mismatched outputs. | `fold_lists/folds/foldXX/heldout_sgy.txt`, `heldout_fb.txt`; coarse NPZ from `runs/<run_id>/foldXX/02_coarse_infer`; robust NPZ from `runs/<run_id>/foldXX/03_physics`; `runs/<run_id>/configs/foldXX/04_physics_qc.yaml`. | `runs/<run_id>/foldXX/04_physics_qc/`. | `RUN_ID=baseline_physical_center proc/fbpick/site54/oof/scripts/run_physics_fold.sh fold00` | Smoke checks one fold or a temporary subset. Full checks all six folds. | Heldout FB may be used for QC metrics after inference products already exist. Do not feed QC results back into training. |
| `05_collect_oof_lists` | Collect coarse and robust OOF NPZ paths in canonical all-SGY order for fine training. | `fold_lists/lists/all_sgy.txt`, `all_fb.txt`; `runs/<run_id>/foldXX/02_coarse_infer/*.coarse.npz`; `runs/<run_id>/foldXX/03_physics/*.robust.npz`. | `runs/<run_id>/aggregate/05_collect_oof_lists/`. | `python proc/fbpick/site54/oof/scripts/collect_oof_robust_lists.py --run-id baseline_physical_center` | Smoke should collect from a smoke run directory. Full requires all 54 coarse and robust outputs. | Uses all heldout inference outputs as OOF features. No model fitting happens in this stage. |
| `06_fine_train` | Train one fine model per fold using non-heldout surveys and their OOF robust features. | `runs/<run_id>/aggregate/05_collect_oof_lists/fine_fold_lists/foldXX/train_sgy.txt`, `train_fb.txt`, `train_robust.txt`; checkpoint selection uses `inner_valid_*` by default; config `runs/<run_id>/configs/foldXX/06_fine_train*.yaml`. | `runs/<run_id>/foldXX/06_fine_train/`; smoke: `runs/<run_id>/foldXX/06_fine_train_smoke/`. | `RUN_ID=baseline_physical_center proc/fbpick/site54/oof/scripts/run_fine_train_fold.sh fold00 0 full` | Smoke uses `06_fine_train_smoke.yaml` and short training in its own output directory. Full uses `06_fine_train.yaml`. | Heldout lists must not be used for training, validation, model selection, or early stopping. |
| `07_fine_infer` | Run each fold's fine model on that fold's heldout surveys. | `runs/<run_id>/aggregate/05_collect_oof_lists/fine_fold_lists/foldXX/heldout_sgy.txt`, `heldout_robust.txt`, `heldout_coarse.txt`; checkpoint from `06_fine_train`. | `runs/<run_id>/foldXX/07_fine_infer/`. | `RUN_ID=baseline_physical_center proc/fbpick/site54/oof/scripts/run_fine_infer_fold.sh fold00 0` | Smoke uses one fold or a small heldout subset. Full writes predictions for all heldout surveys in all folds. | Heldout SGY, robust NPZ, and coarse NPZ are inference inputs. Heldout FB is not an inference config input; use it only in `08_eval` for final reporting. |
| `08_eval` | Aggregate OOF prediction quality across folds. | `runs/<run_id>/aggregate/05_collect_oof_lists/fine_fold_lists/foldXX/heldout_sgy.txt`, `heldout_fb.txt`; fine predictions from `07_fine_infer`; optional coarse/robust stages. | `runs/<run_id>/aggregate/08_eval/`. | `python proc/fbpick/site54/oof/scripts/evaluate_fine_oof.py --run-id baseline_physical_center` | Smoke evaluates one fold or subset. Full evaluates all six folds and all requested stages. | Heldout FB is allowed for final reporting only. Evaluation choices must not be fed back into the same CV run's training. |

## Unified Rerun Entry Point

Use `scripts/run_site54_oof_cv.py` for clean reruns. It derives all run-scoped paths once and passes them explicitly to each stage:

```bash
python proc/fbpick/site54/oof/scripts/run_site54_oof_cv.py \
  --run-id baseline_physical_center \
  --stage all
```

Supported high-level stages are:

```text
prepare_configs
coarse_train
coarse_infer
physics
collect
fine_configs
fine_train
fine_infer
eval
check
```

Use `--from-stage coarse_train --to-stage physics` for ranges, `--fold fold00` for fold-scoped stages, and `--dry-run` to print commands without executing them. `collect`, `fine_configs`, `eval`, and `check` require `--fold all`. The runner writes `runs/<run_id>/manifest.yaml` for non-dry runs and updates completed or failed stage status as stages finish.

## Run layout

New CV outputs should use this layout:

```text
runs/<run_id>/
  manifest.yaml
  configs/
    fold00/
      01_coarse_train.yaml
      01_coarse_train_smoke.yaml
      02_coarse_infer.yaml
      03_physics.yaml
      04_physics_qc.yaml
      06_fine_train.yaml
      06_fine_train_smoke.yaml
      07_fine_infer.yaml
  logs/
    launcher/
    fold00/
      01_coarse_train.log
      02_coarse_infer.log
      03_physics.log
      04_physics_qc.log
      06_fine_train.log
      07_fine_infer.log
  fold00/
    01_coarse_train/
    01_coarse_train_smoke/
    02_coarse_infer/
    03_physics/
    04_physics_qc/
    06_fine_train/
    07_fine_infer/
  fold01/
  ...
  aggregate/
    05_collect_oof_lists/
      fine_fold_lists/
        fold00/
    08_eval/
```

Minimum `manifest.yaml`:

```yaml
run_id: baseline_physical_center
cv_root: /workspace/proc/fbpick/site54/oof
fold_list_root: /workspace/proc/fbpick/site54/oof/fold_lists
config_root: /workspace/proc/fbpick/site54/oof/runs/baseline_physical_center/configs
run_root: /workspace/proc/fbpick/site54/oof/runs/baseline_physical_center
folds:
  - fold00
  - fold01
  - fold02
  - fold03
  - fold04
  - fold05
stages:
  01_coarse_train: enabled
  02_coarse_infer: enabled
  03_physics: enabled
  04_physics_qc: enabled
  05_collect_oof_lists: enabled
  06_fine_train: enabled
  07_fine_infer: enabled
  08_eval: enabled
notes: ""
```

## Generated Configs

The formal generated config layout is `runs/<run_id>/configs/foldXX/<stage>.yaml`:

```text
runs/<run_id>/configs/fold00/01_coarse_train.yaml
runs/<run_id>/configs/fold00/02_coarse_infer.yaml
runs/<run_id>/configs/fold00/03_physics.yaml
runs/<run_id>/configs/fold00/04_physics_qc.yaml
runs/<run_id>/configs/fold00/06_fine_train.yaml
runs/<run_id>/configs/fold00/07_fine_infer.yaml
```

Flat generated config names such as `config_train_fbpick_coarse_fold00.yaml`, `config_infer_fbpick_coarse_fold00_heldout.yaml`, `config_run_fbpick_physics_fold00_heldout.yaml`, `config_run_fbpick_physics_qc_fold00.yaml`, and `config_train_fbpick_fine_oof_fold00.yaml` are legacy compatibility artifacts. The config generators write them only when `--legacy-flat-configs true` is passed, and normal run commands must use the run-scoped config layout above.

## Coarse And Physics Run Commands

Generate run-scoped configs for coarse train/infer, physics, and physics QC:

```bash
python proc/fbpick/site54/oof/scripts/make_coarse_fold_configs.py \
  --cv-root /workspace/proc/fbpick/site54/oof \
  --run-id baseline_physical_center \
  --run-root /workspace/proc/fbpick/site54/oof/runs/baseline_physical_center \
  --fold-list-root /workspace/proc/fbpick/site54/oof/fold_lists \
  --config-root /workspace/proc/fbpick/site54/oof/runs/baseline_physical_center/configs \
  --legacy-flat-configs false

python proc/fbpick/site54/oof/scripts/make_physics_fold_configs.py \
  --cv-root /workspace/proc/fbpick/site54/oof \
  --run-id baseline_physical_center \
  --run-root /workspace/proc/fbpick/site54/oof/runs/baseline_physical_center \
  --fold-list-root /workspace/proc/fbpick/site54/oof/fold_lists \
  --config-root /workspace/proc/fbpick/site54/oof/runs/baseline_physical_center/configs \
  --legacy-flat-configs false \
  --overwrite
```

Run one fold:

```bash
RUN_ID=baseline_physical_center \
  bash proc/fbpick/site54/oof/scripts/run_coarse_train_fold.sh fold00

RUN_ID=baseline_physical_center \
  bash proc/fbpick/site54/oof/scripts/run_coarse_infer_fold.sh fold00

RUN_ID=baseline_physical_center \
  bash proc/fbpick/site54/oof/scripts/run_physics_fold.sh fold00
```

`run_physics_fold.sh` runs `03_physics` and then `04_physics_qc`. Logs are written under `runs/<run_id>/logs/foldXX/`.

Legacy stage output mappings are listed in the legacy path table above. Logs for new runs are written under `runs/<run_id>/logs/foldXX/`.

## Fine Run Commands

Collect run-scoped OOF coarse and robust lists:

```bash
python proc/fbpick/site54/oof/scripts/collect_oof_robust_lists.py \
  --cv-root /workspace/proc/fbpick/site54/oof \
  --run-id baseline_physical_center \
  --run-root /workspace/proc/fbpick/site54/oof/runs/baseline_physical_center \
  --fold-list-root /workspace/proc/fbpick/site54/oof/fold_lists \
  --out-dir /workspace/proc/fbpick/site54/oof/runs/baseline_physical_center/aggregate/05_collect_oof_lists
```

Generate run-scoped fine train and fine inference configs:

```bash
python proc/fbpick/site54/oof/scripts/make_fine_fold_configs.py \
  --cv-root /workspace/proc/fbpick/site54/oof \
  --run-id baseline_physical_center \
  --run-root /workspace/proc/fbpick/site54/oof/runs/baseline_physical_center \
  --fold-list-root /workspace/proc/fbpick/site54/oof/fold_lists \
  --config-root /workspace/proc/fbpick/site54/oof/runs/baseline_physical_center/configs \
  --oof-list-dir /workspace/proc/fbpick/site54/oof/runs/baseline_physical_center/aggregate/05_collect_oof_lists \
  --fine-valid-policy inner_valid_from_nonheldout \
  --fine-inner-valid-size 2
```

By default this reads the tracked OOF-local templates:

```text
proc/fbpick/site54/oof/config_templates/fine_train.yaml
proc/fbpick/site54/oof/config_templates/fine_infer.yaml
```

Run one fold or all folds:

```bash
RUN_ID=baseline_physical_center \
  bash proc/fbpick/site54/oof/scripts/run_fine_train_fold.sh fold00 0 full

RUN_ID=baseline_physical_center \
  bash proc/fbpick/site54/oof/scripts/run_fine_infer_fold.sh fold00 0

RUN_ID=baseline_physical_center \
  bash proc/fbpick/site54/oof/scripts/run_fine_train_all_sequential.sh 0 full

RUN_ID=baseline_physical_center \
  bash proc/fbpick/site54/oof/scripts/run_fine_infer_all_sequential.sh 0
```

Evaluate the run-scoped fine OOF outputs:

```bash
python proc/fbpick/site54/oof/scripts/evaluate_fine_oof.py \
  --fold-list-root /workspace/proc/fbpick/site54/oof/runs/baseline_physical_center/aggregate/05_collect_oof_lists/fine_fold_lists \
  --pred-root /workspace/proc/fbpick/site54/oof/runs/baseline_physical_center \
  --pred-stage-subdir 07_fine_infer \
  --out-dir /workspace/proc/fbpick/site54/oof/runs/baseline_physical_center/aggregate/08_eval \
  --run-id baseline_physical_center
```

Check the run-scoped CV outputs:

```bash
python proc/fbpick/site54/oof/scripts/check_cv_outputs.py \
  --cv-root /workspace/proc/fbpick/site54/oof \
  --run-id baseline_physical_center
```

The formal fine OOF script paths are:

```text
proc/fbpick/site54/oof/scripts/collect_oof_robust_lists.py
proc/fbpick/site54/oof/scripts/make_fine_fold_configs.py
proc/fbpick/site54/oof/scripts/evaluate_fine_oof.py
proc/fbpick/site54/oof/scripts/check_cv_outputs.py
```

`--fine-valid-policy inner_valid_from_nonheldout` is the default and keeps heldout
lists out of `06_fine_train` checkpoint selection. `--fine-valid-policy fixed_last`
uses all non-heldout samples for fine training and writes `ckpt.metric: last`,
so `ckpt/best.pt` is the final epoch checkpoint rather than the best validation
loss checkpoint. `--fine-valid-policy heldout_metric_legacy` is available only
for reproducing old behavior; it prints a warning because heldout loss is then
used to choose `ckpt/best.pt`. Generated default-policy `06_fine_train*.yaml`
files must not reference `heldout_*` list files. `07_fine_infer.yaml` may
reference heldout SGY, robust NPZ, and coarse NPZ lists, but not heldout FB;
heldout FB is reserved for final reporting in `08_eval`.

## Fold-list validation

Run this before generating or running fold configs:

```bash
python proc/fbpick/site54/oof/scripts/check_fold_lists.py
```

The default root is `/workspace/proc/fbpick/site54/oof/fold_lists`. The check confirms:

- six folds exist at `fold_lists/folds/fold00` through `fold05`
- each fold has matching SGY/FB counts for `train`, `inner_valid`, and `heldout`
- heldout SGY paths are not duplicated across folds
- total heldout survey count is 54

Use `--check-exists` only on machines that have all referenced SGY and FB files mounted.
