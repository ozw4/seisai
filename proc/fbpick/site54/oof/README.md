# site54 OOF CV runbook

`/workspace/proc/fbpick/site54/oof` is the canonical root for the site54 54-survey 6-fold OOF CV experiment. Keep new fold lists, configs, logs, stage outputs, manifests, and aggregate reports under this root so coarse, physics, fine, and eval artifacts can be traced as one CV run.

## Canonical paths

- CV root: `/workspace/proc/fbpick/site54/oof`
- Fold list root: `/workspace/proc/fbpick/site54/oof/fold_lists`
- Config root: `/workspace/proc/fbpick/site54/oof/configs`
- New run root: `/workspace/proc/fbpick/site54/oof/runs/<run_id>`

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

The following paths are legacy duplicates retained only for existing configs and historical outputs:

- `/workspace/proc/fbpick/site54/oof/folds`
- `/workspace/proc/fbpick/site54/oof/site54_oof_6fold_lists`
- `/workspace/proc/fbpick/site54/oof/lists/all_sgy.txt`
- `/workspace/proc/fbpick/site54/oof/lists/all_fb.txt`

Do not use these paths for new configs or new scripts. New fold-list references must use `/workspace/proc/fbpick/site54/oof/fold_lists`.

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
| `01_coarse_train` | Train one coarse model per fold. | `fold_lists/folds/foldXX/train_sgy.txt`, `train_fb.txt`, `inner_valid_sgy.txt`, `inner_valid_fb.txt`; `configs/config_train_fbpick_coarse_foldXX*.yaml`. | New: `runs/<run_id>/foldXX/01_coarse_train/`; legacy configs currently write `coarse_foldXX_train_out/` or `coarse_foldXX_train_smoke_out/`. | `proc/fbpick/site54/oof/scripts/run_coarse_train_fold.sh fold00 0 full` | Smoke uses `config_train_fbpick_coarse_foldXX_smoke.yaml` and short training. Full uses `config_train_fbpick_coarse_foldXX.yaml`. | Heldout lists must not be used for training, validation, model selection, or early stopping. |
| `02_coarse_infer` | Produce coarse predictions for each fold's heldout surveys. | `fold_lists/folds/foldXX/heldout_sgy.txt`; checkpoint from `01_coarse_train`. | New: `runs/<run_id>/foldXX/02_coarse_infer/`; legacy configs currently write `coarse_oof/foldXX/`. | `proc/fbpick/site54/oof/scripts/run_coarse_infer_fold.sh fold00 0` | Smoke should run on the smoke checkpoint and a temporary run directory. Full runs every heldout SGY for the fold. | Heldout SGY may be used only as inference input. Heldout FB is still not used. |
| `03_physics` | Convert heldout coarse predictions to physical-center robust NPZ files. | `fold_lists/folds/foldXX/heldout_sgy.txt`; heldout coarse NPZ from `02_coarse_infer`; `configs/config_run_fbpick_physics_foldXX_heldout.yaml`. | New: `runs/<run_id>/foldXX/03_physics/`; legacy configs currently write `robust_oof/foldXX/`. | `proc/fbpick/site54/oof/scripts/run_physics_fold.sh fold00` | Smoke should target a temporary run directory and a small heldout subset. Full runs all heldout predictions. | Heldout FB is not used. Heldout SGY and coarse NPZ are inference inputs only. |
| `04_physics_qc` | Check robust outputs against heldout SGY/FB lists and summarize missing or mismatched outputs. | `fold_lists/folds/foldXX/heldout_sgy.txt`, `heldout_fb.txt`; robust NPZ from `03_physics`; `configs/config_run_fbpick_physics_qc_foldXX.yaml`. | New: `runs/<run_id>/foldXX/04_physics_qc/`; existing checks are report/log oriented. | `python cli/run_fbpick_physics_qc.py --config proc/fbpick/site54/oof/configs/config_run_fbpick_physics_qc_fold00.yaml` | Smoke checks one fold or a temporary subset. Full checks all six folds. | Heldout FB may be used for QC metrics after inference products already exist. Do not feed QC results back into training. |
| `05_collect_oof_lists` | Collect coarse and robust OOF NPZ paths in canonical all-SGY order for fine training. | `fold_lists/lists/all_sgy.txt`, `all_fb.txt`; `coarse_oof/foldXX/*.coarse.npz`; `robust_oof/foldXX/*.robust.npz`. | New: `runs/<run_id>/aggregate/05_collect_oof_lists/`; legacy output is `lists/oof_train_*`. | `python proc/fbpick/site54/oof/scripts/collect_oof_robust_lists.py --out-dir proc/fbpick/site54/oof/runs/<run_id>/aggregate/05_collect_oof_lists` | Smoke should collect from a smoke run directory. Full requires all 54 coarse and robust outputs. | Uses all heldout inference outputs as OOF features. No model fitting happens in this stage. |
| `06_fine_train` | Train one fine model per fold using non-heldout surveys and their OOF robust/coarse features. | `fine_fold_lists/foldXX/train_sgy.txt`, `train_fb.txt`, `train_robust.txt`; config `configs/config_train_fbpick_fine_oof_foldXX.yaml`; optional coarse init. | New: `runs/<run_id>/foldXX/06_fine_train/`; legacy configs currently write `/workspace/proc/fbpick/site54/fbpick_fine_train_oof_foldXX_out`. | `python cli/run_fbpick_fine_train.py --config proc/fbpick/site54/oof/configs/config_train_fbpick_fine_oof_fold00.yaml` | Smoke should use the smoke base config or a generated small-list config and a temporary run directory. Full uses the generated fold config for each fold. | The fold's heldout SGY/FB/robust lists may be used only for inference/evaluation fields in the config, not for train batches or model selection. |
| `07_fine_infer` | Run each fold's fine model on that fold's heldout surveys. | `fine_fold_lists/foldXX/heldout_sgy.txt`, `heldout_fb.txt`, `heldout_robust.txt`; checkpoint from `06_fine_train`. | New: `runs/<run_id>/foldXX/07_fine_infer/`; current evaluation default expects `fine_infer/foldXX/`. | `python cli/run_fbpick_fine_infer.py --config <config with foldXX heldout inputs and ckpt_path>` | Smoke uses one fold or a small heldout subset. Full writes predictions for all heldout surveys in all folds. | Heldout SGY and robust NPZ are inference inputs. Heldout FB may be carried for reporting but must not tune the model. |
| `08_eval` | Aggregate OOF prediction quality across folds. | `fine_fold_lists/foldXX/heldout_sgy.txt`, `heldout_fb.txt`; fine predictions from `07_fine_infer`; optional coarse/robust stages. | New: `runs/<run_id>/aggregate/08_eval/`; current default is `fine_eval/`. | `python proc/fbpick/site54/scripts/evaluate_fbpick_fine_oof.py --pred-root proc/fbpick/site54/oof/fine_infer --out-dir proc/fbpick/site54/oof/runs/<run_id>/aggregate/08_eval` | Smoke evaluates one fold or subset. Full evaluates all six folds and all requested stages. | Heldout FB is allowed for final reporting only. Evaluation choices must not be fed back into the same CV run's training. |

## Run layout

New CV outputs should use this layout:

```text
runs/<run_id>/
  manifest.yaml
  logs/
    fold00/
      01_coarse_train.log
      02_coarse_infer.log
      03_physics.log
      04_physics_qc.log
      06_fine_train.log
      07_fine_infer.log
  fold00/
    01_coarse_train/
    02_coarse_infer/
    03_physics/
    04_physics_qc/
    06_fine_train/
    07_fine_infer/
  fold01/
  ...
  aggregate/
    05_collect_oof_lists/
    08_eval/
```

Minimum `manifest.yaml`:

```yaml
run_id: baseline_physical_center
cv_root: /workspace/proc/fbpick/site54/oof
fold_list_root: /workspace/proc/fbpick/site54/oof/fold_lists
config_root: /workspace/proc/fbpick/site54/oof/configs
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
