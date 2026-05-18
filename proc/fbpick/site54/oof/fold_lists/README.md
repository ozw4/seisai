# site54 OOF 6-fold lists

This is the canonical fold-list root for the site54 54-survey 6-fold OOF CV split:

```text
/workspace/proc/fbpick/site54/oof/fold_lists
```

The parent CV runbook is:

```text
/workspace/proc/fbpick/site54/oof/README.md
```

## Fold design

The split is proximity-aware. Close or campaign-related groups are kept together where practical:

- KKS lines are held out together.
- Noto lines are held out together.
- Okinawa lines are held out together.
- Aso lines are held out together.
- Jtsanage lines are held out together.
- Kuju lines are held out together.
- Noshiro, Iwaki, South Izu, Tachikawa, and Kego pairs are kept together.

Because large related groups are kept together, heldout folds are intentionally not perfectly balanced by labeled trace count. In particular, `fold00` contains all KKS lines and is the largest heldout fold.

## Layout

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
      fold_meta.json
    fold01/
    ...
  fold_assignments.csv
  fold_summary.csv
  site54_manifest.csv
```

## Files per fold

- `heldout_sgy.txt`, `heldout_fb.txt`: OOF inference target for that fold.
- `train_sgy.txt`, `train_fb.txt`: strict training list, excluding both heldout and inner validation entries.
- `inner_valid_sgy.txt`, `inner_valid_fb.txt`: validation list for coarse training. These entries are not in `train_sgy.txt`.
- `train_all_nonheldout_sgy.txt`, `train_all_nonheldout_fb.txt`: all non-heldout entries. Use only when maximum training data is more important than keeping a separate inner validation split.
- `*_names.txt`: region names for inspection.
- `fold_meta.json`: per-fold metadata.

## Metadata

Tracked split metadata lives in this directory:

- `fold_summary.csv`: heldout region count and known labeled-trace sums by fold.
- `fold_assignments.csv`: region-to-fold assignment table.
- `site54_manifest.csv`: source list manifest and user-table coverage markers.
- `folds/foldXX/fold_meta.json`: per-fold metadata.

The user-provided table included trace/labeled-trace metadata for 44 of the 54 uploaded list entries. Entries missing from that table are marked as `missing_from_user_table` in `site54_manifest.csv`.

## Validation

Run:

```bash
python proc/fbpick/site54/oof/scripts/check_fold_lists.py
```

The checker defaults to this canonical root and confirms six folds, SGY/FB length matches for `train`, `inner_valid`, and `heldout`, no duplicate heldout SGY paths across folds, and 54 total heldout surveys.
