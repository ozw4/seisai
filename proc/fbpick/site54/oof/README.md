# site54 OOF 6-fold list files

Generated from the uploaded `sgy_list_site54.txt` and `fb_list_site54.txt`.
Both input lists contain 54 entries and are paired line-by-line.

## Fold design

This is a proximity-aware 6-fold split. The assignment keeps close/campaign-related groups together where practical:

- KKS lines are held out together.
- Noto lines are held out together.
- Okinawa lines are held out together.
- Aso lines are held out together.
- Jtsanage lines are held out together.
- Kuju lines are held out together.
- Noshiro, Iwaki, South Izu, Tachikawa, Kego pairs are kept together.

Because large related groups are kept together, the heldout folds are intentionally not perfectly balanced by labeled trace count. In particular, `fold00` contains all KKS lines and is the largest heldout fold.

## Files per fold

Each `folds/foldXX/` directory contains:

- `heldout_sgy.txt`, `heldout_fb.txt`: OOF inference target for that fold.
- `train_sgy.txt`, `train_fb.txt`: strict training list, excluding both heldout and inner validation entries.
- `inner_valid_sgy.txt`, `inner_valid_fb.txt`: validation list for coarse training. These entries are not in `train_sgy.txt`.
- `train_all_nonheldout_sgy.txt`, `train_all_nonheldout_fb.txt`: all non-heldout entries. Use this instead of `train_sgy.txt` only if you prefer maximum training data and can tolerate train/validation overlap if you also use `inner_valid_*`.
- `*_names.txt`: region names for inspection.
- `fold_meta.json`: fold metadata.

Recommended OOF coarse training config for fold XX:

```yaml
paths:
  segy_files: /workspace/proc/fbpick/oof/folds/foldXX/train_sgy.txt
  fb_files: /workspace/proc/fbpick/oof/folds/foldXX/train_fb.txt
  infer_segy_files: /workspace/proc/fbpick/oof/folds/foldXX/inner_valid_sgy.txt
  infer_fb_files: /workspace/proc/fbpick/oof/folds/foldXX/inner_valid_fb.txt
  out_dir: /workspace/proc/fbpick/oof/coarse_foldXX_train_out
```

Recommended OOF coarse inference config for fold XX:

```yaml
paths:
  segy_files: /workspace/proc/fbpick/oof/folds/foldXX/heldout_sgy.txt
  out_dir: /workspace/proc/fbpick/oof/coarse_oof/foldXX
infer:
  ckpt_path: /workspace/proc/fbpick/oof/coarse_foldXX_train_out/ckpt/best.pt
```

## Metadata note

The user-provided table included trace/labeled-trace metadata for 44 of the 54 uploaded list entries. The following entries are present in the uploaded list files but were missing from the table and are marked as `missing_from_user_table` in `site54_manifest.csv`:

- DAIDAITOKU04_Kouga
- DAIDAITOKU04_Oosaka
- DAIDAITOKU04_refraction
- DAIDAITOKU_Suzuka
- noto_togi18
- okinawa2011_OH11-0A
- okinawa2022_OH11-02
- okinawa2022_OH11-03
- okinawa2022_OH11-04
- okinawa2022_OH11-05

## Fold heldout summary

See `fold_summary.csv` for heldout region count and known labeled-trace sums.
