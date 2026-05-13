# Arakawa layout migration

このメモは、旧 `proc/arakawa` layout から current layout へ移行するための対応表です。
通常実行の手順は `proc/arakawa/README.md`、runtime speedup 実験は
`proc/arakawa/README_RUNTIME_SPEEDUP.md` を参照してください。

## Path mapping

| old path | current path |
|---|---|
| `proc/arakawa/configs/coarse_one.yaml` | `proc/arakawa/configs/templates/coarse.yaml` |
| `proc/arakawa/configs/physics_one.yaml` | `proc/arakawa/configs/templates/physics.yaml` |
| `proc/arakawa/configs/physics_qc_one_no_fb.yaml` | `proc/arakawa/configs/templates/physics_qc_no_fb.yaml` |
| `proc/arakawa/configs/fine_one.yaml` | `proc/arakawa/configs/templates/fine.yaml` |
| `proc/arakawa/configs/runtime_speedup/*.yaml` | `proc/arakawa/experiments/runtime_speedup/configs/*.yaml` |
| `proc/arakawa/generated_configs/` | `proc/arakawa/outputs/generated_configs/` |
| `proc/arakawa/runtime_runs/` | `proc/arakawa/outputs/runtime_runs/` |
| `proc/arakawa/coarse/` | `proc/arakawa/outputs/coarse/` |
| `proc/arakawa/robust/` | `proc/arakawa/outputs/robust/` |
| `proc/arakawa/fine/` | `proc/arakawa/outputs/fine/` |
| `proc/arakawa/grstat/` | `proc/arakawa/outputs/grstat/` |
| `proc/arakawa/qc/` | `proc/arakawa/outputs/qc/` |
| `proc/arakawa/qc_no_fb/` | `proc/arakawa/outputs/qc_no_fb/` |
| `proc/arakawa/eval/` | `proc/arakawa/outputs/eval/` |
| `proc/arakawa/fb_dummy/` | `proc/arakawa/outputs/fb_dummy/` |
| `proc/arakawa/runtime_compare/` | `proc/arakawa/outputs/runtime_compare/` |

## Remove already-tracked outputs from the index

過去の clone で生成物が git index に入っている場合は、index からだけ外します。
作業ファイルは削除されません。

```bash
git rm --cached -r --ignore-unmatch \
  proc/arakawa/grstat \
  proc/arakawa/generated_configs \
  proc/arakawa/runtime_runs
```

必要に応じて旧 layout の他の生成物も同じ方法で外します。

```bash
git rm --cached -r --ignore-unmatch \
  proc/arakawa/coarse \
  proc/arakawa/robust \
  proc/arakawa/fine \
  proc/arakawa/qc \
  proc/arakawa/qc_no_fb \
  proc/arakawa/eval \
  proc/arakawa/fb_dummy \
  'proc/arakawa/*arakawa_physical_export_summary.json'
```
