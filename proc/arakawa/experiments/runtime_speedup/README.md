# Arakawa runtime speedup experiment

The canonical runtime speedup configs live in:

```text
proc/arakawa/experiments/runtime_speedup/configs/
```

Run from the repository root:

```bash
python -m cli.run_arakawa_fbpick_physical_export \
  --config proc/arakawa/experiments/runtime_speedup/configs/A0_full.yaml
```

Run `A0_full` first. The other configs reuse its coarse output and write under:

```text
proc/arakawa/outputs/runtime_runs/<STAGE>/
```

The legacy `proc/arakawa/configs/runtime_speedup/` configs remain during the
layout transition, but new docs and edits should use the canonical configs here.
