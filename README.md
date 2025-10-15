# seisai

Modular, PyTorch‑friendly **SEG‑Y gather dataset toolkit**.

## Install

```bash
# Pin by commit for reproducibility
pip install "seisai @ git+https://github.com/<you>/seisai.git@<commit-sha>"
# or local dev
pip install -e ./seisai
```

## Quickstart

```python
from torch.utils.data import DataLoader
from seisai import SegyGatherPipelineDataset

ds = SegyGatherPipelineDataset(
    segy_files=["/path/input.sgy"],
    fb_files=["/path/fb.npy"],
    target_mode="fb_seg",    # or "recon"
    use_header_cache=True,
)
loader = DataLoader(ds, batch_size=2, num_workers=2)
batch = next(iter(loader))
```

## Core pieces

* **SegyGatherPipelineDataset** – end‑to‑end: sample → load/normalize → augment → FBLC gate → mask → target.
* **TraceSubsetLoader** – mmap load + per‑trace standardize; fixed time length.
* **TraceSubsetSampler** – choose gathers by `ffid/chno/cmp`; optional *superwindow*.
* **Time/Space/Freq Augmenters** – independent; **runtime‑overridable**.
* **TraceMasker** – replace/add gaussian noise; returns indices.
* **FirstBreakGate** – FB lateral consistency (percentile of adjacent FB diffs).
* **FBTargetBuilder** – Gaussian FB heatmap `(1,H,W)`.

## Testing

```bash
pytest -q
# Enable integration tests with real data
export FBP_TEST_SEGY=/abs/path/sample.sgy
export FBP_TEST_FB=/abs/path/fb.npy
```

## License

MIT
