from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import segyio

from seisai_engine.pipelines.common import build_encdec2d_model, save_checkpoint


def write_unstructured_segy(path: str | Path, traces_hw: np.ndarray, dt_us: int) -> None:
    """Write a tiny unstructured SEG-Y.

    This mirrors the synthetic SEG-Y writer used in seisai-dataset unit tests,
    but is duplicated here to keep e2e tests self-contained.
    """
    out_path = Path(path)
    arr = np.asarray(traces_hw, dtype=np.float32)
    if arr.ndim != 2:
        msg = 'traces_hw must be 2D (n_traces, n_samples)'
        raise ValueError(msg)
    if arr.shape[0] <= 0 or arr.shape[1] <= 0:
        msg = 'traces_hw must be non-empty'
        raise ValueError(msg)

    n_traces, n_samples = arr.shape
    out_path.parent.mkdir(parents=True, exist_ok=True)

    spec = segyio.spec()
    # Mandatory fields for segyio.create
    spec.iline = 189
    spec.xline = 193
    spec.format = 5  # IEEE float32
    spec.sorting = 2
    spec.samples = np.arange(n_samples, dtype=np.int32)
    spec.tracecount = int(n_traces)

    with segyio.create(str(out_path), spec) as f:
        f.bin[segyio.BinField.Interval] = int(dt_us)
        for i in range(n_traces):
            sx = 100
            sy = 2000
            gx = 1000 + i * 10
            gy = 2000
            f.header[i] = {
                segyio.TraceField.FieldRecord: 1,
                segyio.TraceField.TraceNumber: int(i + 1),
                segyio.TraceField.CDP: 1,
                segyio.TraceField.offset: int((i + 1) * 10),
                segyio.TraceField.SourceX: int(sx),
                segyio.TraceField.SourceY: int(sy),
                segyio.TraceField.GroupX: int(gx),
                segyio.TraceField.GroupY: int(gy),
                segyio.TraceField.SourceGroupScalar: 1,
            }
            f.trace[i] = arr[i]


def make_synthetic_segy(tmp_path: Path, *, n_traces: int = 8, n_samples: int = 64) -> Path:
    t = np.arange(n_samples, dtype=np.float32)
    traces = np.stack([t + (10.0 * i) for i in range(n_traces)], axis=0)
    segy_path = tmp_path / 'synthetic.sgy'
    write_unstructured_segy(segy_path, traces, dt_us=1000)
    return segy_path


def make_dummy_ckpt(
    tmp_path: Path,
    *,
    pipeline: str,
    in_chans: int,
    out_chans: int,
    backbone: str = 'resnet18',
) -> Path:
    """Create a minimal checkpoint compatible with the segy2segy infer CLIs."""
    model_sig: dict[str, Any] = {
        'backbone': str(backbone),
        'pretrained': False,
        'in_chans': int(in_chans),
        'out_chans': int(out_chans),
    }
    model = build_encdec2d_model(dict(model_sig))
    ckpt = {
        'version': 1,
        'pipeline': str(pipeline),
        'epoch': 0,
        'global_step': 0,
        'model_state_dict': model.state_dict(),
        'model_sig': dict(model_sig),
        # Infer CLIs require this key to exist (used for merge + hashing).
        'cfg': {},
    }
    ckpt_path = tmp_path / f'{pipeline}_dummy_best.pt'
    save_checkpoint(ckpt_path, ckpt)
    return ckpt_path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def segy_text_header_contains(path: Path, needle: str) -> bool:
    with segyio.open(str(path), 'r', ignore_geometry=True) as f:
        # segyio returns bytes for text headers; decode robustly.
        raw = f.text[0]
        txt = raw.decode('ascii', errors='ignore') if isinstance(raw, (bytes, bytearray)) else str(raw)
    return needle in txt
