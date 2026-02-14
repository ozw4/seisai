from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from seisai_engine.pipelines.common import maybe_load_init_weights
from seisai_engine.pipelines.common.checkpoint_io import save_checkpoint


class DummyModel(nn.Module):
    def __init__(self, in_chans: int, out_chans: int) -> None:
        super().__init__()
        self.stem = nn.Conv2d(in_chans, 4, kernel_size=1, bias=False)
        self.seg_head = nn.Conv2d(4, out_chans, kernel_size=1, bias=False)
        self.last_load_state_result = None
        self.last_loaded_state_keys: list[str] = []

    def load_state_dict(self, state_dict, strict: bool = True):
        self.last_loaded_state_keys = list(state_dict.keys())
        result = super().load_state_dict(state_dict, strict=strict)
        self.last_load_state_result = result
        return result


def _make_model(*, in_chans: int, out_chans: int, stem_fill: float, seg_fill: float) -> DummyModel:
    model = DummyModel(in_chans=in_chans, out_chans=out_chans)
    with torch.no_grad():
        model.stem.weight.fill_(float(stem_fill))
        model.seg_head.weight.fill_(float(seg_fill))
    return model


def _save_ckpt(
    *,
    ckpt_path: Path,
    in_chans: int,
    out_chans: int,
    model: DummyModel,
) -> None:
    payload = {
        'version': 1,
        'pipeline': 'psn',
        'epoch': 0,
        'global_step': 1,
        'model_sig': {'in_chans': int(in_chans), 'out_chans': int(out_chans)},
        'model_state_dict': model.state_dict(),
    }
    save_checkpoint(ckpt_path, payload)


def test_maybe_load_init_weights_in_chans_mismatch_raises(tmp_path: Path) -> None:
    ckpt_model = _make_model(in_chans=1, out_chans=1, stem_fill=1.0, seg_fill=2.0)
    ckpt_path = tmp_path / 'best.pt'
    _save_ckpt(ckpt_path=ckpt_path, in_chans=1, out_chans=1, model=ckpt_model)

    model = _make_model(in_chans=2, out_chans=1, stem_fill=-1.0, seg_fill=-2.0)

    with pytest.raises(RuntimeError) as exc_info:
        maybe_load_init_weights(
            cfg={'train': {'init_ckpt': str(ckpt_path)}},
            base_dir=tmp_path,
            model=model,
            model_sig={'in_chans': 2, 'out_chans': 1},
        )

    message = str(exc_info.value)
    assert 'ckpt_in=1' in message
    assert 'cur_in=2' in message
    assert str(ckpt_path.resolve()) in message


def test_maybe_load_init_weights_out_chans_mismatch_loads_without_seg_head(
    tmp_path: Path,
) -> None:
    ckpt_model = _make_model(in_chans=1, out_chans=1, stem_fill=3.0, seg_fill=7.0)
    ckpt_dir = tmp_path / 'ckpt'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / 'best.pt'
    _save_ckpt(ckpt_path=ckpt_path, in_chans=1, out_chans=1, model=ckpt_model)

    model = _make_model(in_chans=1, out_chans=3, stem_fill=-3.0, seg_fill=-7.0)
    seg_head_before = model.seg_head.weight.detach().clone()

    maybe_load_init_weights(
        cfg={'train': {'init_ckpt': 'ckpt/best.pt'}},
        base_dir=tmp_path,
        model=model,
        model_sig={'in_chans': 1, 'out_chans': 3},
    )

    assert torch.equal(model.stem.weight, ckpt_model.stem.weight)
    assert not torch.equal(model.seg_head.weight, ckpt_model.seg_head.weight)
    assert torch.equal(model.seg_head.weight, seg_head_before)
    assert model.last_load_state_result is not None
    assert model.last_load_state_result.unexpected_keys == []
    assert model.last_load_state_result.missing_keys == ['seg_head.weight']
    assert model.last_loaded_state_keys == ['stem.weight']


def test_maybe_load_init_weights_in_out_match_loads_all_keys(tmp_path: Path) -> None:
    ckpt_model = _make_model(in_chans=1, out_chans=3, stem_fill=11.0, seg_fill=13.0)
    ckpt_path = tmp_path / 'best.pt'
    _save_ckpt(ckpt_path=ckpt_path, in_chans=1, out_chans=3, model=ckpt_model)

    model = _make_model(in_chans=1, out_chans=3, stem_fill=-11.0, seg_fill=-13.0)

    maybe_load_init_weights(
        cfg={'train': {'init_ckpt': str(ckpt_path)}},
        base_dir=tmp_path,
        model=model,
        model_sig={'in_chans': 1, 'out_chans': 3},
    )

    assert torch.equal(model.stem.weight, ckpt_model.stem.weight)
    assert torch.equal(model.seg_head.weight, ckpt_model.seg_head.weight)
    assert model.last_load_state_result is not None
    assert model.last_load_state_result.unexpected_keys == []
    assert model.last_load_state_result.missing_keys == []
    assert model.last_loaded_state_keys == ['stem.weight', 'seg_head.weight']
