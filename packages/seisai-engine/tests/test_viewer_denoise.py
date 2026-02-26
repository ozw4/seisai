from pathlib import Path

import numpy as np
import pytest
import seisai_engine.viewer.denoise as denoise_module
import torch
from seisai_engine.viewer.denoise import _pad_chw_to_min_tile, infer_denoise_hw


def test_pad_chw_to_min_tile_expands_and_zero_fills() -> None:
    x_chw = np.arange(1 * 3 * 5, dtype=np.float32).reshape(1, 3, 5)

    padded_chw, orig_hw, target_hw = _pad_chw_to_min_tile(x_chw, tile=(4, 8))

    assert orig_hw == (3, 5)
    assert target_hw == (4, 8)
    assert padded_chw.shape == (1, 4, 8)
    assert np.array_equal(padded_chw[:, :3, :5], x_chw)
    assert np.all(padded_chw[:, 3:, :] == 0.0)
    assert np.all(padded_chw[:, :, 5:] == 0.0)


def test_infer_denoise_hw_restores_original_shape_with_pad_and_crop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    section_hw = np.arange(3 * 5, dtype=np.float32).reshape(3, 5)
    captured: dict[str, tuple[int, ...]] = {}

    bundle = denoise_module._ViewerModelBundleDenoise(
        model=torch.nn.Identity(),
        in_chans=1,
        out_chans=1,
        used_ema=False,
    )

    monkeypatch.setattr(
        denoise_module,
        '_get_model_bundle',
        lambda **_: bundle,
    )

    def _dummy_infer_tiled(
        model: torch.nn.Module,
        x_chw: np.ndarray,
        *,
        tile: tuple[int, int],
        overlap: tuple[int, int],
        amp: bool,
        tiles_per_batch: int,
    ) -> np.ndarray:
        del model, tile, overlap, amp, tiles_per_batch
        captured['x_shape'] = tuple(x_chw.shape)
        return np.zeros((1, x_chw.shape[1], x_chw.shape[2]), dtype=np.float32)

    monkeypatch.setattr(denoise_module, '_infer_tiled', _dummy_infer_tiled)

    out_hw = infer_denoise_hw(
        section_hw,
        ckpt_path=tmp_path / 'dummy.pt',
        device='cpu',
        tile=(4, 8),
        overlap=(1, 2),
    )

    assert captured['x_shape'] == (1, 4, 8)
    assert out_hw.shape == (3, 5)
    assert out_hw.dtype == np.float32
    assert out_hw.flags.c_contiguous


@pytest.mark.parametrize(
    ('in_chans', 'out_chans', 'match_text'),
    [
        (2, 1, 'in_chans=1'),
        (1, 2, 'out_chans=1'),
    ],
)
def test_build_model_bundle_rejects_non_single_channel_model_sig(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    in_chans: int,
    out_chans: int,
    match_text: str,
) -> None:
    ckpt = {
        'version': 1,
        'pipeline': 'pair',
        'epoch': 1,
        'global_step': 1,
        'model_state_dict': {},
        'model_sig': {
            'backbone': 'dummy_backbone',
            'in_chans': in_chans,
            'out_chans': out_chans,
        },
    }
    called = {'build': False}

    def _never_called(model_kwargs: dict) -> torch.nn.Module:
        del model_kwargs
        called['build'] = True
        return torch.nn.Identity()

    monkeypatch.setattr(denoise_module, 'load_checkpoint', lambda _: ckpt)
    monkeypatch.setattr(denoise_module, 'build_encdec2d_model', _never_called)

    with pytest.raises(ValueError, match=match_text):
        denoise_module._build_model_bundle(
            ckpt_path=tmp_path / 'dummy.pt',
            device=torch.device('cpu'),
            use_ema=None,
        )
    assert called['build'] is False


def test_select_state_dict_none_mode_honors_truthy_infer_used_ema() -> None:
    ema_state = {'w': torch.tensor([1.0])}
    model_state = {'w': torch.tensor([2.0])}
    ckpt = {
        'infer_used_ema': np.bool_(True),
        'ema_state_dict': ema_state,
        'model_state_dict': model_state,
    }

    selected, used_ema = denoise_module._select_checkpoint_state_dict(
        ckpt,
        use_ema=None,
    )

    assert used_ema is True
    assert selected is ema_state
