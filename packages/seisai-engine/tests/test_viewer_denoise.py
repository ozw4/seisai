from pathlib import Path

import numpy as np
import pytest
import seisai_engine.viewer.denoise as denoise_module
import torch
from seisai_engine.predict import infer_tiled_bchw
from seisai_engine.viewer.denoise import _pad_chw_to_min_tile, infer_denoise_hw


class _IdentityOutModel(torch.nn.Module):
    out_chans = 1

    def __init__(self) -> None:
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gain


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


def test_infer_denoise_hw_mask_ratio_zero_keeps_direct_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    section_hw = np.arange(5 * 7, dtype=np.float32).reshape(5, 7)
    bundle = denoise_module._ViewerModelBundleDenoise(
        model=_IdentityOutModel(),
        in_chans=1,
        out_chans=1,
        used_ema=False,
    )
    monkeypatch.setattr(denoise_module, '_get_model_bundle', lambda **_: bundle)

    called = {'direct': 0}

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
        called['direct'] += 1
        return np.full((1, x_chw.shape[1], x_chw.shape[2]), 7.0, dtype=np.float32)

    monkeypatch.setattr(denoise_module, '_infer_tiled', _dummy_infer_tiled)

    def _never_cover(*args: object, **kwargs: object) -> torch.Tensor:
        del args, kwargs
        raise AssertionError('cover should not be called when mask_ratio == 0')

    monkeypatch.setattr(denoise_module, 'cover_all_traces_predict_striped', _never_cover)

    out_hw = infer_denoise_hw(
        section_hw,
        ckpt_path=tmp_path / 'dummy.pt',
        device='cpu',
        tile=(4, 6),
        overlap=(2, 3),
        mask_ratio=0.0,
        noise_std=9.9,
        mask_noise_mode='add',
        seed=987,
        passes_batch=2,
    )

    assert called['direct'] == 1
    assert out_hw.shape == section_hw.shape
    assert np.allclose(out_hw, 7.0)


def test_infer_denoise_hw_cover_branch_reflects_mask_params(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    section_hw = np.linspace(1.0, 2.0, 8 * 12, dtype=np.float32).reshape(8, 12)
    bundle = denoise_module._ViewerModelBundleDenoise(
        model=_IdentityOutModel(),
        in_chans=1,
        out_chans=1,
        used_ema=False,
    )
    monkeypatch.setattr(denoise_module, '_get_model_bundle', lambda **_: bundle)

    orig_cover = denoise_module.cover_all_traces_predict_striped
    called = {'cover': 0}

    def _cover_wrapper(*args: object, **kwargs: object) -> torch.Tensor:
        called['cover'] += 1
        return orig_cover(*args, **kwargs)

    monkeypatch.setattr(denoise_module, 'cover_all_traces_predict_striped', _cover_wrapper)

    out_noise0 = infer_denoise_hw(
        section_hw,
        ckpt_path=tmp_path / 'dummy.pt',
        device='cpu',
        tile=(4, 6),
        overlap=(2, 3),
        amp=False,
        mask_ratio=0.5,
        noise_std=0.0,
        mask_noise_mode='replace',
        seed=12345,
        passes_batch=2,
    )
    out_noise1 = infer_denoise_hw(
        section_hw,
        ckpt_path=tmp_path / 'dummy.pt',
        device='cpu',
        tile=(4, 6),
        overlap=(2, 3),
        amp=False,
        mask_ratio=0.5,
        noise_std=0.5,
        mask_noise_mode='replace',
        seed=12345,
        passes_batch=2,
    )
    out_replace = infer_denoise_hw(
        section_hw,
        ckpt_path=tmp_path / 'dummy.pt',
        device='cpu',
        tile=(4, 6),
        overlap=(2, 3),
        amp=False,
        mask_ratio=0.5,
        noise_std=0.5,
        mask_noise_mode='replace',
        seed=77,
        passes_batch=2,
    )
    out_seed_alt = infer_denoise_hw(
        section_hw,
        ckpt_path=tmp_path / 'dummy.pt',
        device='cpu',
        tile=(4, 6),
        overlap=(2, 3),
        amp=False,
        mask_ratio=0.5,
        noise_std=0.5,
        mask_noise_mode='replace',
        seed=78,
        passes_batch=2,
    )
    out_add = infer_denoise_hw(
        section_hw,
        ckpt_path=tmp_path / 'dummy.pt',
        device='cpu',
        tile=(4, 6),
        overlap=(2, 3),
        amp=False,
        mask_ratio=0.5,
        noise_std=0.5,
        mask_noise_mode='add',
        seed=77,
        passes_batch=2,
    )

    assert called['cover'] > 0
    assert out_noise0.shape == section_hw.shape
    assert out_noise0.dtype == np.float32
    assert not np.allclose(out_noise0, out_noise1)
    assert not np.allclose(out_replace, out_seed_alt)
    assert not np.allclose(out_replace, out_add)


def test_infer_tiled_bchw_preserves_shape() -> None:
    model = _IdentityOutModel()
    x_bchw = torch.arange(2 * 1 * 5 * 7, dtype=torch.float32).reshape(2, 1, 5, 7)

    y_bchw = infer_tiled_bchw(
        model,
        x_bchw,
        tile=(4, 6),
        overlap=(2, 3),
        amp=False,
        use_tqdm=False,
        tiles_per_batch=2,
    )

    assert isinstance(y_bchw, torch.Tensor)
    assert tuple(y_bchw.shape) == (2, 1, 5, 7)
    assert y_bchw.dtype == torch.float32


def test_infer_tiled_bchw_fails_fast_on_device_mismatch() -> None:
    model = _IdentityOutModel().to(device=torch.device('meta'))
    x_bchw = torch.zeros((1, 1, 4, 6), dtype=torch.float32)

    with pytest.raises(ValueError, match=r'x_bchw\.device .* != model\.device .*'):
        infer_tiled_bchw(
            model,
            x_bchw,
            tile=(4, 6),
            overlap=(0, 0),
            amp=False,
            use_tqdm=False,
            tiles_per_batch=1,
        )
