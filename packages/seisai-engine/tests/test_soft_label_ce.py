import torch
from seisai_engine.loss.soft_label_ce import (
    build_pixel_mask_from_batch,
    soft_label_ce_map,
    soft_label_ce_masked_mean,
)


def _make_soft_target(logits: torch.Tensor, *, class_dim: int = 1) -> torch.Tensor:
    p = torch.softmax(logits.detach(), dim=int(class_dim))
    # ensure it's a proper probability distribution along class_dim
    return p.to(dtype=logits.dtype, device=logits.device)


def test_soft_label_ce_map_shape_and_formula() -> None:
    B, C, H, W = 2, 3, 4, 5
    logits = torch.randn((B, C, H, W), dtype=torch.float32)
    target = _make_soft_target(logits)

    loss_map = soft_label_ce_map(logits, target, class_dim=1)
    assert isinstance(loss_map, torch.Tensor)
    assert tuple(loss_map.shape) == (B, H, W)

    log_p = torch.log_softmax(logits, dim=1)
    ref = -(target * log_p).sum(dim=1)
    assert torch.allclose(loss_map, ref, atol=0.0, rtol=0.0)


def test_build_pixel_mask_from_batch_bhw_no_mask_bool() -> None:
    B, C, H, W = 2, 3, 4, 5
    target = torch.zeros((B, C, H, W), dtype=torch.float32)
    trace_valid = torch.tensor(
        [
            [True, True, False, True],
            [True, False, False, True],
        ],
        dtype=torch.bool,
    )
    label_valid = torch.tensor(
        [
            [True, False, True, True],
            [True, True, False, False],
        ],
        dtype=torch.bool,
    )
    batch = {
        'target': target,
        'trace_valid': trace_valid,
        'label_valid': label_valid,
    }

    mask = build_pixel_mask_from_batch(batch)
    assert mask.dtype is torch.bool
    assert tuple(mask.shape) == (B, H, W)

    ref_bh = trace_valid & label_valid
    ref = ref_bh[:, :, None].expand(B, H, W)
    assert torch.equal(mask, ref)


def test_build_pixel_mask_from_batch_with_mask_bool_variants() -> None:
    B, C, H, W = 2, 3, 4, 5
    target = torch.zeros((B, C, H, W), dtype=torch.float32)
    trace_valid = torch.ones((B, H), dtype=torch.bool)
    label_valid = torch.ones((B, H), dtype=torch.bool)

    # (B,H,W)
    mb_bhw = torch.zeros((B, H, W), dtype=torch.bool)
    mb_bhw[:, 1, 2] = True
    mask = build_pixel_mask_from_batch(
        {
            'target': target,
            'trace_valid': trace_valid,
            'label_valid': label_valid,
            'mask_bool': mb_bhw,
        }
    )
    assert torch.equal(mask, mb_bhw)

    # (B,H) -> expand
    mb_bh = torch.zeros((B, H), dtype=torch.bool)
    mb_bh[:, 2] = True
    mask2 = build_pixel_mask_from_batch(
        {
            'target': target,
            'trace_valid': trace_valid,
            'label_valid': label_valid,
            'mask_bool': mb_bh,
        }
    )
    ref2 = mb_bh[:, :, None].expand(B, H, W)
    assert torch.equal(mask2, ref2)

    # (B,W) -> expand
    mb_bw = torch.zeros((B, W), dtype=torch.bool)
    mb_bw[:, 3] = True
    mask3 = build_pixel_mask_from_batch(
        {
            'target': target,
            'trace_valid': trace_valid,
            'label_valid': label_valid,
            'mask_bool': mb_bw,
        }
    )
    ref3 = mb_bw[:, None, :].expand(B, H, W)
    assert torch.equal(mask3, ref3)


def test_soft_label_ce_masked_mean_empty_is_zero_and_backward_ok() -> None:
    B, C, H, W = 2, 3, 4, 5
    logits = torch.randn((B, C, H, W), dtype=torch.float32, requires_grad=True)
    target = _make_soft_target(logits)
    mask = torch.zeros((B, H, W), dtype=torch.bool)

    loss = soft_label_ce_masked_mean(logits, target, mask)
    assert isinstance(loss, torch.Tensor)
    assert tuple(loss.shape) == ()
    assert torch.isfinite(loss)
    assert float(loss.detach().item()) == 0.0

    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum().item()) == 0.0


def test_soft_label_ce_masked_mean_ignores_nonfinite_in_masked_out_pixels() -> None:
    B, C, H, W = 2, 3, 4, 5
    logits = torch.randn((B, C, H, W), dtype=torch.float32, requires_grad=True)
    # Inject NaNs only in masked-out pixels.
    logits.data[:, :, 0, 0] = float('nan')
    target = torch.full((B, C, H, W), 1.0 / float(C), dtype=torch.float32)

    mask = torch.zeros((B, H, W), dtype=torch.bool)
    mask[:, 1, 1] = True  # select one finite pixel per sample

    loss = soft_label_ce_masked_mean(logits, target, mask)
    assert torch.isfinite(loss), f'loss must be finite, got {loss}'

    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all(), 'grad must be finite'
    assert float(logits.grad[:, :, 0, 0].abs().sum().item()) == 0.0
