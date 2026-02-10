import torch

from seisai_engine.loss import composite


def _build_mse_mixed_specs():
    return composite.parse_loss_specs(
        [
            {
                'kind': 'mse',
                'weight': 1.0,
                'scope': 'masked_only',
                'params': {},
            },
            {
                'kind': 'mse',
                'weight': 0.5,
                'scope': 'all',
                'params': {},
            },
        ],
        default_scope='masked_only',
    )


def test_weighted_sum_mixed_scopes() -> None:
    pred = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    target = torch.zeros_like(pred)
    mask_bh = torch.tensor([[False, True]])
    batch = {'mask_bool': mask_bh}

    criterion = composite.build_weighted_criterion(_build_mse_mixed_specs())
    loss = criterion(pred, target, batch)

    diff = (pred - target) ** 2
    mask_bchw = mask_bh[:, None, :, None].expand_as(diff)
    masked_mse = diff[mask_bchw].mean()
    all_mse = diff.mean()
    expected = masked_mse + 0.5 * all_mse

    assert torch.allclose(loss, expected)


def test_mask_bool_shape_equivalence() -> None:
    pred = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    target = torch.zeros_like(pred)
    mask_bh = torch.tensor([[True, False]])
    mask_bhw = mask_bh[:, :, None].expand(-1, -1, pred.shape[-1])

    criterion = composite.build_weighted_criterion(_build_mse_mixed_specs())
    loss_bh = criterion(pred, target, {'mask_bool': mask_bh})
    loss_bhw = criterion(pred, target, {'mask_bool': mask_bhw})

    assert torch.allclose(loss_bh, loss_bhw)


def test_batch_not_mutated() -> None:
    pred = torch.randn(1, 1, 2, 4)
    target = torch.randn_like(pred)
    mask_bh = torch.tensor([[True, False]])
    batch = {'mask_bool': mask_bh, 'note': 'keep'}
    keys_before = set(batch.keys())

    loss_specs = composite.parse_loss_specs(
        [
            {
                'kind': 'shift_robust_mse',
                'weight': 1.0,
                'scope': 'masked_only',
                'params': {'shift_max': 1},
            }
        ],
        default_scope='masked_only',
    )
    criterion = composite.build_weighted_criterion(loss_specs)
    _ = criterion(pred, target, batch)

    assert set(batch.keys()) == keys_before
