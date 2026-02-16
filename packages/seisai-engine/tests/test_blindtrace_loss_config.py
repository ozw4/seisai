from seisai_engine.pipelines.blindtrace import train as blindtrace_train


def test_legacy_shift_loss_specs() -> None:
    cfg = {
        'loss_kind': 'shift_robust_mse',
        'shift_max': 4,
        'loss_scope': 'masked_only',
    }
    specs = blindtrace_train._build_loss_specs_from_cfg(
        cfg, label_prefix='train', default_scope='masked_only'
    )

    assert len(specs) == 1
    spec = specs[0]
    assert spec.kind == 'shift_robust_mse'
    assert spec.weight == 1.0
    assert spec.scope == 'masked_only'
    assert spec.params == {'shift_max': 4}


def test_legacy_shift_l1_loss_specs() -> None:
    cfg = {
        'loss_kind': 'shift_robust_l1',
        'shift_max': 3,
        'loss_scope': 'all',
    }
    specs = blindtrace_train._build_loss_specs_from_cfg(
        cfg, label_prefix='train', default_scope='masked_only'
    )

    assert len(specs) == 1
    spec = specs[0]
    assert spec.kind == 'shift_robust_l1'
    assert spec.weight == 1.0
    assert spec.scope == 'all'
    assert spec.params == {'shift_max': 3}


def test_legacy_fx_weight_adds_term() -> None:
    cfg = {
        'loss_kind': 'mse',
        'loss_scope': 'all',
        'fx_weight': 0.2,
        'fx_use_log': False,
        'fx_eps': 1.0e-5,
        'fx_f_lo': 2,
        'fx_f_hi': 10,
    }
    specs = blindtrace_train._build_loss_specs_from_cfg(
        cfg, label_prefix='train', default_scope='masked_only'
    )

    assert len(specs) == 2
    base, fx = specs
    assert base.kind == 'mse'
    assert base.weight == 1.0
    assert base.scope == 'all'

    assert fx.kind == 'fx_mag_mse'
    assert fx.weight == 0.2
    assert fx.params == {
        'use_log': False,
        'eps': 1.0e-5,
        'f_lo': 2,
        'f_hi': 10,
    }


def test_losses_list_takes_precedence_over_legacy() -> None:
    cfg = {
        'losses': [
            {
                'kind': 'l1',
                'weight': 1.0,
                'scope': 'masked_only',
                'params': {},
            }
        ],
        'loss_kind': 'mse',
        'loss_scope': 'all',
    }
    specs = blindtrace_train._build_loss_specs_from_cfg(
        cfg, label_prefix='train', default_scope='masked_only'
    )

    assert len(specs) == 1
    spec = specs[0]
    assert spec.kind == 'l1'
    assert spec.scope == 'masked_only'
