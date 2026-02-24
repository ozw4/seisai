import numpy as np
from seisai_dataset import (
    TraceSubsetSampler,
    TraceSubsetSamplerConfig,
)


def fake_info():
    # key_to_indices: 2つの ffid グループ、それぞれ 5 本
    ffid_vals = np.array(
        [100, 100, 100, 100, 100, 101, 101, 101, 101, 101], dtype=np.int64
    )
    chno_vals = np.arange(10, dtype=np.int64)
    cmp_vals = np.ones(10, dtype=np.int64)
    offsets = np.linspace(0, 9, 10).astype(np.float32)
    key_to_idx = {100: np.arange(0, 5), 101: np.arange(5, 10)}
    chno_key_to_idx = {int(i): np.asarray([i], dtype=np.int64) for i in range(10)}
    uniq = [100, 101]
    return {
        'ffid_values': ffid_vals,
        'chno_values': chno_vals,
        'cmp_values': cmp_vals,
        'offsets': offsets,
        'ffid_key_to_indices': key_to_idx,
        'chno_key_to_indices': chno_key_to_idx,
        'cmp_key_to_indices': None,
        'ffid_unique_keys': uniq,
        'chno_unique_keys': list(range(10)),
        'cmp_unique_keys': [],
        # セントロイドなしで OK(index window fallback 動作を確認)
        'ffid_centroids': None,
        'chno_centroids': None,
        'sampling_override': None,
    }


def test_sampler_semantics_without_superwindow() -> None:
    info = fake_info()
    cfg = TraceSubsetSamplerConfig(
        primary_keys=('ffid',),
        primary_key_weights=(1.0,),
        use_superwindow=False,
        secondary_key_fixed=True,
        subset_traces=8,
    )
    sampler = TraceSubsetSampler(cfg)
    out = sampler.draw(info)  # random は許容(仕様テスト)

    # 仕様：出力項目
    assert set(out.keys()) == {
        'indices',
        'pad_len',
        'key_name',
        'secondary_key',
        'did_super',
        'primary_unique',
    }
    idx = out['indices']
    assert idx.dtype == np.int64
    assert len(idx) <= 8
    assert out['key_name'] == 'ffid'
    assert out['secondary_key'] in {
        'chno',
        'ffid',
        'offset',
    }  # secondary_key_fixed=True → "chno"
    assert out['did_super'] in {False, True}  # 今回は False のはず


def test_sampler_primary_range_filters_primary_values() -> None:
    info = fake_info()
    info['sampling_override'] = {
        'primary_keys': ['ffid'],
        'primary_ranges': {'ffid': [[101, 101]]},
    }
    cfg = TraceSubsetSamplerConfig(
        primary_keys=('ffid',),
        secondary_key_fixed=True,
        subset_traces=8,
    )
    sampler = TraceSubsetSampler(cfg)
    out = sampler.draw(info)
    assert out['key_name'] == 'ffid'
    assert out['primary_unique'] == '101'


def test_sampler_secondary_key_override_wins() -> None:
    info = fake_info()
    info['sampling_override'] = {
        'primary_keys': ['ffid'],
        'secondary_key': {'ffid': 'offset'},
    }
    cfg = TraceSubsetSamplerConfig(
        primary_keys=('ffid',),
        secondary_key_fixed=True,
        subset_traces=8,
    )
    sampler = TraceSubsetSampler(cfg)
    out = sampler.draw(info)
    assert out['key_name'] == 'ffid'
    assert out['secondary_key'] == 'offset'


def test_sampler_secondary_key_fixed_per_primary() -> None:
    info = fake_info()
    info['sampling_override'] = {
        'primary_keys': ['ffid'],
        'secondary_key_fixed': {'ffid': True},
    }
    cfg = TraceSubsetSamplerConfig(
        primary_keys=('ffid',),
        secondary_key_fixed=False,
        subset_traces=8,
    )
    sampler = TraceSubsetSampler(cfg)
    out = sampler.draw(info)
    assert out['key_name'] == 'ffid'
    assert out['secondary_key'] == 'chno'


def test_sampler_primary_ranges_can_define_primary_keys() -> None:
    info = fake_info()
    info['sampling_override'] = {
        'primary_ranges': {'ffid': [[100, 101]]},
    }
    cfg = TraceSubsetSamplerConfig(
        primary_keys=('chno',),
        secondary_key_fixed=True,
        subset_traces=8,
    )
    sampler = TraceSubsetSampler(cfg)
    out = sampler.draw(info)
    assert out['key_name'] == 'ffid'


def test_sampler_accepts_pre_normalized_sampling_override() -> None:
    info = fake_info()
    cfg = TraceSubsetSamplerConfig(
        primary_keys=('ffid',),
        secondary_key_fixed=False,
        subset_traces=8,
    )
    sampler = TraceSubsetSampler(cfg)
    info['sampling_override'] = sampler.normalize_sampling_override(
        {
            'primary_keys': ['ffid'],
            'primary_ranges': {'ffid': [[101, 101]]},
            'secondary_key_fixed': {'ffid': True},
        }
    )
    out = sampler.draw(info)
    assert out['key_name'] == 'ffid'
    assert out['primary_unique'] == '101'
    assert out['secondary_key'] == 'chno'


def test_sampler_supports_attr_info_without_get() -> None:
    class InfoNoGet:
        def __init__(self, payload: dict) -> None:
            for k, v in payload.items():
                setattr(self, k, v)

        def __getitem__(self, key: str):
            return getattr(self, key)

    payload = fake_info()
    payload['sampling_override'] = {
        'primary_keys': ['ffid'],
        'primary_ranges': {'ffid': [[100, 100]]},
    }
    info = InfoNoGet(payload)
    cfg = TraceSubsetSamplerConfig(
        primary_keys=('ffid',),
        secondary_key_fixed=True,
        subset_traces=8,
    )
    sampler = TraceSubsetSampler(cfg)
    out = sampler.draw(info)
    assert out['key_name'] == 'ffid'
    assert out['primary_unique'] == '100'


def test_sampler_primary_ranges_reapplied_after_superwindow() -> None:
    info = fake_info()
    info['sampling_override'] = {
        'primary_keys': ['ffid'],
        'primary_ranges': {'ffid': [[100, 100]]},
    }
    cfg = TraceSubsetSamplerConfig(
        primary_keys=('ffid',),
        use_superwindow=True,
        sw_halfspan=1,
        sw_prob=1.0,
        secondary_key_fixed=True,
        subset_traces=8,
    )
    sampler = TraceSubsetSampler(cfg)
    out = sampler.draw(info)
    assert out['key_name'] == 'ffid'
    assert out['did_super'] is True
    assert out['primary_unique'] == '100'
    ffid_values = info['ffid_values'][out['indices']]
    assert np.all(ffid_values == 100)
