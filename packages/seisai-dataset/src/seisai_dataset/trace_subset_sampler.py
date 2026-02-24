import random
from typing import Any

import numpy as np

from .config import TraceSubsetSamplerConfig
from .file_info import FileInfo


class TraceSubsetSampler:
    """ギャザー(=keyで定義されるトレース集合)から、連続サブセット(最大H=subset_traces)の trace indices を抽出するコンポーネン.

    入力: file_infos の 1 要素(dict) を想定(MaskedSegyGather が作る構造)
    必須キー: '{ffid|chno|cmp}_unique_keys', '{ffid|chno|cmp}_key_to_indices',
                            '{ffid|chno|cmp}_values', 'offsets',
                            'ffid_centroids','chno_centroids' (あれば距離KNNに使用)
    出力: dict
    - indices: np.ndarray[int64]    連続サブセットのインデックス
    - pad_len: int                  subset_traces に満たない場合の不足本数
    - key_name: str                 選んだ primary key 名('ffid'|'chno'|'cmp')
    - secondary_key: str            二次整列に用いたキー('ffid'|'chno'|'offset')
    - did_super: bool               このサンプルでsuperwindowを実際に適用したか
    - primary_unique: str           サブセット内 primary 値のユニークを昇順に','連結
    """

    def __init__(self, cfg: TraceSubsetSamplerConfig) -> None:
        self.cfg = cfg
        self._valid_primary = {'ffid', 'chno', 'cmp'}
        self._valid_secondary = {'ffid', 'chno', 'offset'}
        self._raw_sampling_override_keys = {
            'primary_keys',
            'primary_ranges',
            'secondary_key_fixed',
            'secondary_key',
        }
        self._normalized_sampling_override_keys = {
            'primary_keys',
            'primary_ranges',
            'secondary_key_fixed_global',
            'secondary_key_fixed_by_primary',
            'secondary_key_by_primary',
        }
        self._allowed_secondary_by_primary = {
            'ffid': {'chno', 'offset'},
            'chno': {'ffid', 'offset'},
            'cmp': {'offset'},
        }

    def draw(
        self, info: dict | FileInfo, *, py_random: random.Random | None = None
    ) -> dict:
        r = py_random or random
        sampling_override = self._resolve_sampling_override(info)

        cmp_available = (
            bool(self._info_get(info, 'cmp_unique_keys'))
            and isinstance(info['cmp_unique_keys'], (list, tuple))
            and len(info['cmp_unique_keys']) > 0
        )

        primary_keys_override = self._resolve_primary_keys_override(sampling_override)
        primary_ranges = self._resolve_primary_ranges(sampling_override)

        # 1) primary key 候補の構築(重み付き / fallback 互換)
        key_candidates, weight_candidates = self._build_primary_candidates(
            cmp_available=cmp_available,
            primary_keys_override=primary_keys_override,
        )
        key_candidates, weight_candidates = self._filter_primary_candidates_by_ranges(
            info,
            key_candidates,
            weight_candidates,
            primary_ranges=primary_ranges,
        )
        if not key_candidates:
            msg = 'No primary key candidates available after applying primary_ranges'
            raise RuntimeError(msg)

        # 選択
        if any(w > 0 for w in weight_candidates) and len(weight_candidates) == len(
            key_candidates
        ):
            key_name = r.choices(key_candidates, weights=weight_candidates, k=1)[0]
        else:
            key_name = r.choice(key_candidates)

        unique_keys = info[f'{key_name}_unique_keys']
        key_to_indices = info[f'{key_name}_key_to_indices']
        if not unique_keys:
            msg = f'No unique keys for {key_name}'
            raise RuntimeError(msg)

        ranges = primary_ranges.get(key_name)
        if ranges:
            unique_keys = [
                int(k) for k in unique_keys if self._value_in_ranges(int(k), ranges)
            ]
            if not unique_keys:
                msg = (
                    f'No unique keys for {key_name} after applying primary_ranges: '
                    f'{ranges}'
                )
                raise RuntimeError(msg)

        key = int(r.choice(unique_keys))
        indices = key_to_indices[key]

        # 2) superwindow(距離KNN/インデックスwindow fallback)
        apply_super, did_super = False, False
        if self.cfg.use_superwindow and self.cfg.sw_halfspan > 0:
            apply_super = True
            if float(self.cfg.sw_prob) < 1.0 and r.random() >= float(self.cfg.sw_prob):
                apply_super = False

            if apply_super:
                did_super = True
                indices = self._apply_superwindow(info, key_name, key, indices)

        indices = np.asarray(indices, dtype=np.int64)
        if ranges:
            indices = self._filter_indices_by_primary_ranges(
                info,
                key_name=key_name,
                indices=indices,
                ranges=ranges,
            )
            if indices.size == 0:
                msg = (
                    f'No trace indices for {key_name} after applying primary_ranges: '
                    f'{ranges}'
                )
                raise RuntimeError(msg)

        # 3) secondary 整列(従来条件の踏襲 + override)
        secondary_by_primary = self._resolve_secondary_by_primary(sampling_override)
        secondary_key = secondary_by_primary.get(key_name)
        if secondary_key is None:
            secondary_key = self._choose_secondary_key(
                key_name,
                apply_super=apply_super,
                secondary_key_fixed=self._resolve_secondary_key_fixed(
                    sampling_override,
                    key_name=key_name,
                    default=bool(self.cfg.secondary_key_fixed),
                ),
                r=r,
            )
        indices = self._stable_lexsort(info, key_name, str(secondary_key), indices)

        # 4) 連続サブセット抽出(最大 subset_traces)
        n_total = len(indices)
        H = int(self.cfg.subset_traces)
        if n_total >= H:
            start_idx = r.randint(0, n_total - H)
            subset_indices = indices[start_idx : start_idx + H]
            pad_len = 0
        else:
            subset_indices = indices
            pad_len = H - n_total

        subset_indices = np.asarray(subset_indices, dtype=np.int64)

        # 5) ログ用 primary ユニーク
        valid_subset = subset_indices >= 0
        if np.any(valid_subset):
            prim_vals_sel = info[f'{key_name}_values'][subset_indices[valid_subset]]
            primary_label_values = np.unique(prim_vals_sel.astype(np.int64, copy=False))
            primary_unique_str = ','.join(map(str, primary_label_values.tolist()))
        else:
            primary_unique_str = ''

        return {
            'indices': subset_indices,
            'pad_len': pad_len,
            'key_name': key_name,
            'secondary_key': secondary_key,
            'did_super': did_super,
            'primary_unique': primary_unique_str,
        }

    @staticmethod
    def _info_get(info: dict | FileInfo, key: str, default=None):
        if isinstance(info, dict):
            return info.get(key, default)
        return getattr(info, key, default)

    def _normalize_prebuilt_sampling_override(
        self, override: dict[str, object]
    ) -> dict[str, object]:
        unknown = set(override) - self._normalized_sampling_override_keys
        if unknown:
            msg = f'sampling_override has unsupported keys: {sorted(unknown)}'
            raise ValueError(msg)
        primary_keys = override.get('primary_keys')
        if primary_keys is not None and not isinstance(primary_keys, tuple):
            msg = (
                'normalized sampling_override.primary_keys must be tuple[str] or None'
            )
            raise TypeError(msg)
        primary_ranges = override.get('primary_ranges')
        if primary_ranges is None:
            primary_ranges = {}
        if not isinstance(primary_ranges, dict):
            msg = 'normalized sampling_override.primary_ranges must be dict'
            raise TypeError(msg)
        fixed_global = override.get('secondary_key_fixed_global')
        if fixed_global is not None and not isinstance(fixed_global, bool):
            msg = (
                'normalized sampling_override.secondary_key_fixed_global '
                'must be bool or None'
            )
            raise TypeError(msg)
        fixed_by_primary = override.get('secondary_key_fixed_by_primary')
        if fixed_by_primary is None:
            fixed_by_primary = {}
        if not isinstance(fixed_by_primary, dict):
            msg = (
                'normalized sampling_override.secondary_key_fixed_by_primary '
                'must be dict'
            )
            raise TypeError(msg)
        secondary_by_primary = override.get('secondary_key_by_primary')
        if secondary_by_primary is None:
            secondary_by_primary = {}
        if not isinstance(secondary_by_primary, dict):
            msg = 'normalized sampling_override.secondary_key_by_primary must be dict'
            raise TypeError(msg)
        return {
            'primary_keys': primary_keys,
            'primary_ranges': primary_ranges,
            'secondary_key_fixed_global': fixed_global,
            'secondary_key_fixed_by_primary': fixed_by_primary,
            'secondary_key_by_primary': secondary_by_primary,
        }

    def _resolve_sampling_override(
        self, info: dict | FileInfo
    ) -> dict[str, object] | None:
        override = self._info_get(info, 'sampling_override')
        if override is None:
            return None
        if not isinstance(override, dict):
            msg = 'sampling_override must be dict when provided'
            raise TypeError(msg)
        has_normalized_only_key = any(
            key in override
            for key in (
                'secondary_key_fixed_global',
                'secondary_key_fixed_by_primary',
                'secondary_key_by_primary',
            )
        )
        if has_normalized_only_key:
            return self._normalize_prebuilt_sampling_override(override)
        return self.normalize_sampling_override(override)

    @staticmethod
    def _coerce_int(value: object, *, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            msg = f'{label} must be int'
            raise TypeError(msg)
        return int(value)

    def _normalize_primary_keys_override(
        self, value: object
    ) -> tuple[str, ...] | None:
        if value is None:
            return None
        if not isinstance(value, (list, tuple)):
            msg = 'sampling_override.primary_keys must be list[str] or tuple[str]'
            raise TypeError(msg)
        if len(value) == 0:
            msg = 'sampling_override.primary_keys must not be empty'
            raise ValueError(msg)
        out: list[str] = []
        seen: set[str] = set()
        for idx, item in enumerate(value):
            if not isinstance(item, str):
                msg = f'sampling_override.primary_keys[{idx}] must be str'
                raise TypeError(msg)
            key = item.strip()
            if key not in self._valid_primary:
                msg = (
                    'sampling_override.primary_keys has invalid value: '
                    f'{item!r}; allowed={sorted(self._valid_primary)}'
                )
                raise ValueError(msg)
            if key in seen:
                msg = f'sampling_override.primary_keys has duplicate: {key}'
                raise ValueError(msg)
            seen.add(key)
            out.append(key)
        return tuple(out)

    def _normalize_range_pairs(
        self, value: object, *, key_name: str
    ) -> tuple[tuple[int, int], ...]:
        if not isinstance(value, (list, tuple)):
            msg = (
                f'sampling_override.primary_ranges[{key_name!r}] must be '
                '[lo, hi] or [[lo, hi], ...]'
            )
            raise TypeError(msg)
        raw_pairs: list[object]
        if (
            len(value) == 2
            and not any(isinstance(v, (list, tuple)) for v in value)
        ):
            raw_pairs = [value]
        else:
            raw_pairs = list(value)

        out: list[tuple[int, int]] = []
        for i, pair in enumerate(raw_pairs):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                msg = (
                    'sampling_override.primary_ranges values must contain [lo, hi] '
                    f'pairs; got index={i} for key={key_name!r}'
                )
                raise TypeError(msg)
            lo = self._coerce_int(pair[0], label=f'primary_ranges[{key_name}][{i}][0]')
            hi = self._coerce_int(pair[1], label=f'primary_ranges[{key_name}][{i}][1]')
            if lo > hi:
                msg = (
                    'sampling_override.primary_ranges requires lo <= hi; '
                    f'got [{lo}, {hi}] for key={key_name!r}'
                )
                raise ValueError(msg)
            out.append((int(lo), int(hi)))
        if not out:
            msg = f'sampling_override.primary_ranges[{key_name!r}] must not be empty'
            raise ValueError(msg)
        return tuple(out)

    def _normalize_primary_ranges_override(
        self, value: object
    ) -> dict[str, tuple[tuple[int, int], ...]]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            msg = 'sampling_override.primary_ranges must be dict'
            raise TypeError(msg)
        out: dict[str, tuple[tuple[int, int], ...]] = {}
        for raw_key, raw_ranges in value.items():
            if not isinstance(raw_key, str):
                msg = 'sampling_override.primary_ranges keys must be str'
                raise TypeError(msg)
            key = raw_key.strip()
            if key not in self._valid_primary:
                msg = (
                    'sampling_override.primary_ranges has invalid key: '
                    f'{raw_key!r}; allowed={sorted(self._valid_primary)}'
                )
                raise ValueError(msg)
            out[key] = self._normalize_range_pairs(raw_ranges, key_name=key)
        return out

    def _normalize_secondary_key_fixed_override(
        self, value: object
    ) -> tuple[bool | None, dict[str, bool]]:
        if value is None:
            return None, {}
        if isinstance(value, bool):
            return bool(value), {}
        if not isinstance(value, dict):
            msg = (
                'sampling_override.secondary_key_fixed must be bool or '
                'dict[str,bool]'
            )
            raise TypeError(msg)
        by_primary: dict[str, bool] = {}
        for raw_key, raw_value in value.items():
            if not isinstance(raw_key, str):
                msg = 'sampling_override.secondary_key_fixed keys must be str'
                raise TypeError(msg)
            key = raw_key.strip()
            if key not in self._valid_primary:
                msg = (
                    'sampling_override.secondary_key_fixed has invalid key: '
                    f'{raw_key!r}; allowed={sorted(self._valid_primary)}'
                )
                raise ValueError(msg)
            if not isinstance(raw_value, bool):
                msg = f'sampling_override.secondary_key_fixed[{key!r}] must be bool'
                raise TypeError(msg)
            by_primary[key] = bool(raw_value)
        return None, by_primary

    def _normalize_secondary_key_override(
        self, value: object
    ) -> dict[str, str]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            msg = 'sampling_override.secondary_key must be dict[str,str]'
            raise TypeError(msg)
        out: dict[str, str] = {}
        for raw_key, raw_value in value.items():
            if not isinstance(raw_key, str):
                msg = 'sampling_override.secondary_key keys must be str'
                raise TypeError(msg)
            if not isinstance(raw_value, str):
                msg = f'sampling_override.secondary_key[{raw_key!r}] must be str'
                raise TypeError(msg)
            primary = raw_key.strip()
            secondary = raw_value.strip()
            if primary not in self._valid_primary:
                msg = (
                    'sampling_override.secondary_key has invalid primary key: '
                    f'{raw_key!r}; allowed={sorted(self._valid_primary)}'
                )
                raise ValueError(msg)
            if secondary not in self._valid_secondary:
                msg = (
                    'sampling_override.secondary_key has invalid secondary value: '
                    f'{raw_value!r}; allowed={sorted(self._valid_secondary)}'
                )
                raise ValueError(msg)
            if secondary not in self._allowed_secondary_by_primary[primary]:
                msg = (
                    'sampling_override.secondary_key has invalid pair: '
                    f'{primary!r} -> {secondary!r}'
                )
                raise ValueError(msg)
            out[primary] = secondary
        return out

    def normalize_sampling_override(
        self, override: object
    ) -> dict[str, object] | None:
        if override is None:
            return None
        if not isinstance(override, dict):
            msg = 'sampling_override must be dict when provided'
            raise TypeError(msg)
        unknown = set(override) - self._raw_sampling_override_keys
        if unknown:
            msg = f'sampling_override has unsupported keys: {sorted(unknown)}'
            raise ValueError(msg)
        primary_keys = self._normalize_primary_keys_override(override.get('primary_keys'))
        primary_ranges = self._normalize_primary_ranges_override(
            override.get('primary_ranges')
        )
        fixed_global, fixed_by_primary = self._normalize_secondary_key_fixed_override(
            override.get('secondary_key_fixed')
        )
        secondary_by_primary = self._normalize_secondary_key_override(
            override.get('secondary_key')
        )
        return {
            'primary_keys': primary_keys,
            'primary_ranges': primary_ranges,
            'secondary_key_fixed_global': fixed_global,
            'secondary_key_fixed_by_primary': fixed_by_primary,
            'secondary_key_by_primary': secondary_by_primary,
        }

    @staticmethod
    def _value_in_ranges(value: int, ranges: tuple[tuple[int, int], ...]) -> bool:
        v = int(value)
        for lo, hi in ranges:
            if int(lo) <= v <= int(hi):
                return True
        return False

    def _resolve_primary_keys_override(
        self, sampling_override: dict[str, object] | None
    ) -> tuple[str, ...] | None:
        if sampling_override is None:
            return None
        primary_keys = sampling_override.get('primary_keys')
        if isinstance(primary_keys, tuple):
            return primary_keys
        primary_ranges = sampling_override.get('primary_ranges')
        if isinstance(primary_ranges, dict) and primary_ranges:
            return tuple(primary_ranges.keys())
        return None

    def _resolve_primary_ranges(
        self, sampling_override: dict[str, object] | None
    ) -> dict[str, tuple[tuple[int, int], ...]]:
        if sampling_override is None:
            return {}
        primary_ranges = sampling_override.get('primary_ranges')
        if isinstance(primary_ranges, dict):
            return primary_ranges
        return {}

    def _resolve_secondary_by_primary(
        self, sampling_override: dict[str, object] | None
    ) -> dict[str, str]:
        if sampling_override is None:
            return {}
        secondary_by_primary = sampling_override.get('secondary_key_by_primary')
        if isinstance(secondary_by_primary, dict):
            return secondary_by_primary
        return {}

    def _resolve_secondary_key_fixed(
        self,
        sampling_override: dict[str, object] | None,
        *,
        key_name: str,
        default: bool,
    ) -> bool:
        if sampling_override is None:
            return bool(default)
        fixed_by_primary = sampling_override.get('secondary_key_fixed_by_primary')
        if isinstance(fixed_by_primary, dict) and key_name in fixed_by_primary:
            return bool(fixed_by_primary[key_name])
        fixed_global = sampling_override.get('secondary_key_fixed_global')
        if isinstance(fixed_global, bool):
            return bool(fixed_global)
        return bool(default)

    # ---- helpers ----
    def _build_primary_candidates(
        self,
        *,
        cmp_available: bool,
        primary_keys_override: tuple[str, ...] | None = None,
    ):
        source_keys = primary_keys_override
        if source_keys is None:
            source_keys = self.cfg.primary_keys

        weight_by_key: dict[str, float] = {}
        if self.cfg.primary_keys:
            for i, key in enumerate(self.cfg.primary_keys):
                if self.cfg.primary_key_weights and i < len(self.cfg.primary_key_weights):
                    weight_by_key[key] = max(float(self.cfg.primary_key_weights[i]), 0.0)
                else:
                    weight_by_key[key] = 1.0

        if source_keys:
            kc, wc = [], []
            for k in source_keys:
                if k not in self._valid_primary:
                    continue
                if k == 'cmp' and not cmp_available:
                    continue
                kc.append(k)
                wc.append(float(weight_by_key.get(k, 1.0)))
            if not kc:  # fallback
                kc = ['ffid', 'chno'] + (['cmp'] if cmp_available else [])
                wc = [1.0] * len(kc)
        else:
            kc = ['ffid', 'chno'] + (['cmp'] if cmp_available else [])
            wc = [1.0] * len(kc)
        return kc, wc

    def _filter_primary_candidates_by_ranges(
        self,
        info: dict | FileInfo,
        key_candidates: list[str],
        weight_candidates: list[float],
        *,
        primary_ranges: dict[str, tuple[tuple[int, int], ...]],
    ) -> tuple[list[str], list[float]]:
        if not primary_ranges:
            return key_candidates, weight_candidates
        out_keys: list[str] = []
        out_weights: list[float] = []
        for key_name, weight in zip(key_candidates, weight_candidates, strict=True):
            ranges = primary_ranges.get(key_name)
            if not ranges:
                out_keys.append(key_name)
                out_weights.append(float(weight))
                continue
            unique_keys = self._info_get(info, f'{key_name}_unique_keys')
            if not isinstance(unique_keys, (list, tuple)):
                continue
            if any(self._value_in_ranges(int(k), ranges) for k in unique_keys):
                out_keys.append(key_name)
                out_weights.append(float(weight))
        return out_keys, out_weights

    def _filter_indices_by_primary_ranges(
        self,
        info: dict | FileInfo,
        *,
        key_name: str,
        indices: np.ndarray,
        ranges: tuple[tuple[int, int], ...],
    ) -> np.ndarray:
        if indices.size == 0:
            return indices
        values = np.asarray(info[f'{key_name}_values'][indices], dtype=np.int64)
        keep_mask = np.array(
            [self._value_in_ranges(int(v), ranges) for v in values],
            dtype=np.bool_,
        )
        return np.asarray(indices[keep_mask], dtype=np.int64)

    def _apply_superwindow(
        self, info: dict | FileInfo, key_name: str, key: int, indices: np.ndarray
    ) -> np.ndarray:
        K = 1 + 2 * int(self.cfg.sw_halfspan)
        sel_keys, k2map = self._select_keys_by_centroid_or_index_window(
            key_name, key, info, K
        )

        chunks = []
        for k2 in sel_keys:
            idxs = k2map.get(int(k2))
            if idxs is not None and len(idxs) > 0:
                chunks.append(idxs)
        if chunks:
            return np.concatenate(chunks).astype(np.int64)
        return np.asarray(indices, dtype=np.int64)

    def _select_keys_by_centroid_or_index_window(
        self, key_name: str, key: int, info: dict | FileInfo, K: int
    ) -> tuple[np.ndarray, dict]:
        def _index_window() -> np.ndarray:
            uniq = self._info_get(info, f'{key_name}_unique_keys', None)
            uniq_arr = (
                np.asarray(uniq, dtype=np.int64)
                if isinstance(uniq, (list, tuple))
                else np.asarray([], dtype=np.int64)
            )
            if uniq_arr.size > 0:
                uniq_sorted = np.sort(uniq_arr)
                center = int(key)
                pos = np.searchsorted(uniq_sorted, center)
                lo = max(0, pos - self.cfg.sw_halfspan)
                hi = min(len(uniq_sorted), pos + self.cfg.sw_halfspan + 1)
                return uniq_sorted[lo:hi]
            return np.asarray([int(key)], dtype=np.int64)

        cent = self._info_get(info, f'{key_name}_centroids')
        if isinstance(cent, dict) and int(key) in cent:
            keys = np.fromiter(cent.keys(), dtype=np.int64)
            coords = np.array([cent[int(k)] for k in keys], dtype=np.float64)
            cx, cy = cent[int(key)]
            d = np.hypot(coords[:, 0] - cx, coords[:, 1] - cy)
            order = np.argsort(d)
            sel_keys = keys[order][:K]
        else:
            sel_keys = _index_window()

        k2map = info[f'{key_name}_key_to_indices']
        return np.asarray(sel_keys, dtype=np.int64), k2map

    def _choose_secondary_key(
        self,
        key_name: str,
        *,
        apply_super: bool,
        secondary_key_fixed: bool,
        r: random.Random,
    ) -> str:
        if not apply_super and not secondary_key_fixed:
            if key_name == 'ffid':
                return r.choice(('chno', 'offset'))
            if key_name == 'chno':
                return r.choice(('ffid', 'offset'))
            # 'cmp'
            return 'offset'
        if apply_super and not secondary_key_fixed:
            return 'offset'
        if key_name == 'ffid':
            return 'chno'
        if key_name == 'chno':
            return 'ffid'
        return 'offset'

    def _stable_lexsort(
        self, info: dict | FileInfo, key_name: str, secondary: str, indices: np.ndarray
    ) -> np.ndarray:
        prim_vals = info[f'{key_name}_values'][indices]
        if secondary == 'chno':
            sec_vals = info['chno_values'][indices]
        elif secondary == 'ffid':
            sec_vals = info['ffid_values'][indices]
        else:
            sec_vals = info['offsets'][indices]

        o = np.argsort(prim_vals, kind='mergesort')
        indices = indices[o]
        sec_vals = sec_vals[o]
        o2 = np.argsort(sec_vals, kind='mergesort')
        return indices[o2]
