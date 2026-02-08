from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import segyio
import torch
from seisai_transforms import (
    PerTraceStandardize,
    ViewCompose,
    project_fb_idx_view,
    project_offsets_view,
    project_time_view,
)
from torch.utils.data import Dataset

from .builder.builder import BuildPlan, InputOnlyPlan
from .config import LoaderConfig
from .file_info import build_file_info
from .trace_subset_preproc import TraceSubsetLoader

DomainName = Literal['shot', 'recv', 'cmp']
SecondarySortKey = Literal['ffid', 'chno', 'offset']


@dataclass(frozen=True)
class InferenceGatherWindowsConfig:
    """推論用: gather を決定論で window 列挙する設定。

    メモ:
    - W 方向は crop しない(不足時のみ右 0pad)。長い W は engine 側が tiled 推論で全域予測。
    - H 方向は window 列挙し、不足は 0pad(pad_last=True のとき末尾 window も追加)。
    """

    domains: tuple[DomainName, ...] = ('shot',)
    secondary_sort: dict[DomainName, SecondarySortKey] | None = None
    win_size_traces: int = 128
    stride_traces: int = 64
    pad_last: bool = True
    target_len: int = 6016


class _NoRandRNG:
    """推論で RNG が呼ばれたら即失敗させるためのダミー RNG。"""

    def random(self, *args, **kwargs):
        msg = 'random() is not allowed in inference'
        raise RuntimeError(msg)

    def uniform(self, *args, **kwargs):
        msg = 'uniform() is not allowed in inference'
        raise RuntimeError(msg)

    def integers(self, *args, **kwargs):
        msg = 'integers() is not allowed in inference'
        raise RuntimeError(msg)


def _stable_lexsort_indices(
    info: dict,
    *,
    primary: str,
    secondary: SecondarySortKey,
    indices: np.ndarray,
) -> np.ndarray:
    if indices.size == 0:
        return indices.astype(np.int64, copy=True)

    prim_vals = info[f'{primary}_values'][indices]
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
    return indices[o2].astype(np.int64, copy=False)


def collate_pad_w_right(batch: Sequence[dict]) -> tuple[torch.Tensor, list[dict]]:
    """可変 W を右 0pad して (B,C,H,Wmax) にまとめる collate。

    Returns:
            (x_bchw, metas)

    """
    if len(batch) == 0:
        msg = 'empty batch'
        raise ValueError(msg)

    xs = [b['input'] for b in batch]
    if not all(isinstance(x, torch.Tensor) for x in xs):
        msg = "batch['input'] must be torch.Tensor"
        raise TypeError(msg)

    C, H = int(xs[0].shape[0]), int(xs[0].shape[1])
    Wmax = max(int(x.shape[2]) for x in xs)
    for x in xs:
        if x.ndim != 3:
            raise ValueError(f'input must be (C,H,W), got {tuple(x.shape)}')
        if int(x.shape[0]) != C or int(x.shape[1]) != H:
            msg = 'all inputs must share (C,H) for padding collate'
            raise ValueError(msg)

    out = torch.zeros((len(xs), C, H, Wmax), dtype=xs[0].dtype)
    for i, x in enumerate(xs):
        w = int(x.shape[2])
        out[i, :, :, :w] = x

    metas = [b['meta'] for b in batch]
    return out, metas


class InferenceGatherWindowsDataset(Dataset):
    """推論用 gather window dataset。

    返す dict 契約:
    - 'input': torch.Tensor (C,H,W)
    - 'meta': dict
            - domain: str
            - group_id: str 例 "<file_idx>:<domain>:<primary_key>"
            - file_idx: int
            - file_path: str
            - primary_key: int
            - gather_len: int   # group の総トレース数
            - abs_h: np.ndarray (H,) int64  # group 内の絶対トレース位置(pad は -1)
            - trace_valid: np.ndarray (H,) bool
            - raw_idx_global: np.ndarray (H,) int64  # raw-global row(pad は -1)
            - fb_idx_view: np.ndarray (H,) int64  # pad は -1
            - offsets_view: np.ndarray (H,) float32
            - time_view: np.ndarray (W,) float32
            - dt_sec: float32
            - dt_eff_sec: float32  # dt_sec / factor
            - n_total: int  # 全ファイル総トレース数(raw-global)

    注:
    - H 方向 window pad は 0 埋め、trace_valid=False, raw_idx_global=-1。
    - W 方向は crop せず、不足時のみ右 0pad で target_len を満たす。
    """

    def __init__(
        self,
        segy_files: Sequence[str],
        fb_files: Sequence[str],
        *,
        plan: BuildPlan | InputOnlyPlan,
        cfg: InferenceGatherWindowsConfig | None = None,
        transform=None,
        ffid_byte=segyio.TraceField.FieldRecord,
        chno_byte=segyio.TraceField.TraceNumber,
        cmp_byte=segyio.TraceField.CDP,
        use_header_cache: bool = True,
        header_cache_dir: str | None = None,
    ) -> None:
        if len(segy_files) == 0 or len(fb_files) == 0:
            msg = 'segy_files / fb_files must be non-empty'
            raise ValueError(msg)
        if len(segy_files) != len(fb_files):
            msg = 'segy_files and fb_files must have the same length'
            raise ValueError(msg)

        self.segy_files = list(segy_files)
        self.fb_files = list(fb_files)
        self.cfg = cfg or InferenceGatherWindowsConfig()
        if self.cfg.win_size_traces <= 0:
            msg = 'win_size_traces must be positive'
            raise ValueError(msg)
        if self.cfg.stride_traces <= 0:
            msg = 'stride_traces must be positive'
            raise ValueError(msg)
        if self.cfg.target_len <= 0:
            msg = 'target_len must be positive'
            raise ValueError(msg)

        sec_default: dict[DomainName, SecondarySortKey] = {
            'shot': 'chno',
            'recv': 'ffid',
            'cmp': 'offset',
        }
        sec = dict(sec_default)
        if self.cfg.secondary_sort is not None:
            sec.update(self.cfg.secondary_sort)
        self._secondary_sort = sec

        if isinstance(plan, BuildPlan):
            self.plan: InputOnlyPlan = InputOnlyPlan.from_build_plan(
                plan,
                include_label_ops=False,
            )
        elif isinstance(plan, InputOnlyPlan):
            self.plan = plan
        else:
            msg = 'plan must be BuildPlan or InputOnlyPlan'
            raise TypeError(msg)

        if transform is None:
            transform = ViewCompose([PerTraceStandardize()])
        self.transform = transform

        self.ffid_byte = ffid_byte
        self.chno_byte = chno_byte
        self.cmp_byte = cmp_byte
        self.use_header_cache = bool(use_header_cache)
        self.header_cache_dir = header_cache_dir

        self._rng = _NoRandRNG()

        self._subsetloader = TraceSubsetLoader(
            LoaderConfig(pad_traces_to=int(self.cfg.win_size_traces))
        )

        self.file_infos: list[dict] = []
        for segy_path, fb_path in zip(self.segy_files, self.fb_files, strict=True):
            info = build_file_info(
                segy_path,
                ffid_byte=self.ffid_byte,
                chno_byte=self.chno_byte,
                cmp_byte=self.cmp_byte,
                header_cache_dir=self.header_cache_dir,
                use_header_cache=self.use_header_cache,
                include_centroids=False,
            )
            fb = np.load(fb_path)
            if int(fb.shape[0]) != int(info['n_traces']):
                raise ValueError(
                    f'fb length {int(fb.shape[0])} != n_traces {int(info["n_traces"])} for {segy_path}'
                )
            info['fb'] = fb
            self.file_infos.append(info)

        self._file_base: list[int] = []
        base = 0
        for info in self.file_infos:
            self._file_base.append(base)
            base += int(info['n_traces'])
        self.n_total = int(base)

        self.groups: list[tuple[int, DomainName, int, np.ndarray, int]] = []
        self.items: list[tuple[int, int, int]] = []
        self._build_items()

    def close(self) -> None:
        for info in self.file_infos:
            segy_obj = info.get('segy_obj')
            if segy_obj is not None:
                segy_obj.close()
        self.file_infos.clear()
        self.groups.clear()
        self.items.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _build_items(self) -> None:
        win = int(self.cfg.win_size_traces)
        stride = int(self.cfg.stride_traces)
        pad_last = bool(self.cfg.pad_last)
        domains = tuple(self.cfg.domains)

        for fi, info in enumerate(self.file_infos):
            for dom in domains:
                if dom == 'shot':
                    k2i = info.get('ffid_key_to_indices')
                    primary = 'ffid'
                elif dom == 'recv':
                    k2i = info.get('chno_key_to_indices')
                    primary = 'chno'
                else:
                    k2i = info.get('cmp_key_to_indices')
                    primary = 'cmp'
                    if k2i is None:
                        continue
                if not k2i:
                    continue

                secondary = self._secondary_sort[dom]
                for pk, idxs in sorted(k2i.items()):
                    idxs_sorted = _stable_lexsort_indices(
                        info,
                        primary=primary,
                        secondary=secondary,
                        indices=np.asarray(idxs, dtype=np.int64),
                    )
                    Htot = int(idxs_sorted.size)
                    if Htot <= 0:
                        continue

                    gidx = len(self.groups)
                    self.groups.append((fi, dom, int(pk), idxs_sorted, Htot))

                    if win >= Htot:
                        starts = [0]
                    else:
                        starts = list(range(0, Htot - win + 1, stride))
                        if pad_last and (starts[-1] + win < Htot):
                            starts.append(Htot - win)

                    for s in starts:
                        e = min(int(s) + win, Htot)
                        self.items.append((gidx, int(s), int(e)))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        gidx, s, e = self.items[i]
        fi, dom, pk, idxs_sorted, Htot = self.groups[gidx]
        info = self.file_infos[fi]

        idx_win = idxs_sorted[s:e]
        H0 = int(idx_win.size)
        if H0 <= 0:
            msg = 'empty window'
            raise RuntimeError(msg)

        x = self._subsetloader.load(info['mmap'], idx_win.astype(np.int64, copy=False))
        if not isinstance(x, np.ndarray) or x.ndim != 2:
            msg = 'TraceSubsetLoader must return 2D numpy array'
            raise TypeError(msg)
        H = int(x.shape[0])
        W0 = int(x.shape[1])
        if int(self.cfg.win_size_traces) != H:
            msg = f'loaded H {H} != win_size_traces {int(self.cfg.win_size_traces)}'
            raise ValueError(msg)
        if H0 > H:
            msg = f'window size {H0} > loaded H {H}'
            raise ValueError(msg)

        trace_valid = np.zeros(H, dtype=np.bool_)
        trace_valid[:H0] = True

        off = info['offsets'][idx_win].astype(np.float32, copy=False)
        fb = info['fb'][idx_win].astype(np.int64, copy=False)
        indices_pad = np.full((H,), -1, dtype=np.int64)
        indices_pad[:H0] = idx_win.astype(np.int64, copy=False)

        if H > H0:
            pad = H - H0
            off = np.concatenate([off, np.zeros(pad, dtype=np.float32)], axis=0)
            fb = np.concatenate([fb, -np.ones(pad, dtype=np.int64)], axis=0)

        raw_idx_global = np.full((H,), -1, dtype=np.int64)
        raw_idx_global[:H0] = self._file_base[fi] + idx_win.astype(np.int64, copy=False)

        abs_h = np.full((H,), -1, dtype=np.int64)
        abs_h[:H0] = np.arange(s, s + H0, dtype=np.int64)

        out = self.transform(x, rng=self._rng, return_meta=True)
        x_view, meta = out if isinstance(out, tuple) else (out, {})
        if not isinstance(x_view, np.ndarray) or x_view.ndim != 2:
            msg = 'transform must return 2D numpy or (2D, meta)'
            raise ValueError(msg)
        Hv, W = x_view.shape
        if int(Hv) != H:
            raise ValueError(f'transform must keep H: got {int(Hv)}, expected {H}')
        if int(W) < W0:
            msg = 'transform must not crop time axis in inference'
            raise ValueError(msg)

        if int(W) < int(self.cfg.target_len):
            Wp = int(self.cfg.target_len)
            y = np.zeros((H, Wp), dtype=x_view.dtype)
            y[:, :W] = x_view
            x_view = y
            W = Wp

        if not isinstance(meta, dict):
            meta = {}
        meta.setdefault('hflip', False)
        meta.setdefault('factor_h', 1.0)
        meta.setdefault('factor', 1.0)
        meta.setdefault('start', 0)

        t_raw = np.arange(W0, dtype=np.float32) * float(info['dt_sec'])
        meta['trace_valid'] = trace_valid
        meta['raw_idx_global'] = raw_idx_global
        meta['abs_h'] = abs_h
        meta['gather_len'] = int(Htot)
        meta['domain'] = dom
        meta['primary_key'] = int(pk)
        meta['group_id'] = f'{fi}:{dom}:{int(pk)}'
        meta['file_idx'] = int(fi)
        meta['file_path'] = str(info['path'])
        meta['n_total'] = int(self.n_total)
        meta['dt_sec'] = np.float32(info['dt_sec'])
        meta['dt_eff_sec'] = np.float32(
            float(info['dt_sec']) / float(meta.get('factor', 1.0))
        )
        meta['offsets_view'] = project_offsets_view(off, H, meta)
        meta['fb_idx_view'] = project_fb_idx_view(fb, H, int(W), meta)
        meta['time_view'] = project_time_view(t_raw, H, int(W), meta)

        sample = {
            'x_view': x_view,
            'meta': meta,
        }
        self.plan.run(sample, rng=self._rng)
        if 'input' not in sample:
            msg = "plan must set sample['input']"
            raise KeyError(msg)

        x_in = sample['input']
        if not isinstance(x_in, torch.Tensor):
            msg = "sample['input'] must be torch.Tensor"
            raise TypeError(msg)
        if x_in.ndim != 3:
            raise ValueError(
                f"sample['input'] must be (C,H,W), got {tuple(x_in.shape)}"
            )
        if int(x_in.shape[1]) != H or int(x_in.shape[2]) != int(W):
            msg = 'input shape must match (H,W) of x_view'
            raise ValueError(msg)

        return {
            'input': x_in,
            'meta': meta,
        }
