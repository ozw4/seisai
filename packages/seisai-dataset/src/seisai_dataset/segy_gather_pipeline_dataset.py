# %%
import contextlib

import numpy as np
import segyio
import torch
from seisai_builders.builder import (
	BuildPlan,
)
from torch.utils.data import Dataset

from .config import LoaderConfig, TraceSubsetSamplerConfig
from .file_info import build_file_info
from .gate_fblc import FirstBreakGate
from .trace_subset_preproc import TraceSubsetLoader
from .trace_subset_sampler import TraceSubsetSampler


class SegyGatherPipelineDataset(Dataset):
	"""SEG-Y ギャザー読み込み → サンプリング → 変換 → FBLC ゲート → （任意）BuildPlanで入出力生成。

	期待する transform:  x(H,W) -> x_view  もしくは  (x_view, meta)
	- meta は少なくとも { 'hflip':bool, 'factor':float, 'start':int, 'did_space':bool, 'factor_h':float } の任意サブセット
	期待する fbgate: FirstBreakGate（min_pick_accept と fblc_accept を持つ）
	期待する plan:  BuildPlan（任意）。与えれば sample に 'input' / 'target' などを組み立てる
	"""

	def __init__(
		self,
		segy_files: list[str],
		fb_files: list[str],
		transform,
		fbgate: FirstBreakGate,
		*,
		plan: BuildPlan | None = None,
		ffid_byte=segyio.TraceField.FieldRecord,
		chno_byte=segyio.TraceField.TraceNumber,
		cmp_byte=segyio.TraceField.CDP,
		primary_keys: tuple[str, ...] | None = None,
		primary_key_weights: tuple[float, ...] | None = None,
		use_superwindow: bool = False,
		sw_halfspan: int = 0,
		sw_prob: float = 0.3,
		use_header_cache: bool = True,
		header_cache_dir: str | None = None,
		subset_traces: int = 128,
		valid: bool = False,
		verbose: bool = False,
	) -> None:
		if len(segy_files) == 0 or len(fb_files) == 0:
			raise ValueError('segy_files / fb_files は空であってはならない')
		if len(segy_files) != len(fb_files):
			raise ValueError('segy_files と fb_files の長さが一致していません')

		self.segy_files = list(segy_files)
		self.fb_files = list(fb_files)
		self.transform = transform
		self.fbgate = fbgate
		self.plan = plan

		self.ffid_byte = ffid_byte
		self.chno_byte = chno_byte
		self.cmp_byte = cmp_byte

		self.primary_keys = tuple(primary_keys) if primary_keys else None
		self.primary_key_weights = (
			tuple(primary_key_weights) if primary_key_weights else None
		)

		self.use_superwindow = bool(use_superwindow)
		self.sw_halfspan = int(sw_halfspan)
		self.sw_prob = float(sw_prob)

		self.use_header_cache = bool(use_header_cache)
		self.header_cache_dir = header_cache_dir

		self.valid = bool(valid)
		self.verbose = bool(verbose)

		self._rng = np.random.default_rng()

		# components
		self.subsetloader = TraceSubsetLoader(
			LoaderConfig(pad_traces_to=int(subset_traces))
		)
		self.sampler = TraceSubsetSampler(
			TraceSubsetSamplerConfig(
				primary_keys=self.primary_keys,
				primary_key_weights=self.primary_key_weights,
				use_superwindow=self.use_superwindow,
				sw_halfspan=self.sw_halfspan,
				sw_prob=self.sw_prob,
				valid=self.valid,
				subset_traces=int(subset_traces),
			)
		)

		# ファイルごとのインデックス辞書等を構築
		self.file_infos: list[dict] = []
		for segy_path, fb_path in zip(self.segy_files, self.fb_files, strict=True):
			info = build_file_info(
				segy_path,
				ffid_byte=self.ffid_byte,
				chno_byte=self.chno_byte,
				cmp_byte=self.cmp_byte,
				header_cache_dir=self.header_cache_dir,
				use_header_cache=self.use_header_cache,
				include_centroids=True,  # or False
			)
			fb = np.load(fb_path)
			info['fb'] = fb
			self.file_infos.append(info)

	def close(self) -> None:
		for info in self.file_infos:
			segy_obj = info.get('segy_obj')
			if segy_obj is not None:
				with contextlib.suppress(Exception):
					segy_obj.close()
		self.file_infos.clear()

	def __del__(self) -> None:
		self.close()

	def __len__(self) -> int:
		return 10**6

	def __getitem__(self, _=None) -> dict:
		while True:
			fidx = int(self._rng.integers(0, len(self.file_infos)))
			info = self.file_infos[fidx]
			mmap = info['mmap']

			s = self.sampler.draw(
				info,
				py_random=np.random.RandomState(int(self._rng.integers(0, 2**31 - 1))),
			)
			indices = np.asarray(s['indices'], dtype=np.int64)
			key_name = s['key_name']
			secondary_key = s['secondary_key']
			did_super = bool(s['did_super'])
			primary_unique_str = s['primary_unique']

			fb_all = info['fb']
			fb_subset = fb_all[indices]

			# ピック数の最低限チェック（読み出し前に棄却したい時はこの位置）
			ok_pick, _, _ = self.fbgate.min_pick_accept(fb_subset)
			if not ok_pick:
				continue

			# 波形読み出し
			x = self.subsetloader.load(mmap, indices)  # (H,W0) float32 を想定
			offsets = info['offsets'][indices].astype(np.float32)

			# 変換（Crop/Pad / TimeStretch 等）
			out = self.transform(x, rng=self._rng, return_meta=True)
			x_view, meta = out if isinstance(out, tuple) else (out, {})
			if not isinstance(x_view, np.ndarray) or x_view.ndim != 2:
				raise ValueError(
					'transform は 2D numpy または (2D, meta) を返す必要があります'
				)

			# FB をビュー空間へ反映
			hflip = bool(meta.get('hflip', False))
			factor = float(meta.get('factor', 1.0))
			start = int(meta.get('start', 0))
			if hflip:
				fb_subset = fb_subset[::-1].copy()
				offsets = offsets[::-1].copy()

			fb_idx_win = np.floor(fb_subset * factor).astype(np.int64) - start
			W = x_view.shape[1]
			invalid = (fb_idx_win <= 0) | (fb_idx_win >= W)
			fb_idx_win[invalid] = -1

			# FBLC gate（After transform）
			dt_eff_sec = info['dt_sec'] / max(factor, 1e-9)
			ok_fblc, p_ms, valid_pairs = self.fbgate.fblc_accept(
				fb_idx_win, dt_eff_sec=dt_eff_sec
			)
			if not ok_fblc:
				if self.verbose:
					print(
						f'Rejecting gather {info["path"]} key={key_name}:{primary_unique_str} '
						f'(FBLC gate; pairs={valid_pairs}, p_ms={p_ms})'
					)
				continue

			# sample dict 準備（BuildPlan を使う場合ここから）
			sample = {
				'x_view': x_view,  # (H,W)
				'dt_sec': float(dt_eff_sec),
				'offsets': offsets,  # (H,)
				'meta': meta,
				'fb_idx': fb_idx_win,  # (H,)
				'file_path': info['path'],
				'indices': indices,
				'key_name': key_name,
				'secondary_key': secondary_key,
				'primary_unique': primary_unique_str,
			}

			if self.plan is not None:
				# 入力/ターゲット構築（例: MaskedSignal, MakeTimeChannel, …）
				self.plan.run(sample, rng=self._rng)

				# モデル学習にすぐ使える形へ（Tensorは SelectStack 内で作成済み）
				# 返却は 'input' / 'target' / 追加メタ
				return {
					'input': sample['input'],  # torch.Tensor (C,H,W)
					'target': sample['target'],  # torch.Tensor (C2,H,W) など
					'mask_bool': sample.get('mask_bool'),  # (H,T) bool（ある場合）
					'dt_sec': torch.tensor(dt_eff_sec, dtype=torch.float32),
					'fb_idx': torch.from_numpy(fb_idx_win),
					'offsets': torch.from_numpy(offsets),
					'file_path': info['path'],
					'indices': indices,
					'key_name': key_name,
					'secondary_key': secondary_key,
					'primary_unique': primary_unique_str,
					'did_superwindow': did_super if self.verbose else None,
				}

			# BuildPlan を使わない場合の最低限の返却（元波形のみ）
			return {
				'x': torch.from_numpy(x_view[None, ...]),  # (1,H,W)
				'dt_sec': torch.tensor(dt_eff_sec, dtype=torch.float32),
				'fb_idx': torch.from_numpy(fb_idx_win),
				'offsets': torch.from_numpy(offsets),
				'file_path': info['path'],
				'indices': indices,
				'key_name': key_name,
				'secondary_key': secondary_key,
				'primary_unique': primary_unique_str,
				'did_superwindow': did_super if self.verbose else None,
			}
