# %%
import contextlib
import random

import numpy as np
import segyio
import torch
from seisai_transforms.view_projection import (
	project_fb_idx_view,
	project_offsets_view,
	project_time_view,
)
from torch.utils.data import Dataset

from .builder.builder import (
	BuildPlan,
)
from .config import LoaderConfig, TraceSubsetSamplerConfig
from .file_info import build_file_info
from .gate_fblc import FirstBreakGate
from .trace_subset_preproc import TraceSubsetLoader
from .trace_subset_sampler import TraceSubsetSampler


class SampleSelector:
	def __init__(self, sampler: TraceSubsetSampler) -> None:
		self.sampler = sampler

	def draw_sample(self, info: dict, rng: np.random.Generator) -> dict:
		seed = int(rng.integers(0, 2**31 - 1))
		s = self.sampler.draw(info, py_random=random.Random(seed))
		return {
			'indices': np.asarray(s['indices'], dtype=np.int64),
			'key_name': s['key_name'],
			'secondary_key': s['secondary_key'],
			'did_super': bool(s['did_super']),
			'primary_unique': s['primary_unique'],
		}


class SampleTransformer:
	def __init__(
		self,
		subsetloader: TraceSubsetLoader,
		transform,
	) -> None:
		self.subsetloader = subsetloader
		self.transform = transform

	def load_transform(
		self,
		info: dict,
		indices: np.ndarray,
		fb_subset: np.ndarray,
		rng: np.random.Generator,
	) -> tuple[np.ndarray, dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		mmap = info['mmap']

		# 波形読み出し（subset_traces に満たない場合は loader が H 方向にパッドする）
		x = self.subsetloader.load(mmap, indices)  # (H,W0)
		H = int(x.shape[0])
		W0 = int(x.shape[1])

		# ここで offsets/fb/indices を H に合わせる（仕様）
		H0 = int(indices.size)
		if H0 > H:
			raise ValueError(f'indices length {H0} > loaded H {H}')

		offsets = info['offsets'][indices].astype(np.float32, copy=False)
		fb_subset = np.asarray(fb_subset, dtype=np.int64)

		trace_valid = np.zeros(H, dtype=np.bool_)
		trace_valid[:H0] = True

		if H > H0:
			pad = H - H0
			offsets = np.concatenate(
				[offsets, np.zeros(pad, dtype=np.float32)],
				axis=0,
			)
			fb_subset = np.concatenate(
				[fb_subset, -np.ones(pad, dtype=np.int64)],
				axis=0,
			)
			indices = np.concatenate(
				[indices.astype(np.int64, copy=False), -np.ones(pad, dtype=np.int64)],
				axis=0,
			)
		else:
			indices = indices.astype(np.int64, copy=False)

		# 変換（Crop/Pad / TimeStretch 等）
		out = self.transform(x, rng=rng, return_meta=True)
		x_view, meta = out if isinstance(out, tuple) else (out, {})
		if not isinstance(x_view, np.ndarray) or x_view.ndim != 2:
			raise ValueError(
				'transform は 2D numpy または (2D, meta) を返す必要があります'
			)

		Hv, W = x_view.shape
		if Hv != H:
			raise ValueError(f'transform must keep H: got Hv={Hv}, expected H={H}')

		t_raw = np.arange(W0, dtype=np.float32) * float(info['dt_sec'])

		meta['trace_valid'] = trace_valid
		meta['fb_idx_view'] = project_fb_idx_view(fb_subset, H, W, meta)
		meta['offsets_view'] = project_offsets_view(offsets, H, meta)
		meta['time_view'] = project_time_view(t_raw, H, W, meta)

		return x_view, meta, offsets, fb_subset, indices, trace_valid


class GateEvaluator:
	def __init__(self, fbgate: FirstBreakGate, *, verbose: bool = False) -> None:
		self.fbgate = fbgate
		self.verbose = bool(verbose)
		self.last_reject: str | None = None

	def min_pick_accept(self, fb_subset: np.ndarray) -> bool:
		ok_pick, _, _ = self.fbgate.min_pick_accept(fb_subset)
		if not ok_pick:
			self.last_reject = 'min_pick'
			return False
		self.last_reject = None
		return True

	def apply_gates(self, meta: dict, did_super: bool, info: dict) -> bool:
		# FBLC gate（After transform）
		factor = float(meta.get('factor', 1.0))
		dt_eff_sec = info['dt_sec'] / max(factor, 1e-9)
		meta['dt_eff_sec'] = float(dt_eff_sec)
		ok_fblc, p_ms, valid_pairs = self.fbgate.fblc_accept(
			meta['fb_idx_view'], dt_eff_sec=dt_eff_sec, did_super=did_super
		)
		if not ok_fblc:
			if self.verbose:
				print(
					f'Rejecting gather {info["path"]} key={meta.get("key_name", "")}:{meta.get("primary_unique", "")} '
					f'(FBLC gate; pairs={valid_pairs}, p_ms={p_ms})'
				)
			self.last_reject = 'fblc'
			return False

		self.last_reject = None
		return True


class OutputBuilder:
	def __init__(self, plan: BuildPlan, rng: np.random.Generator) -> None:
		if not isinstance(plan, BuildPlan):
			raise TypeError('plan must be BuildPlan')
		self.plan = plan
		self.rng = rng

	def build_output(
		self,
		sample: dict,
		meta: dict,
		fb_subset: np.ndarray,
		offsets: np.ndarray,
		info: dict,
	) -> dict:
		dt_eff_sec = float(meta.get('dt_eff_sec', info['dt_sec']))
		sample_for_plan = {
			'x_view': sample['x_view'],  # (H,W)
			'dt_sec': dt_eff_sec,
			'offsets': offsets,  # (H,)
			'fb_idx': fb_subset,  # (H,)
			'meta': meta,
			'file_path': info['path'],
			'indices': sample['indices'],
			'key_name': sample['key_name'],
			'secondary_key': sample['secondary_key'],
			'primary_unique': sample['primary_unique'],
		}

		self.plan.run(sample_for_plan, rng=self.rng)
		if 'input' not in sample_for_plan or 'target' not in sample_for_plan:
			raise KeyError("plan must populate 'input' and 'target'")

		return {
			'input': sample_for_plan['input'],  # torch.Tensor (C,H,W)
			'target': sample_for_plan['target'],  # torch.Tensor (C2,H,W) など
			'mask_bool': sample_for_plan.get('mask_bool'),  # (H,T) bool（ある場合）
			'meta': meta,  # ビュー情報
			'dt_sec': torch.tensor(dt_eff_sec, dtype=torch.float32),
			'fb_idx': torch.from_numpy(fb_subset),
			'offsets': torch.from_numpy(offsets),
			'file_path': info['path'],
			'indices': sample['indices'],
			'key_name': sample['key_name'],
			'secondary_key': sample['secondary_key'],
			'primary_unique': sample['primary_unique'],
			'did_superwindow': sample['did_super'],
		}


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
		plan: BuildPlan,
		*,
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
		max_trials: int = 2048,
		sample_selector: SampleSelector | None = None,
		sample_transformer: SampleTransformer | None = None,
		gate_evaluator: GateEvaluator | None = None,
		output_builder: OutputBuilder | None = None,
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
		self.max_trials = int(max_trials)

		# components
		if sample_selector is None or sample_transformer is None:
			subsetloader = TraceSubsetLoader(
				LoaderConfig(pad_traces_to=int(subset_traces))
			)
		if sample_selector is None:
			sampler = TraceSubsetSampler(
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
			sample_selector = SampleSelector(sampler)
		if sample_transformer is None:
			sample_transformer = SampleTransformer(subsetloader, transform)
		if gate_evaluator is None:
			gate_evaluator = GateEvaluator(fbgate, verbose=self.verbose)
		if output_builder is None:
			output_builder = OutputBuilder(plan, self._rng)

		self.sample_selector = sample_selector
		self.sample_transformer = sample_transformer
		self.gate_evaluator = gate_evaluator
		self.output_builder = output_builder

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
		return 1024

	def __getitem__(self, _=None) -> dict:
		rej_pick = 0
		rej_fblc = 0
		for _attempt in range(self.max_trials):
			fidx = int(self._rng.integers(0, len(self.file_infos)))
			info = self.file_infos[fidx]
			sample = self.sample_selector.draw_sample(info, self._rng)
			indices = sample['indices']
			did_super = sample['did_super']

			fb_subset = info['fb'][indices]
			if not self.gate_evaluator.min_pick_accept(fb_subset):
				rej_pick += 1
				continue

			x_view, meta, offsets, fb_subset = self.sample_transformer.load_transform(
				info, indices, fb_subset, self._rng
			)
			meta['key_name'] = sample['key_name']
			meta['primary_unique'] = sample['primary_unique']
			sample['x_view'] = x_view

			if not self.gate_evaluator.apply_gates(meta, did_super, info):
				if self.gate_evaluator.last_reject == 'fblc':
					rej_fblc += 1
				continue

			output = self.output_builder.build_output(
				sample, meta, fb_subset, offsets, info
			)
			if not self.verbose:
				output['did_superwindow'] = None
			return output

		raise RuntimeError(
			f'failed to draw a valid sample within max_trials={self.max_trials}; '
			f'rejections: min_pick={rej_pick}, fblc={rej_fblc}, '
			f'files={len(self.file_infos)}'
		)
