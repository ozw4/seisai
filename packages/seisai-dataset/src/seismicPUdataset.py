from __future__ import annotations

import csv
import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class WindowItem:
	npy_path: str
	start_index: int  # inclusive, in samples
	label: int  # 1=event, 0=non-event(unlabeled)
	start_time_iso: str
	p_offset_samples: int | None


def parse_p_arrivals_local(csv_path: str) -> list[datetime]:
	if not os.path.isfile(csv_path):
		raise FileNotFoundError(f'csv not found: {csv_path}')
	arrivals: list[datetime] = []
	with open(csv_path, newline='') as f:
		reader = csv.DictReader(f)
		if 'P_arrival_time_local' not in reader.fieldnames:
			raise KeyError(
				f"CSV must contain 'P_arrival_time_local'. Found: {reader.fieldnames}"
			)
		for row in reader:
			s = row['P_arrival_time_local'].strip()
			dt = datetime.fromisoformat(s)
			if dt.tzinfo is None:
				raise ValueError(f'P_arrival_time_local must be timezone-aware: {s}')
			arrivals.append(dt)
	return arrivals


def format_day_path(
	root_dir: str, day_jst: datetime, fs: int, pad_seconds: int, filename_template: str
) -> str:
	"""filename_template supports tokens:
	  {YYYYMMDD}, {fs}, {pad}
	Default matches 'JST_20200211_100Hz_pad30s.npy'
	"""
	ymd = day_jst.strftime('%Y%m%d')
	fname = filename_template.format(YYYYMMDD=ymd, fs=fs, pad=pad_seconds)
	return os.path.join(root_dir, fname)


def jst_day_time_bounds(day: datetime, pad_seconds: int) -> tuple[datetime, datetime]:
	jst = ZoneInfo('Asia/Tokyo')
	day0 = datetime(day.year, day.month, day.day, 0, 0, 0, tzinfo=jst)
	return day0 - timedelta(seconds=pad_seconds), day0 + timedelta(
		days=1, seconds=pad_seconds
	)


def day_iter(start_date: str, end_date: str) -> Iterable[datetime]:
	if len(start_date) != 8 or len(end_date) != 8:
		raise ValueError("start_date/end_date must be 'YYYYMMDD'")
	jst = ZoneInfo('Asia/Tokyo')
	s = datetime(
		int(start_date[:4]), int(start_date[4:6]), int(start_date[6:8]), tzinfo=jst
	)
	e = datetime(int(end_date[:4]), int(end_date[4:6]), int(end_date[6:8]), tzinfo=jst)
	if e < s:
		raise ValueError('end_date must be >= start_date')
	cur = s
	while cur <= e:
		yield cur
		cur = cur + timedelta(days=1)


def build_allowed_negative_start_ranges(
	t0: datetime,
	t1: datetime,
	event_times: list[datetime],
	win_sec: float,
	margin_sec: float,
) -> list[tuple[datetime, datetime]]:
	if t1 <= t0:
		raise ValueError('t1 must be after t0')
	if win_sec <= 0.0:
		raise ValueError('win_sec must be positive')
	left = t0
	right = t1 - timedelta(seconds=win_sec)
	if right <= left:
		return []
	intervals: list[tuple[datetime, datetime]] = []
	for e in event_times:
		a = e - timedelta(seconds=(margin_sec + win_sec))
		b = e + timedelta(seconds=margin_sec)
		a = max(a, left)
		b = min(b, right)
		if b > a:
			intervals.append((a, b))
	intervals.sort(key=lambda x: x[0])
	merged: list[tuple[datetime, datetime]] = []
	for a, b in intervals:
		if not merged or a > merged[-1][1]:
			merged.append([a, b])  # type: ignore[list-item]
		else:
			merged[-1][1] = max(merged[-1][1], b)  # type: ignore[index]
	allowed: list[tuple[datetime, datetime]] = []
	cur = left
	for a, b in merged:
		if a > cur:
			allowed.append((cur, a))
		cur = max(cur, b)
	if cur < right:
		allowed.append((cur, right))
	return allowed


class SeismicPUWindowDataset(Dataset):
	"""60 s 窓を前提にしていた実装を一般化。
	任意の fs（サンプリング周波数）と win_sec（窓長）、pad_seconds（前後パディング秒）に対応。

	- npy: shape (C, N)
	- 1ファイルの時間範囲: [JST日付00:00:00 - pad, 翌日00:00:00 + pad)
	- 期待サンプル数: (86400 + 2*pad_seconds) * fs
	- 正例: P到達の pre_event_sec 前から win_sec の窓
	- 負例: すべてのイベント時刻 e に対し、開始時刻が (e - margin - win_sec, e + margin) の外側
	"""

	def __init__(
		self,
		root_dir: str,
		csv_path: str,
		start_date: str,
		end_date: str,
		win_sec: float,
		fs: int,
		pre_event_sec: float = 10.0,
		neg_per_pos: int = 1,
		neg_margin_sec: float = 120.0,
		channel_indices: list[int] | None = None,
		pad_seconds: int = 30,
		filename_template: str = 'JST_{YYYYMMDD}_{fs}Hz_pad{pad}s.npy',
		seed: int = 42,
		validate_shape: bool = True,
	):
		self.root_dir = os.fspath(root_dir)
		self.csv_path = os.fspath(csv_path)
		self.win_sec = float(win_sec)
		self.fs = int(fs)
		self.win_samples = int(round(self.win_sec * self.fs))
		self.pre_event_sec = float(pre_event_sec)
		self.pre_event_samples = int(round(self.pre_event_sec * self.fs))
		self.neg_per_pos = int(neg_per_pos)
		self.neg_margin_sec = float(neg_margin_sec)
		self.channel_indices = channel_indices
		self.pad_seconds = int(pad_seconds)
		self.filename_template = str(filename_template)
		self.seed = int(seed)
		self.validate_shape = bool(validate_shape)

		if self.win_samples <= 0:
			raise ValueError('win_sec * fs must be >= 1 sample')
		if not (0.0 <= self.pre_event_sec < self.win_sec):
			raise ValueError('pre_event_sec must satisfy 0 <= pre_event_sec < win_sec')

		self.jst = ZoneInfo('Asia/Tokyo')
		self.arrivals = parse_p_arrivals_local(self.csv_path)

		items: list[WindowItem] = []
		rng = np.random.default_rng(self.seed)

		for day0 in day_iter(start_date, end_date):
			t0, t1 = jst_day_time_bounds(day0, self.pad_seconds)
			npy_path = format_day_path(
				self.root_dir, day0, self.fs, self.pad_seconds, self.filename_template
			)
			if not os.path.isfile(npy_path):
				raise FileNotFoundError(
					f'npy not found for {day0.strftime("%Y-%m-%d")}: {npy_path}'
				)

			hdr = np.load(npy_path, mmap_mode='r')
			if hdr.ndim != 2:
				raise ValueError(
					f'npy must be 2D (C, N). Got {hdr.shape} at {npy_path}'
				)
			expected_n = (86400 + 2 * self.pad_seconds) * self.fs
			if self.validate_shape and hdr.shape[1] != expected_n:
				raise ValueError(
					f'unexpected N at {npy_path}: {hdr.shape[1]} vs {expected_n}'
				)
			if self.channel_indices is not None:
				for c in self.channel_indices:
					if c < 0 or c >= hdr.shape[0]:
						raise IndexError(
							f'channel index {c} out of range [0, {hdr.shape[0] - 1}]'
						)

			pos_times_day: list[datetime] = []
			for e in self.arrivals:
				s = e - timedelta(seconds=self.pre_event_sec)
				if s >= t0 and (s + timedelta(seconds=self.win_sec)) <= t1:
					pos_times_day.append(e)

			for e in pos_times_day:
				s = e - timedelta(seconds=self.pre_event_sec)
				start_index = int(round((s - t0).total_seconds() * self.fs))
				if start_index < 0 or (start_index + self.win_samples) > expected_n:
					raise ValueError('positive window out of bounds')
				items.append(
					WindowItem(
						npy_path=npy_path,
						start_index=start_index,
						label=1,
						start_time_iso=s.isoformat(),
						p_offset_samples=self.pre_event_samples,
					)
				)

			allowed_ranges = build_allowed_negative_start_ranges(
				t0=t0,
				t1=t1,
				event_times=pos_times_day,
				win_sec=self.win_sec,
				margin_sec=self.neg_margin_sec,
			)
			n_pos = len(pos_times_day)
			n_neg_target = n_pos * self.neg_per_pos if n_pos > 0 else self.neg_per_pos

			neg_starts: list[datetime] = []
			if n_neg_target > 0 and len(allowed_ranges) > 0:
				lengths = [max(0.0, (b - a).total_seconds()) for a, b in allowed_ranges]
				total = sum(lengths)
				if total > 0.0:
					for _ in range(n_neg_target):
						u = rng.random() * total
						acc = 0.0
						for (a, b), L in zip(allowed_ranges, lengths, strict=False):
							if acc + L >= u:
								offset = u - acc
								s_time = a + timedelta(seconds=offset)
								neg_starts.append(s_time)
								break
							acc += L

			for s in neg_starts:
				start_index = int(round((s - t0).total_seconds() * self.fs))
				if start_index < 0 or (start_index + self.win_samples) > expected_n:
					raise ValueError('negative window out of bounds')
				items.append(
					WindowItem(
						npy_path=npy_path,
						start_index=start_index,
						label=0,
						start_time_iso=s.isoformat(),
						p_offset_samples=None,
					)
				)

		if len(items) == 0:
			raise ValueError('no windows collected')
		self.items: list[WindowItem] = items

	def __len__(self) -> int:
		return len(self.items)

	def __getitem__(self, idx: int):
		if idx < 0 or idx >= len(self.items):
			raise IndexError('index out of range')
		it = self.items[idx]
		arr = np.load(it.npy_path, mmap_mode='r')  # (C, N)
		if self.channel_indices is None:
			x = arr[:, it.start_index : it.start_index + self.win_samples]
		else:
			x = arr[
				self.channel_indices, it.start_index : it.start_index + self.win_samples
			]
		if x.shape[1] != self.win_samples:
			raise ValueError(
				f'slice length {x.shape[1]} != win_samples {self.win_samples}'
			)
		x_t = torch.from_numpy(np.ascontiguousarray(x)).float()
		y_t = torch.tensor([it.label], dtype=torch.int64)
		meta: dict[str, object] = {
			'start_time_iso': it.start_time_iso,
			'npy_path': it.npy_path,
			'start_index': it.start_index,
			'fs': self.fs,
			'win_sec': self.win_sec,
			'pad_seconds': self.pad_seconds,
			'p_offset_samples': it.p_offset_samples,
		}
		return x_t, y_t, meta
