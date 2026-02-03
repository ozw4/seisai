"""Logging utilities for tracking and reporting smoothed metrics during training."""

import datetime
import itertools
import time
from collections import defaultdict, deque
from collections.abc import Iterable

import torch
import torch.distributed as dist

from seisai_utils.dist import is_dist_avail_and_initialized


class SmoothedValue:
	"""Track a series of values and provide access to smoothed values over a.

	window or the global series average.
	"""

	def __init__(self, window_size: int = 20, fmt: str | None = None) -> None:
		"""Initialize the smoothed value tracker with a window size and format."""
		if fmt is None:
			fmt = '{median:.4f} ({global_avg:.4f})'
		self.deque = deque(maxlen=window_size)
		self.total = 0.0
		self.count = 0
		self.fmt = fmt

	def update(self, value: float, n: int = 1) -> None:
		self.deque.append(value)
		self.count += n
		self.total += value * n

	def synchronize_between_processes(self) -> None:
		"""Warning: does not synchronize the deque!."""
		if not is_dist_avail_and_initialized():
			return
		t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
		dist.barrier()
		dist.all_reduce(t)
		t = t.tolist()
		self.count = int(t[0])
		self.total = t[1]

	@property
	def median(self) -> float:
		d = torch.tensor(list(self.deque))
		return d.median().item()

	@property
	def avg(self) -> float:
		d = torch.tensor(list(self.deque), dtype=torch.float32)
		return d.mean().item()

	@property
	def global_avg(self) -> float:
		if self.count == 0:
			return 0.0
		return self.total / self.count

	@property
	def max(self) -> float:
		"""Return the maximum value in the current window.

		Raises
		------
		ValueError
			If no values have been recorded yet.

		"""
		return max(self.deque)

	@property
	def value(self) -> float:
		"""Return the most recently added value in the window.

		Raises
		------
		IndexError
			If no values have been recorded yet.

		"""
		return self.deque[-1]

	def __str__(self) -> str:
		"""Return a formatted string representation of the smoothed values."""
		if self.count == 0:
			try:
				return self.fmt.format(median=0, avg=0, global_avg=0, max=0, value=0)
			except (KeyError, IndexError, ValueError, TypeError):
				# if format string is invalid or missing fields, return a simple zero
				return '0'
		return self.fmt.format(
			median=self.median,
			avg=self.avg,
			global_avg=self.global_avg,
			max=self.max,
			value=self.value,
		)


class MetricLogger:
	"""Log and synchronize multiple smoothed metrics during iteration.

	Attributes
	----------
	meters : dict[str, SmoothedValue]
		Mapping of metric names to smoothed value trackers.
	delimiter : str
		Delimiter used when formatting metric output.

	"""

	def __init__(self, delimiter: str = '\t') -> None:
		"""Initialize a metric logger with an optional delimiter for output formatting."""
		self.meters = defaultdict(SmoothedValue)
		self.delimiter = delimiter

	def update(self, **kwargs):
		for k, v in kwargs.items():
			if isinstance(v, torch.Tensor):
				v = v.item()
			if not isinstance(v, (float, int)):
				msg = f"Metric '{k}' must be a float or int, got {type(v).__name__}"
				raise TypeError(msg)
			self.meters[k].update(v)

	def __getattr__(self, attr: str) -> SmoothedValue:
		"""Provide attribute-style access to registered meters.

		Parameters
		----------
		attr : str
			Name of the meter to retrieve.

		Returns
		-------
		SmoothedValue
			The requested meter instance.

		Raises
		------
		AttributeError
			If the attribute is not a registered meter and not present on the instance.

		"""
		if attr in self.meters:
			return self.meters[attr]
		if attr in self.__dict__:
			return self.__dict__[attr]
		msg = f"'{type(self).__name__}' object has no attribute '{attr}'"
		raise AttributeError(msg)

	def __str__(self) -> str:
		"""Return a formatted string representation of all tracked metrics."""
		loss_str = []
		for name, meter in self.meters.items():
			loss_str.append(f'{name}: {meter!s}')
		return self.delimiter.join(loss_str)

	def synchronize_between_processes(self) -> None:
		"""Synchronize all registered meters across distributed processes.

		This calls :meth:`SmoothedValue.synchronize_between_processes` on every meter,
		updating each meter's global `count` and `total` via `torch.distributed`.
		"""
		for meter in self.meters.values():
			meter.synchronize_between_processes()

	def add_meter(self, name: str, meter: SmoothedValue) -> None:
		"""Add a pre-configured meter to the logger.

		Parameters
		----------
		name : str
			The metric name to associate with the meter.
		meter : SmoothedValue
			The smoothed value tracker instance to register.

		"""
		self.meters[name] = meter

	def log_every(self, iterable: Iterable, print_freq: int, header: str | None = None):
		if isinstance(iterable, list):
			length = max(len(x) for x in iterable)
			iterable = [x if len(x) == length else itertools.cycle(x) for x in iterable]
			iterable = zip(*iterable, strict=False)
		else:
			length = len(iterable)
		if not header:
			header = ''
		start_time = time.time()
		end = time.time()
		iter_time = SmoothedValue(fmt='{avg:.4f}')
		data_time = SmoothedValue(fmt='{avg:.4f}')
		space_fmt = ':' + str(len(str(length))) + 'd'
		if torch.cuda.is_available():
			log_msg = self.delimiter.join(
				[
					header,
					'[{0' + space_fmt + '}/{1}]',
					'eta: {eta}',
					'{meters}',
					'time: {time}',
					'data: {data}',
					'max mem: {memory:.0f}',
				]
			)
		else:
			log_msg = self.delimiter.join(
				[
					header,
					'[{0' + space_fmt + '}/{1}]',
					'eta: {eta}',
					'{meters}',
					'time: {time}',
					'data: {data}',
				]
			)
		MB = 1024.0 * 1024.0
		for i, obj in enumerate(iterable):
			data_time.update(time.time() - end)
			yield obj  # <-- yield the batch in for loop
			iter_time.update(time.time() - end)
			if i % print_freq == 0:
				eta_seconds = iter_time.global_avg * (length - i)
				eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
				if torch.cuda.is_available():
					print(
						log_msg.format(
							i,
							length,
							eta=eta_string,
							meters=str(self),
							time=str(iter_time),
							data=str(data_time),
							memory=torch.cuda.max_memory_allocated() / MB,
						)
					)
				else:
					print(
						log_msg.format(
							i,
							length,
							eta=eta_string,
							meters=str(self),
							time=str(iter_time),
							data=str(data_time),
						)
					)
			end = time.time()
		total_time = time.time() - start_time
		total_time_str = str(datetime.timedelta(seconds=int(total_time)))
		print(f'{header} Total time: {total_time_str}')
