import torch
from torch import nn

from .blocks import DecoderBlock2d


class UnetDecoder2d(nn.Module):
	def __init__(
		self,
		encoder_channels: tuple[int],
		skip_channels: tuple[int] = None,
		decoder_channels: tuple = (256, 128, 64, 32),
		scale_factors: tuple = (2, 2, 2, 2),
		norm_layer: nn.Module = nn.Identity,
		attention_type: str = 'scse',
		intermediate_conv: bool = True,
		upsample_mode: str = 'bilinear',
	):
		super().__init__()

		# 期待段数 = encoder_levels - 1
		need = len(encoder_channels) - 1

		# --- decoder_channels を need に合わせる ---
		dec = list(decoder_channels)
		if len(dec) < need:
			dec += [dec[-1]] * (need - len(dec))  # 末尾を繰り返して延長
		elif len(dec) > need:
			dec = dec[:need]  # 余剰をカット
		decoder_channels = tuple(dec)
		self.decoder_channels = decoder_channels

		# --- scale_factors も need に合わせる(★これが今回の修正ポイント) ---
		sf = list(
			scale_factors
			if isinstance(scale_factors, (list, tuple))
			else [scale_factors]
		)
		if len(sf) < len(decoder_channels):
			sf += [sf[-1]] * (len(decoder_channels) - len(sf))
		elif len(sf) > len(decoder_channels):
			sf = sf[: len(decoder_channels)]
		self.scale_factors = tuple(sf)

		# skip_channels 安全化
		if skip_channels is None:
			skip_channels = list(encoder_channels[1:]) + [0]
		if len(skip_channels) < len(decoder_channels):
			skip_channels += [0] * (len(decoder_channels) - len(skip_channels))
		else:
			skip_channels = skip_channels[: len(decoder_channels)]

		in_channels = [encoder_channels[0]] + list(decoder_channels[:-1])

		self.blocks = nn.ModuleList(
			[
				DecoderBlock2d(
					ic,
					sc,
					dc,
					norm_layer,
					attention_type,
					intermediate_conv,
					upsample_mode,
					self.scale_factors[i],  # ← 調整後を使う
				)
				for i, (ic, sc, dc) in enumerate(
					zip(in_channels, skip_channels, decoder_channels, strict=False)
				)
			]
		)

	def forward(self, feats: list[torch.Tensor]):
		res = [feats[0]]
		feats = feats[1:]
		for i, b in enumerate(self.blocks):
			skip = feats[i] if i < len(feats) else None
			res.append(b(res[-1], skip=skip))
		return res
