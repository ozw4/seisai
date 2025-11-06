from torch import nn


class SegmentationHead2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3):
		super().__init__()
		self.conv = nn.Conv2d(
			in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
		)

	def forward(self, x):
		return self.conv(x)
