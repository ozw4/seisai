# %%
# パラメータ・スイープ例（可視化つき、subplot禁止のため全て単独Figure）
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from seisai_transforms.masking import MaskGenerator

# ---- 合成データ（同一）----
H, T = 64, 256
base_rng = np.random.default_rng(0)
t = np.linspace(0.0, 1.0, T, dtype=np.float32)
x = np.empty((H, T), dtype=np.float32)
for h in range(H):
	f = 3.0 + 6.0 * (h / max(1, H - 1))
	phase = 2.0 * np.pi * base_rng.random()
	x[h] = np.sin(2.0 * np.pi * f * t + phase)
x += 0.1 * base_rng.standard_normal(size=x.shape).astype(np.float32)


# ---- ユーティリティ ----
def show(img: np.ndarray, title: str) -> None:
	plt.figure()
	plt.imshow(img.T, aspect='auto')
	plt.title(title)


def demo_apply(gen, x: np.ndarray, rng_seed: int, tag: str) -> None:
	rng = np.random.default_rng(rng_seed)
	xm, m = gen.apply(x, rng=rng, return_mask=True)
	cov = float(m.mean())  # 被覆率
	show(m, f'{tag} | mask bool (cover={cov:.3f})')
	show(xm, f'{tag} | masked')


# ---- オリジナル一枚だけ表示 ----
show(x, 'Original (H,T)')

# =========================
# 1) トレース帯マスクのスイープ
# =========================
trace_cases = [
	dict(ratio=0.15, width=2, mode='replace', noise_std=1.0, seed=100),
	dict(ratio=0.35, width=4, mode='replace', noise_std=0.3, seed=101),
	dict(
		ratio=0.50, width=8, mode='replace', noise_std=0.0, seed=102
	),  # 置換・ゼロ埋め
	dict(ratio=0.25, width=6, mode='add', noise_std=0.5, seed=103),
	dict(
		ratio=0.40, width=10, mode='add', noise_std=0.0, seed=104
	),  # 加算・ノイズ0=見かけ変化なし
]

for i, c in enumerate(trace_cases):
	gen = MaskGenerator.traces(
		ratio=c['ratio'], width=c['width'], mode=c['mode'], noise_std=c['noise_std']
	)
	tag = f'traces #{i + 1}: ratio={c["ratio"]}, width={c["width"]}, mode={c["mode"]}, noise_std={c["noise_std"]}'
	demo_apply(gen, x, c['seed'], tag)

# =========================
# 2) チェッカージッターのスイープ
# =========================
checker_cases = [
	dict(
		block_h=6,
		block_t=12,
		cell_h=12,
		cell_t=24,
		jitter_h=2,
		jitter_t=4,
		keep_prob=0.8,
		offset_h=0,
		offset_t=0,
		mode='replace',
		noise_std=1.0,
		seed=200,
	),
	dict(
		block_h=8,
		block_t=16,
		cell_h=16,
		cell_t=32,
		jitter_h=4,
		jitter_t=8,
		keep_prob=0.6,
		offset_h=0,
		offset_t=0,
		mode='add',
		noise_std=0.5,
		seed=201,
	),
	dict(
		block_h=12,
		block_t=24,
		cell_h=24,
		cell_t=48,
		jitter_h=0,
		jitter_t=0,
		keep_prob=0.4,
		offset_h=0,
		offset_t=0,
		mode='replace',
		noise_std=0.3,
		seed=202,
	),
	# 位相ズラし（offset でタイルの位相を変更 → 推論時アンサンブルの例）
	dict(
		block_h=8,
		block_t=16,
		cell_h=16,
		cell_t=32,
		jitter_h=0,
		jitter_t=0,
		keep_prob=1.0,
		offset_h=8,
		offset_t=16,
		mode='replace',
		noise_std=1.0,
		seed=203,
	),
	dict(
		block_h=8,
		block_t=16,
		cell_h=16,
		cell_t=32,
		jitter_h=0,
		jitter_t=0,
		keep_prob=1.0,
		offset_h=12,
		offset_t=20,
		mode='replace',
		noise_std=1.0,
		seed=204,
	),
]


plt.show()
