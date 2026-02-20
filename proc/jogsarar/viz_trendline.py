# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# =========================
# ここだけ直して使う
# =========================
PROB_NPZ_PATH = Path(
    '/home/dcuser/data/ActiveSeisField/jogsarar_out/0020_geom_set_1401.prob.npz'
)

USE_ABS_OFFSET = True  # True: |offset| で global 1本（おすすめ）
OFFSET_SCALE = 1.0  # offsetの単位がmでない場合の補正（ftなら0.3048等）
HUBER_C = 1.345
IRLS_ITERS = 5

MAX_PLOT_POINTS = 300_000  # 散布図の点数上限（重い場合は下げる）
MAX_FIT_POINTS = 1_500_000  # フィットの点数上限（重い場合は下げる）
RANDOM_SEED = 0

# weightsとして使いたいキーがnpzにある場合に自動採用（なければ1）
WEIGHT_KEYS_CANDIDATES = [
    'conf_prob1',
    'conf_rs1',
    'w_conf',
]


# =========================
# ロバスト直線フィット（Huber IRLS）
# =========================
def huber_irls_line_fit(
    x: np.ndarray, y: np.ndarray, w_base: np.ndarray, *, c: float, iters: int
) -> tuple[float, float]:
    """Y ≈ a + b x を weighted + Huber IRLS でフィット
    - w_base: 0以上（信頼度など）。ここにHuber重みが掛かる
    """
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    w = w_base.astype(np.float64, copy=False)

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x = x[m]
    y = y[m]
    w = w[m]

    if x.size < 2:
        raise ValueError('not enough valid points to fit')

    # 初期：base weightのみでWLS
    a, b = _wls_line_fit(x, y, w)

    for _ in range(iters):
        r = y - (a + b * x)

        med = np.median(r)
        mad = np.median(np.abs(r - med))
        scale = max(1.4826 * mad, 1e-6)

        u = r / (c * scale)
        w_h = np.ones_like(u)
        big = np.abs(u) > 1.0
        w_h[big] = 1.0 / np.abs(u[big])

        w_eff = w * w_h
        a, b = _wls_line_fit(x, y, w_eff)

    return float(a), float(b)


def _wls_line_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Weighted least squares for y ≈ a + b x via lstsq on sqrt(w)*X"""
    sw = np.sqrt(w)
    A = np.stack([np.ones_like(x), x], axis=1) * sw[:, None]
    bvec = y * sw
    coef, _, _, _ = np.linalg.lstsq(A, bvec, rcond=None)
    return float(coef[0]), float(coef[1])


# =========================
# データ読込＆可視化
# =========================
z = np.load(PROB_NPZ_PATH)

offsets = np.asarray(z['offsets'], dtype=np.float64) * float(OFFSET_SCALE)
if 'trend_t_sec' in z.files:
    y_src = np.asarray(z['trend_t_sec'], dtype=np.float64)
    if 'trend_covered' in z.files:
        covered = np.asarray(z['trend_covered'], dtype=bool)
    else:
        covered = np.isfinite(y_src)
    y_label = 'trend_t_sec (sec)'
    y_title = 'trend_t_sec'
else:
    if 'pick_final' not in z.files:
        raise ValueError(
            "npz must contain either 'trend_t_sec' or 'pick_final' for visualization"
        )
    dt = float(np.asarray(z['dt_sec']).item())
    pick_final = np.asarray(z['pick_final'], dtype=np.float64)
    y_src = pick_final * dt
    covered = np.isfinite(pick_final) & (pick_final > 0.0)
    if 'n_samples_orig' in z.files:
        ns = int(np.asarray(z['n_samples_orig']).item())
        covered &= pick_final < float(ns)
    y_label = 'pick_final (sec)'
    y_title = 'pick_final(sec, trend_t_sec missing in Stage1)'

# 有効点マスク
mask = covered & np.isfinite(y_src)
x_all = offsets[mask]
y_all = y_src[mask]

if USE_ABS_OFFSET:
    x_all = np.abs(x_all)

# base weight（なければ1）
w_base = np.ones_like(y_all, dtype=np.float64)
for k in WEIGHT_KEYS_CANDIDATES:
    if k in z.files:
        wk = np.asarray(z[k], dtype=np.float64)
        if wk.shape == offsets.shape:
            w_base = wk[mask].clip(min=0.0)
            break

# 点数が多い場合はフィット用にサンプル
rng = np.random.default_rng(RANDOM_SEED)
n_all = x_all.size
if n_all == 0:
    raise ValueError(f'no valid points (covered & finite) in {PROB_NPZ_PATH}')

if n_all > MAX_FIT_POINTS:
    idx_fit = rng.choice(n_all, size=MAX_FIT_POINTS, replace=False)
    x_fit = x_all[idx_fit]
    y_fit = y_all[idx_fit]
    w_fit = w_base[idx_fit]
else:
    x_fit = x_all
    y_fit = y_all
    w_fit = w_base

# global trendline をロバストにフィット
a_g, b_g = huber_irls_line_fit(x_fit, y_fit, w_fit, c=HUBER_C, iters=IRLS_ITERS)

# 描画用に散布点をサンプル
if n_all > MAX_PLOT_POINTS:
    idx_plot = rng.choice(n_all, size=MAX_PLOT_POINTS, replace=False)
    x_plot = x_all[idx_plot]
    y_plot = y_all[idx_plot]
else:
    x_plot = x_all
    y_plot = y_all

# global line
xmin = float(np.nanmin(x_all))
xmax = float(np.nanmax(x_all))
x_line = np.linspace(xmin, xmax, 400)
y_line = a_g + b_g * x_line

# 速度っぽい表示（bがsec/mなら v=1/b）
v_est = np.inf if b_g == 0.0 else 1.0 / abs(b_g)

fig, ax = plt.subplots(figsize=(14, 8))
ax.scatter(
    x_plot,
    y_plot,
    s=2.0,
    alpha=0.5,
    rasterized=True,
    label='all ffid points (sampled)',
)
ax.plot(
    x_line,
    y_line,
    lw=2.0,
    label=f'global line (Huber IRLS): y = {a_g:.4f} + {b_g:.4e} x',
)

ax.set_xlabel('|offset| (scaled)' if USE_ABS_OFFSET else 'offset (scaled)')
ax.set_ylabel(y_label)
ax.set_title(
    f'offset vs {y_title} (all ffid) + global line\n'
    f'N_valid={n_all:,}, fit_N={x_fit.size:,}, plot_N={x_plot.size:,}, '
    f'v_est≈{v_est:.1f} (unit depends on offset)'
)
ax.grid(True, alpha=0.2)
ax.legend(loc='best')
fig.tight_layout()
plt.show()
