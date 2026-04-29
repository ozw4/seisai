# SeisAI `fbpick` 実装計画（初期 v0 / historical）

Version: v0.3
Date: 2026-04-03
Status: historical

## 0. この文書の扱い

このファイルは初期 `fbpick` v0 実装計画の履歴である。
`fbpick-coarse` の local-crop / tiled coarse 前提は現在の仕様ではない。
coarse stage の正本は `docs/fbpick_coarse_global_anchor.md` であり、固定
`global_anchor_resize` contract (`3 x 256 x 2048`) を使う。

本書の physics / fine に関する背景メモは参照用として残すが、coarse の設計・
テスト・example config を変更する場合は `docs/fbpick_coarse_global_anchor.md`
を優先する。

---

## 1. v0 の実装スコープ

### 1.1 訓練フェーズ

1. **Coarse モデル学習**
2. **Physics / robustification**
3. **Fine モデル学習**

### 1.2 推論フェーズ

4. **Coarse 推論**
5. **Physics / robustification**
6. **Fine 推論**

### 1.3 v0 では実装しないもの

以下は将来拡張であり、今回の Codex 実装対象外とする。

- 高信頼度ピック抽出の本格運用
- 自己学習 / fine-tuning / 再推論ループ
- inversion adapter の solver 本体接続
- signed offset の採用
- domain の shot 以外への拡張

---

## 2. 実装前に必ず確認する既存ファイル

Codex は書き始める前に、少なくとも次の既存ファイルを読むこと。

### 2.1 dataset / build plan / transform

- `packages/seisai-dataset/src/seisai_dataset/builder/builder.py`
- `packages/seisai-dataset/src/seisai_dataset/builder/__init__.py`
- `packages/seisai-dataset/src/seisai_dataset/infer_window_dataset.py`
- `packages/seisai-dataset/src/seisai_dataset/segy_gather_pipeline_dataset.py`
- `packages/seisai-dataset/src/seisai_dataset/segy_gather_phase_pipeline_dataset.py`
- `packages/seisai-dataset/src/seisai_dataset/segy_gather_base.py`
- `packages/seisai-dataset/src/seisai_dataset/transform_flow_utils.py`
- `packages/seisai-dataset/src/seisai_dataset/target_fb.py`
- `packages/seisai-transforms/src/seisai_transforms/augment.py`

### 2.2 engine / model / loss / infer

- `packages/seisai-engine/src/seisai_engine/pipelines/common/config_loaders.py`
- `packages/seisai-engine/src/seisai_engine/pipelines/common/encdec2d_model.py`
- `packages/seisai-engine/src/seisai_engine/pipelines/common/init_weights.py`
- `packages/seisai-engine/src/seisai_engine/pipelines/common/tiled_infer.py`
- `packages/seisai-engine/src/seisai_engine/infer/runner.py`
- `packages/seisai-engine/src/seisai_engine/loss/fbseg_kl_loss.py`
- `packages/seisai-engine/src/seisai_engine/pipelines/psn/config.py`
- `packages/seisai-engine/src/seisai_engine/pipelines/psn/build_dataset.py`
- `packages/seisai-engine/src/seisai_engine/pipelines/psn/train.py`
- `packages/seisai-engine/src/seisai_engine/pipelines/blindtrace/build_dataset.py`
- `packages/seisai-models/src/seisai_models/models/encdec2d.py`

### 2.3 examples / 参考資産

- `examples/example_train_fbp.py`
- `proc/jogsarar/stage1/cfg.py`
- `proc/jogsarar/stage1/process_one.py`
- `proc/jogsarar/stage2/cfg.py`
- `proc/jogsarar/stage2/core.py`
- `proc/jogsarar/stage4/cfg.py`
- `proc/jogsarar/jogsarar_viz/stage1_gather.py`
- `proc/jogsarar/jogsarar_viz/stage4_gather.py`
- `proc/jogsarar/common/NPZ_CONTRACT.md`

### 2.4 重要な注意

`proc/jogsarar` は**参照元**であり、`packages/*` 配下の本実装から **runtime dependency を張ってはいけない**。
理由は次のとおり。

- `proc/jogsarar` には script-local import が混ざっている
- package としての import 安定性を前提にしていない
- monorepo の install/export 境界に載っていない

したがって、pure なロジックを使いたい場合は、`packages/seisai-pick` または新設 `packages/seisai-engine/src/seisai_engine/pipelines/fbpick/*` へ**必要最小限を移植**し、テストを付けること。

---

## 3. 既存資産のうち流用するもの / そのままでは使えないもの

## 3.1 そのまま流用するもの

### dataset / build plan

- `SegyGatherPipelineDataset`
- `RandomCropOrPad`
- `DeterministicCropOrPad`
- `PerTraceStandardize`
- `MakeTimeChannel`
- `MakeOffsetChannel`
- `SelectStack`
- `InputOnlyPlan`
- `BuildPlan`

### model / loss / infer

- `EncDec2D`
- `build_encdec2d_kwargs()` / `build_encdec2d_model()` の流儀
- `FbSegKLLossView`
- `TiledWConfig`, `TiledHConfig`
- `infer_batch_tiled_w`, `infer_batch_tiled_h`
- `iter_infer_loader_tiled_w`, `iter_infer_loader_tiled_h`

### pick / confidence / trend

- `seisai_pick.residual_statics`
- `seisai_pick.score.confidence_from_prob`
- `seisai_pick.score.confidence_from_residual_statics`
- `seisai_pick.score.confidence_from_trend_resid`
- `seisai_pick.trend.trend_fit`
- `seisai_pick.trend.gaussian_prior_from_trend`

## 3.2 そのままでは使えないもの

### PSN pipeline の制約

現行 PSN pipeline は config loader の時点で次を強制する。

- `model.in_chans == 1`
- `model.out_chans == 3`

今回必要なのは次であり、PSN をそのまま流用してはならない。

- Coarse: `in_chans=3`, `out_chans=1`
- Fine: `in_chans=1`, `out_chans=1`

### `MakeTimeChannel`

`MakeTimeChannel` は `meta['time_view']` を 2D 展開するだけで、**固定スケール正規化**を持たない。
そのため、後段に `NormalizeTimeByConst` が必要である。

### `MakeOffsetChannel`

`MakeOffsetChannel(normalize=True)` は **view 内 z-score** であり、今回必要な
**corpus 共通の絶対物理座標正規化**とは意味が異なる。
使う場合は `normalize=False` で生値展開し、後段に `NormalizeOffsetByConst` を挟む。

### `FBGaussMap`

既存 `FBGaussMap` は `sigma` を **bin 単位固定値**で受ける。
今回必要なのは **ms 単位固定値 → `dt` 依存で sample 化**なので、そのままでは使えない。

### `InferenceGatherWindowsDataset`

現状の `InferenceGatherWindowsDataset` は `fb_files` を必須にしており、
**raw SGY のみを入れる coarse inference 入口**としては不適切である。
`fb_files=None` を受けられるように拡張するか、同等の raw-only dataset を追加する必要がある。

### `DeterministicCropOrPad`

`DeterministicCropOrPad(target_len=L)` は `W>L` のとき**中央 crop**する。
Coarse 推論の raw SGY では中央 crop をしてはいけない。
`W>6016` の record は **W タイル推論**で扱う。

### `RandomCropOrPad`

`RandomCropOrPad(target_len=L)` は **pick-aware ではない**。
Coarse 学習で `W>L` の record を扱う場合、P 初動が view に入らない window が出るので、
今回の要件には不十分である。

### `train.init_ckpt`

既存 `maybe_load_init_weights()` は `in_chans` 不一致で即失敗する。
この厳格性は維持する。
Fine への重み転用は、**専用 checkpoint 変換 utility**で解決し、
`common/init_weights.py` 自体は緩めない。

---

## 4. 固定するアーキテクチャ方針

### 4.1 ドメインと学習方針

- v0 は **shot domain** のみを対象にする
- 複数 survey を **1 モデル群**で扱う
- `dt` 差があっても **リサンプリングしない**
- 異なるサイズは **固定 window 学習 + tile 推論**で吸収する

### 4.2 パイプライン構成

実装するのは次の 2 段 + 1 モジュールである。

1. **Coarse**: 大域的な P 到来帯を読む
2. **Physics / robustification**: 粗ピックを物理整合で補正・棄却・補完する
3. **Fine**: `robust_pick_i` 周辺の局所窓から最終 pick を読む

### 4.3 役割分担

- Coarse は **物理座標 aware** にする
- Fine は **waveform only** にする
- Coarse / Fine ともに **点回帰ではなく heatmap** を出す
- Physics は深層学習モデルと分離する
- `global_qc` package / CLI は v0 では作らない

---

## 5. Coarse モデル仕様（固定値）

## 5.1 入力

Coarse 入力は **3ch**。

1. `waveform`
2. `offset_ch`
3. `time_ch`

入力 shape:

```text
(B, 3, 128, 6016)
```

## 5.2 学習 window

- `trace_len = 128`
- `time_len = 6016`

## 5.3 推論 window / tile

- trace 方向は `128` trace window を列挙する
- `overlap_h = 96`
- したがって window 列挙の stride は `32`
- time 方向は model 入力幅 `6016` を維持する
- `W <= 6016` なら右 pad のみ
- `W > 6016` なら **W tile 推論**を使う
- raw inference で中央 crop は禁止

### 実装上の写像

SeisAI の既存 runner は H/W 同時 2D tile の public API を持たない。
v0 では次の分解で実装する。

- dataset 側で H window を列挙する
- 各 H window に対して `iter_infer_loader_tiled_w` または `infer_batch_tiled_w` を使い W tile 推論する
- H overlap は pipeline 側で raw trace 軸へ再集約する

## 5.4 waveform 前処理

- `PerTraceStandardize(eps=1e-8)` を基本とする
- `W < 6016` は zero padding
- `W > 6016` の学習 sample では **pick-aware crop** を使う

### pick-aware crop の既定ルール

Coarse 学習で `W0 > 6016` の場合は、subset 内の有効 P pick を `p_valid` として、
**全有効 pick が window に入る start 範囲**から start をサンプルする。

```text
start_lo = max(0, max(p_valid) - (L - 1))
start_hi = min(min(p_valid), W0 - L)
```

- `L = 6016`
- `start_lo <= start_hi` のとき、その整数範囲から一様サンプルする
- feasible interval が空の sample は reject して再サンプルする
- 暗黙 fallback は入れない

このため、Coarse train dataset では `RandomCropOrPad` 単独ではなく、
**pick-aware な `SampleTransformer`** を pipeline 側で差し込むこと。

## 5.5 offset / time 座標表現

### time channel

`MakeTimeChannel` の出力を固定スケールで正規化する。

```text
time_abs_sec[j] = original_record_time_of_view_sample_j
time_ch = clip(time_abs_sec / T_ref_sec, 0.0, 1.5)
```

- `T_ref_sec` は training corpus 全体の `time_abs_sec` 分布 p99 で固定する
- survey ごとの min-max は採用しない

### offset channel

offset は **絶対値**を既定にする。

```text
offset_abs_m = abs(offset_m)
offset_ch = clip(offset_abs_m / O_ref_m, 0.0, 1.5)
```

- `O_ref_m` は training corpus 全体の `abs(offset_m)` 分布 p99 で固定する
- survey ごとの z-score は採用しない
- signed offset は将来検討に回す

## 5.6 target

- 出力は **P-only 1ch heatmap**
- 教師は trace ごとの GT P pick を中心にした Gaussian
- `coarse_sigma_ms = 10.0`

```text
sigma_samples = coarse_sigma_ms / (dt_sec * 1000.0)
sigma_samples = clip(sigma_samples, 4.0, 40.0)
```

`FBGaussMapMs` は `meta['dt_eff_sec']` があればそれを優先し、無ければ `meta['dt_sec']` を使うこと。

## 5.7 loss

v0 では既存 `FbSegKLLossView(tau=1.0, eps=0.0)` を使う。
Coarse 用の新規 bespoke loss は追加しない。

## 5.8 backbone

- 標準: `caformer_s18`
- 比較ベースライン: `convnext_small`
- 大型候補: `caformer_b36.sail_in22k_ft_in1k`

---

## 6. Physics / robustification 仕様（固定方針）

## 6.1 基本思想

Physics は深層学習モデルと分離したモジュールとして実装する。
v0 は **physics-lite** を先に成立させる。

目的は次のとおり。

- coarse pick の明らかな外れを弾く
- feasible band / trend / confidence でロバスト中心を作る
- 必要なら observed pick を理論値または trend で置換する
- Fine 用の `robust_pick_i` を決める

## 6.2 v0 の段階

### Phase A: physics-lite

1. `coarse_pick_i`, `coarse_pmax` を得る
2. feasible band を掛ける
3. trendline を求める
4. confidence を算出する
5. keep / reject / fill を決める
6. `robust_pick_i` を出す

### Phase B: inversion adapter の殻

solver 本体は未接続でよいが、将来差し替え可能な interface は用意する。

```python
fit(high_conf_pick_table) -> model_artifact
predict(model_artifact, geometry_table) -> theoretical_pick_sec
```

### Phase C: observed / theoretical merge の置き場固定

observed/theoretical/trend の統合ルールを 1 箇所に固定する。
Fine / self-training / visualization が同じ `robust.npz` を正本参照できるようにする。

## 6.3 初期値（`proc/jogsarar` 由来）

### feasible band

- `vmin_mask = 100.0 m/s`
- `vmax_mask = 5000.0 m/s`
- `t0_lo_ms = -10.0`
- `t0_hi_ms = 100.0`
- `taper_ms = 10.0`

### trend / local fit

- `trend_local_section_len = 16`
- `trend_local_stride = 4`
- `trend_local_huber_c = 1.345`
- `trend_local_iters = 3`
- `trend_local_vmin_mps = 300.0`
- `trend_local_vmax_mps = 8000.0`
- `trend_sigma_ms = 6.0`
- `trend_min_pts = 12`
- `trend_var_half_win_traces = 8`
- `trend_var_sigma_std_ms = 6.0`
- `trend_var_min_count = 3`

### residual statics / final snap

- `use_residual_statics = true`
- `rs_pre_snap_mode = trough`
- `rs_pre_samples = 20`
- `rs_post_samples = 20`
- `rs_max_lag = 8`
- `rs_k_neighbors = 5`
- `rs_n_iter = 2`
- `rs_c_th = 0.5`
- `use_final_snap = true`
- `final_snap_mode = trough`
- `final_snap_ltcor = 3`

### keep / reject 閾値

- `drop_low_frac = 0.05`
- 使用スコア: `conf_prob1`, `conf_trend1`, `conf_rs1`
- 閾値モード:
  - 学習 artifact 作成時: `global`
  - 単独推論時: `per_segy` を許容

### stage2 的 robust center 形成

- `half_win = 128`
- `local_global_diff_th_samples = 128`
- `local_discard_radius_traces = 32`
- `local_inv_drop_th_samples = 10.0`
- `local_inv_min_consec_steps = 2`
- `global_vmin_m_s = 300.0`
- `global_vmax_m_s = 6000.0`
- `global_side_min_pts = 16`

## 6.4 v0 でやること

### pick table 正規化

`coarse.npz` から少なくとも次の列を持つ table を作る。

- `shot_id`
- `trace_id`
- `ffid`
- `chno`
- `offset_m`
- `dt_sec`
- `coarse_pick_i`
- `coarse_pick_t_sec`
- `coarse_pmax`
- `conf_prob1`
- `conf_trend1`
- `conf_rs1`

### 共通ロジック化

`proc/jogsarar/stage1` / `stage2` のロジックを runtime import せず、
`packages/seisai-pick` または `fbpick/physics/*` へ pure function として移植する。

### robust 出力

`robust.npz` には最低限次を持たせる。

- `robust_pick_i`
- `robust_pick_t_sec`
- `robust_conf`
- `robust_source`
- `used_theoretical_mask`
- `reason_mask`

---

## 7. Fine モデル仕様（固定値）

## 7.1 役割

- `robust_pick_i` 周辺の局所窓から最終 P pick を読む
- 大域探索ではなく、**局所補正専用**とする

## 7.2 入力

- waveform only
- `in_chans = 1`

入力 shape:

```text
(B, 1, 128, 256)
```

## 7.3 局所窓

各 trace の中心 `c = robust_pick_i` に対して、時間方向 window は次で固定する。

```text
[c - 128 : c + 128)
```

- 長さは 256 samples
- local 座標では `128` を基準中心として扱う
- 範囲外は zero padding

## 7.4 trace 方向

- `trace_len = 128`
- `overlap_h = 96`
- stride は `32`

## 7.5 target

- 出力は **1ch heatmap**
- 教師は local 窓内 GT pick を中心にした Gaussian
- `fine_sigma_ms = 3.0`

```text
sigma_samples = fine_sigma_ms / (dt_sec * 1000.0)
sigma_samples = clip(sigma_samples, 1.5, 12.0)
```

## 7.6 学習中心

Fine 学習の center は v0 では **physics 後 robust center ベース**で固定する。
GT center / jitter / coarse offset mixture は将来拡張とする。

## 7.7 Fine 初期化

### 方針

Fine は Coarse checkpoint から初期化する。
ただし `3ch -> 1ch` のため、generic `init_ckpt` の緩和ではなく
**専用変換 utility**を作る。

### 変換ルール

1. Coarse / Fine は同一 backbone を使う
2. state_dict の**最初の conv**だけ waveform channel を切り出す
3. encoder / decoder 重みはそのまま流用する
4. `seg_head.*` は drop して Fine 側で再初期化する
5. generic `maybe_load_init_weights()` は変更しない

```text
fine_first_conv_weight = coarse_first_conv_weight[:, 0:1, :, :]
```

ここで `channel 0 = waveform` と固定する。

### 実装上の写像

変換 utility は、新しい checkpoint を生成して `model_sig` を
`in_chans=1`, `out_chans=1` に書き換える。
その変換後 checkpoint を Fine train の `train.init_ckpt` に渡す。

### 平均化しない理由

Coarse の ch1/ch2 は `offset` / `time` であり waveform と意味が異なるため、
チャンネル平均は採用しない。

---

## 8. Artifact / NPZ 契約

## 8.1 Coarse 中間 artifact

ファイル名例:

```text
<stem>.coarse.npz
```

最低限のキー:

- `dt_sec`
- `n_samples_orig`
- `n_traces`
- `ffid_values`
- `chno_values`
- `offsets_m`
- `trace_indices`
- `coarse_pick_i`
- `coarse_pick_t_sec`
- `coarse_pmax`
- `coarse_prob_summary`
- `lineage`

heatmap 全保存は必須ではない。
保存容量を抑えるため、v0 は `argmax / pmax / optional compressed logits` を許容する。

## 8.2 Robust 中間 artifact

ファイル名例:

```text
<stem>.robust.npz
```

最低限のキー:

- `robust_pick_i`
- `robust_pick_t_sec`
- `robust_conf`
- `robust_source`
- `used_theoretical_mask`
- `reason_mask`
- `lineage`

## 8.3 最終出力ファイル

```text
<stem>.fbpick_final.npz
```

### 必須キー

```text
dt_sec                () float32
n_samples_orig        () int32
n_traces              () int32

ffid_values           (Ntr,) int32
chno_values           (Ntr,) int32
offsets_m             (Ntr,) float32
trace_indices         (Ntr,) int64

coarse_pick_i         (Ntr,) int32
coarse_pmax           (Ntr,) float32

robust_pick_i         (Ntr,) int32
robust_conf           (Ntr,) float32
robust_source         (Ntr,) uint8
used_theoretical_mask (Ntr,) bool
reason_mask           (Ntr,) uint8

window_start_i        (Ntr,) int32
window_end_i          (Ntr,) int32

fine_pick_local_f     (Ntr,) float32
fine_pick_local_i     (Ntr,) int32
fine_pmax             (Ntr,) float32

final_pick_f          (Ntr,) float32
final_pick_i          (Ntr,) int32
final_pick_t_sec      (Ntr,) float32
final_conf            (Ntr,) float32
high_conf_mask        (Ntr,) bool
reject_mask           (Ntr,) bool
```

### `robust_source`

```text
0 = coarse observed
1 = theoretical replacement
2 = trend/global fill
```

### v0 の float pick 規約

sub-sample 補間は v0 では入れない。
したがって `*_pick_f` は **整数 argmax を float32 にキャストした値**でよい。

### `final_conf` の v0 既定

明示的な別仕様が入るまでは、v0 では次を既定とする。

```text
final_conf = min(robust_conf, fine_pmax)
```

これは fail-safe な conservative rule であり、あとで差し替えやすい。

### `window_start_i`, `window_end_i`

```text
window_start_i = robust_pick_i - 128
window_end_i   = robust_pick_i + 127
```

### lineage

各 NPZ には少なくとも次を保持する。

- `iter_id`
- `source_model_id`
- `cfg_hash`
- `git_sha`

`proc/jogsarar/common/lineage.py` は参考にしてよいが、runtime 依存は作らないこと。

---

## 9. 可視化仕様

標準可視化は gather overlay とする。

### main overlay

raw waveform gather 上に次を重ねる。

- `coarse_pick_i`: 薄いシアン
- `robust_pick_i`: 黄色
- `window_start_i`: 黄色破線
- `window_end_i`: 黄色破線
- `final_pick_i`: 赤
- `high_conf final pick`: 緑強調

### 出力ファイル名例

```text
<stem>.overview.png
<stem>.coarse_conf.png
<stem>.ffidXXXX.png
```

### 実装参考

- `proc/jogsarar/jogsarar_viz/stage1_gather.py`
- `proc/jogsarar/jogsarar_viz/stage4_gather.py`

ただし v0 では **robust window 境界線**を追加する。

---

## 10. 追加・変更するファイル

以下を v0 の実装対象とする。
この粒度より大きく逸脱する構成変更は避ける。

## 10.1 dataset 側

### 既存ファイルを変更

- `packages/seisai-dataset/src/seisai_dataset/builder/builder.py`
- `packages/seisai-dataset/src/seisai_dataset/builder/__init__.py`
- `packages/seisai-dataset/src/seisai_dataset/infer_window_dataset.py`
- `packages/seisai-dataset/src/seisai_dataset/__init__.py`

### 新規追加

- `packages/seisai-dataset/src/seisai_dataset/local_window_dataset.py`

### 追加する build op

`builder.py` に次を追加する。

- `NormalizeTimeByConst`
- `NormalizeOffsetByConst`
- `FBGaussMapMs`

#### `NormalizeTimeByConst`

想定 signature:

```python
NormalizeTimeByConst(
    src='time_ch_raw',
    dst='time_ch',
    ref_sec=T_ref_sec,
    clip_lo=0.0,
    clip_hi=1.5,
)
```

#### `NormalizeOffsetByConst`

想定 signature:

```python
NormalizeOffsetByConst(
    src='offset_ch_raw',
    dst='offset_ch',
    ref_m=O_ref_m,
    use_abs=True,
    clip_lo=0.0,
    clip_hi=1.5,
)
```

#### `FBGaussMapMs`

想定 signature:

```python
FBGaussMapMs(
    dst='y_fb_map',
    src='fb_idx_view',
    sigma_ms=10.0,
    sigma_samples_min=4.0,
    sigma_samples_max=40.0,
)
```

実装要件:

- `meta['dt_eff_sec']` があればそれを使う
- なければ `meta['dt_sec']` を使う
- valid trace だけ Gaussian を入れる
- invalid / out-of-view trace はゼロ行列
- 返す shape は既存 `FBGaussMap` と同じ `(H, W)`

### `InferenceGatherWindowsDataset` の変更方針

`fb_files: Sequence[str] | None = None` を受けられるようにする。
raw-only inference では次とする。

- `fb_idx_view` は **全 trace -1** を基本とする
- `trace_valid` は実 trace だけ True
- 中央 crop はしない
- `W < target_len` のときだけ右 pad

既存の label 付き挙動は壊さない。

### `LocalWindowDataset`

新規 dataset は Fine train / Fine infer で共用する。

入力:

- `segy_files`
- `fb_files`（train 時のみ必須）
- `robust_npz_files`
- shot domain 固定

要件:

- H window は `128`, stride `32`
- 各 trace の局所時間窓中心は `robust_pick_i`
- local 時間長は `256`
- local meta に `fb_idx_view` を **local index** で持たせる
- raw index / trace index / center / `window_start_i` / `window_end_i` を持たせる
- train で local 窓内に GT が入らない trace は `label_valid=False` または reject 方針を明示する

v0 では、Fine train sample は **GT が local window 内に入るものだけ採用**でよい。
外れる sample は reject して再サンプルする。

## 10.2 engine 側

新規 package:

```text
packages/seisai-engine/src/seisai_engine/pipelines/fbpick/
  __init__.py
  common/
    __init__.py
    artifacts.py
    io.py
    ref_stats.py
    viz_overlay.py
    ckpt_convert.py
  coarse/
    __init__.py
    config.py
    build_plan.py
    build_dataset.py
    build_model.py
    loss.py
    train.py
    infer.py
  physics/
    __init__.py
    config.py
    pick_table.py
    feasible.py
    trend.py
    merge.py
    run.py
  fine/
    __init__.py
    config.py
    build_plan.py
    build_dataset.py
    build_model.py
    loss.py
    train.py
    infer.py
```

### `common/artifacts.py`

- NPZ key 定義の唯一の正本
- `robust_source` enum
- `reason_mask` enum
- lineage payload helper

### `common/io.py`

- `save_coarse_npz()` / `load_coarse_npz()`
- `save_robust_npz()` / `load_robust_npz()`
- `save_final_npz()` / `load_final_npz()`

### `common/ref_stats.py`

- training corpus から `T_ref_sec`, `O_ref_m` を計算する helper
- p99 を返す
- 出力は JSON / YAML のどちらでもよいが 1 形式に固定する

### `common/ckpt_convert.py`

- Coarse checkpoint を Fine 用 init checkpoint に変換する pure utility
- generic init loader は変更しない

### `common/viz_overlay.py`

- 最終 overlay PNG 出力
- coarse / robust / fine / high_conf を重ねる

### `coarse/build_plan.py`

BuildPlan は次を基本にする。

1. `IdentitySignal(src='x_view', dst='x_wave', copy=False)`
2. `MakeOffsetChannel(dst='offset_ch_raw', normalize=False)`
3. `NormalizeOffsetByConst(...)`
4. `MakeTimeChannel(dst='time_ch_raw')`
5. `NormalizeTimeByConst(...)`
6. `FBGaussMapMs(dst='y_fb_map', sigma_ms=10.0, ...)`
7. `SelectStack(keys=['x_wave', 'offset_ch', 'time_ch'], dst='input', ...)`
8. `SelectStack(keys=['y_fb_map'], dst='target', ...)`

### `coarse/build_dataset.py`

- labeled coarse train dataset builder
- infer 用 raw-only dataset builder
- pick-aware `SampleTransformer` をここで定義または隣接 module に切り出す
- `SegyGatherPipelineDataset` は coarse train に使う
- `InferenceGatherWindowsDataset` は coarse infer に使う

### `coarse/loss.py`

- `FbSegKLLossView` を薄く包むだけでよい
- bespoke loss は作らない

### `coarse/infer.py`

- raw survey 1 本を入力にする
- H window を列挙し、各 window で W tile 推論する
- H overlap は raw trace 軸へ平均集約する
- 一時的に `(n_traces, n_samples_orig)` の accumulation buffer を使ってよい
- 最終的に `coarse_pick_i`, `coarse_pmax` を作り `coarse.npz` 保存
- full heatmap は保存必須ではない

### `physics/run.py`

- `coarse.npz` 読み込み
- pick table 化
- feasible band / trend / confidence / keep / reject / fill
- `robust.npz` 保存

### `fine/build_plan.py`

BuildPlan は次を基本にする。

1. `IdentitySignal(src='x_view', dst='x_wave', copy=False)`
2. `FBGaussMapMs(dst='y_fb_map', sigma_ms=3.0, ...)`
3. `SelectStack(keys=['x_wave'], dst='input', ...)`
4. `SelectStack(keys=['y_fb_map'], dst='target', ...)`

### `fine/build_dataset.py`

- `LocalWindowDataset` を使う
- train / infer 双方で H window 列挙を共通化する
- infer では GT なしの local dataset を組めるようにする

### `fine/infer.py`

- `robust.npz` を読む
- local windows を推論
- H overlap は trace ごとに local logits を平均集約する
- `fine_pick_local_i`, `fine_pmax` を作る
- raw 軸へ戻して `final_pick_i`, `final_pick_t_sec` を作る
- `fbpick_final.npz` と overlay PNG を保存する

## 10.3 CLI 追加

```text
cli/run_fbpick_fit_refs.py
cli/run_fbpick_coarse_train.py
cli/run_fbpick_coarse_infer.py
cli/run_fbpick_physics.py
cli/run_fbpick_fine_train.py
cli/run_fbpick_fine_infer.py
```

既存 `cli/run_psn_train.py` と同じ thin entrypoint 形式にする。

## 10.4 examples 追加

```text
examples/config_fit_fbpick_refs.yaml
examples/config_train_fbpick_coarse.yaml
examples/config_infer_fbpick_coarse.yaml
examples/config_run_fbpick_physics.yaml
examples/config_train_fbpick_fine.yaml
examples/config_infer_fbpick_fine.yaml
```

既存の YAML セクション名に寄せること。
新しい config system を別建てしない。

---

## 11. Config 方針

## 11.1 共通方針

既存 pipeline と同じく、基本は次の top-level セクションを使う。

- `paths`
- `dataset`
- `train`
- `transform`
- `infer`
- `vis`
- `ckpt`
- `model`
- `tracking`

## 11.2 Coarse train で新たに必要な値

- `train.fb_sigma_ms: 10.0`
- `transform.time_len: 6016`
- `model.in_chans: 3`
- `model.out_chans: 1`
- `norm_refs.time_ref_sec`
- `norm_refs.offset_ref_m`

## 11.3 Fine train で新たに必要な値

- `train.fb_sigma_ms: 3.0`
- `transform.time_len: 256`
- `model.in_chans: 1`
- `model.out_chans: 1`
- `paths.robust_npz_files`
- `train.init_ckpt` には **変換済み coarse ckpt** を渡す

## 11.4 infer config の方針

- coarse infer は `tile_w=6016` と `overlap_h=96` を持つ
- Fine infer は local window 固定なので `time_len=256` を持つ
- `paths.segy_files` は inference 対象 1 survey 分を受ける
- 出力先は `paths.out_dir`

## 11.5 ref stats helper

`run_fbpick_fit_refs.py` は training corpus から次を出す。

- `time_ref_sec = p99(time_abs_sec)`
- `offset_ref_m = p99(abs(offset_m))`

これを coarse / fine config から読み込めるようにする。

---

## 12. 学習フロー

## 12.1 Coarse 学習

1. raw SGY + GT pick を読む
2. trace 方向 `128` の subset を引く
3. `W <= 6016` は pad、`W > 6016` は pick-aware crop を使う
4. waveform を標準化する
5. absolute time / absolute offset channel を追加する
6. `FBGaussMapMs(sigma_ms=10.0)` で target を作る
7. `EncDec2D(backbone=caformer_s18, in_chans=3, out_chans=1)` を学習する

## 12.2 Fine 学習

1. raw SGY + robust center + GT pick を読む
2. `128 x 256` の局所窓を切る
3. waveform のみを入力にする
4. `FBGaussMapMs(sigma_ms=3.0)` で local target を作る
5. Fine 用変換済み ckpt で初期化する
6. `EncDec2D(backbone=caformer_s18, in_chans=1, out_chans=1)` を学習する

---

## 13. 推論フロー

## 13.1 Coarse 推論

1. 1 survey 分の raw SGY を入力する
2. H window を列挙する
3. 各 window で W tile 推論する
4. overlap を raw trace 軸に平均集約する
5. `coarse_pick_i` と `coarse_pmax` を作る
6. `coarse.npz` を保存する

## 13.2 Physics / robustification

1. `coarse.npz` を読む
2. feasible band / trend / confidence を計算する
3. keep / reject / fill を決める
4. 必要に応じて theoretical / trend で置換する
5. `robust.npz` を保存する

## 13.3 Fine 推論

1. `robust.npz` を読む
2. `robust_pick_i` を中心に local windows を作る
3. Fine heatmap を推論する
4. overlap を trace 軸で平均集約する
5. local argmax を raw sample index に戻す
6. `final_pick_i`, `final_pick_t_sec`, `final_conf` を作る
7. `fbpick_final.npz` と overlay PNG を保存する

---

## 14. 実装順序（Codex の作業単位）

この順番で実装すると戻りが少ない。

### Step 1: dataset 基盤を固める

- `NormalizeTimeByConst`
- `NormalizeOffsetByConst`
- `FBGaussMapMs`
- `InferenceGatherWindowsDataset(fb_files=None)` 対応
- dataset unit tests

### Step 2: Coarse train/infer を通す

- `fbpick/coarse/*`
- YAML loader
- pick-aware sample transformer
- `coarse.npz` writer
- coarse smoke test

### Step 3: physics-lite を通す

- pick table
- feasible band / trend / confidence の pure function 化
- `robust.npz` writer
- robust unit tests

### Step 4: Fine 用データセットと ckpt 変換

- `LocalWindowDataset`
- `ckpt_convert.py`
- Fine build_plan / build_dataset
- unit tests

### Step 5: Fine train/infer と final writer

- `fbpick/fine/*`
- final npz writer
- overlay viz
- final smoke test

### Step 6: CLI / example / docs 整備

- thin CLI 追加
- example YAML 追加
- 必要なら README への最小追記

---

## 15. テスト要件

最低限、次のテストを追加する。

### dataset tests

- `FBGaussMapMs` が `dt_eff_sec` を優先すること
- `FBGaussMapMs` の sigma clamp が効くこと
- `NormalizeTimeByConst` / `NormalizeOffsetByConst` の clip が正しいこと
- `InferenceGatherWindowsDataset` が `fb_files=None` で raw-only inference できること
- `LocalWindowDataset` が local `fb_idx_view` を正しく返すこと

### engine tests

- coarse build_plan が `input:(3,H,W)` / `target:(1,H,W)` を返すこと
- fine build_plan が `input:(1,H,W)` / `target:(1,H,W)` を返すこと
- `ckpt_convert.py` が first conv を 1ch に切り出せること
- `coarse.npz` / `robust.npz` / `final.npz` の key 契約が一致すること

### smoke / e2e

- synthetic small dataで coarse train 1 epoch が通ること
- synthetic small dataで coarse infer → physics → fine infer が通ること

---

## 16. 実装上の禁止事項

- `proc/jogsarar` を library runtime import しない
- generic `maybe_load_init_weights()` を緩めない
- PSN pipeline に first-break 特有の分岐を混ぜ込まない
- `global_qc` package/CLI を先に作らない
- survey ごとの別 head / 別 model を作らない
- `dt` resampling を導入しない
- inference で raw gather を中央 crop しない
- fallback を暗黙に入れない

---

## 17. v0 で固定する最終一覧

### Coarse

- domain: `shot`
- input: `waveform + abs(offset_m) + absolute time`
- input channels: `3`
- window: `128 x 6016`
- overlap_h: `96`
- target: `P-only Gaussian heatmap`
- `sigma_ms = 10.0`
- `sigma_samples clamp = [4.0, 40.0]`
- backbone: `caformer_s18`
- baseline backbone: `convnext_small`

### Fine

- input: `waveform only`
- input channels: `1`
- window: `128 x 256`
- overlap_h: `96`
- target: `GT-centered local Gaussian heatmap`
- `sigma_ms = 3.0`
- `sigma_samples clamp = [1.5, 12.0]`
- backbone: `caformer_s18`
- init: `coarse waveform-channel transfer + seg_head reset`

### Physics

- feasible band: `vmin=100`, `vmax=5000`, `t0_lo=-10ms`, `t0_hi=100ms`
- threshold drop fraction: `0.05`
- trend defaults: `section_len=16`, `stride=4`, `huber_c=1.345`, `iters=3`
- local window half width: `128`

### Output

- final format: `NPZ`
- main final file: `<stem>.fbpick_final.npz`
- standard visualization: `gather overlay PNG`

---

## 18. 判断まとめ

今回の `fbpick` v0 は次の思想で固定した。

- **複数 survey を 1 モデル群で扱う**
- **リサンプリングせず、物理座標を明示的に入力する**
- **Coarse で大域探索し、Fine で局所補正する**
- **Physics は独立モジュールとして責務分離する**
- **repo 既存資産は最大限使うが、足りない部分だけ最小拡張する**
