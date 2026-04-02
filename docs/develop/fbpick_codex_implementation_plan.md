# SeisAI `fbpick` 実装設計書（Codex 向け）

## 0. 文書の目的

本書は、SeisAI モノレポ上に **反射法地震探査の初動（first-break / P波初動）自動読み取り pipeline** を正式実装するための、Codex 向け実装指示書である。

目的は次の 3 点。

1. **既存資産を最大流用**して新規 pipeline を追加する。
2. **coarse → fine → global QC** の 3 段構成を、SeisAI の既存流儀に沿って配置する。
3. 最終的に **1 調査の SEG-Y を渡すと first-break を返す** 入口までを用意する。

本書は設計と実装境界を明示する。**一般論ではなく、この repo の現状ファイル構成に対する具体指示**とする。

---

## 1. 前提と基本方針

### 1.1 前提

- 作業場所は **SeisAI モノレポ本体**とする。
- 新規実装は monorepo の既存 package 構成に合わせる。
- 実行環境は repo 直下での **editable install** を前提とする。
- `proc/jogsarar` は **参照元**として扱い、**本体の置き場にはしない**。

### 1.2 実装ポリシー

以下を厳守すること。

- **既存コードを最大流用**する。重複実装を避ける。
- 想定可能なエラーは **即時失敗**とし、暗黙フォールバックは禁止。
- 後方互換は考慮しない。今回の新規 pipeline を明示的に追加する。
- `psn` を無理に拡張して first-break を押し込まない。**`fbpick` を新設**する。
- `coarse` は 3ch 入力（`amplitude / offset / time`）を基本とする。
- `fine` 初版は **amplitude-only の 1ch local window 入力**とする。
- `global QC` は最初は **学習内蔵ではなく推論後段**として実装する。
- `fine` は回帰ではなく **確率分布出力**とする。

### 1.3 非目標

初版では次を必須にしない。

- end-to-end で inversion に backprop する学習
- static correction の完全自動化
- `seisai-models` への専用 backbone 追加
- 既存 `viewer/fbpick.py` の全面刷新

---

## 2. 既存資産の参照元（実ファイル）

Codex は次の既存実装を必ず参照してから書くこと。

### 2.1 `seisai-engine` 側

- `packages/seisai-engine/src/seisai_engine/pipelines/psn/config.py`
- `packages/seisai-engine/src/seisai_engine/pipelines/psn/build_plan.py`
- `packages/seisai-engine/src/seisai_engine/pipelines/psn/build_dataset.py`
- `packages/seisai-engine/src/seisai_engine/pipelines/psn/build_model.py`
- `packages/seisai-engine/src/seisai_engine/pipelines/psn/loss.py`
- `packages/seisai-engine/src/seisai_engine/pipelines/psn/train.py`
- `packages/seisai-engine/src/seisai_engine/pipelines/blindtrace/build_plan.py`
- `packages/seisai-engine/src/seisai_engine/pipelines/common/encdec2d_model.py`
- `packages/seisai-engine/src/seisai_engine/pipelines/common/config_loaders.py`
- `packages/seisai-engine/src/seisai_engine/pipelines/common/tiled_infer.py`

### 2.2 `seisai-dataset` 側

- `packages/seisai-dataset/src/seisai_dataset/builder/builder.py`
- `packages/seisai-dataset/src/seisai_dataset/infer_window_dataset.py`
- `packages/seisai-dataset/src/seisai_dataset/segy_gather_pipeline_dataset.py`
- `packages/seisai-dataset/src/seisai_dataset/segy_gather_phase_pipeline_dataset.py`
- `packages/seisai-dataset/src/seisai_dataset/target_fb.py`
- `packages/seisai-dataset/src/seisai_dataset/gate_fblc.py`

### 2.3 `seisai-pick` 側

- `packages/seisai-pick/src/seisai_pick/residual_statics.py`
- `packages/seisai-pick/src/seisai_pick/score/confidence_from_prob.py`
- `packages/seisai-pick/src/seisai_pick/score/confidence_from_residual_statics.py`
- `packages/seisai-pick/src/seisai_pick/score/confidence_from_trend_resid.py`
- `packages/seisai-pick/src/seisai_pick/trend/trend_fit.py`
- `packages/seisai-pick/src/seisai_pick/trend/gaussian_prior_from_trend.py`

### 2.4 examples / `proc/jogsarar`

- `examples/example_train_fbp.py`
  - 3ch 入力（`MakeOffsetChannel`, `MakeTimeChannel`）
  - first-break target (`FBGaussMap`)
  - tiled inference 例
- `proc/jogsarar/common/NPZ_CONTRACT.md`
  - stage 間 artifact 契約の考え方
- `proc/jogsarar/stage2/*`
  - coarse seed から local window を切る責務分離
- `proc/jogsarar/stage4/*`
  - local 推論結果を raw 軸に戻す責務分離

---

## 3. 新規に追加するディレクトリとファイル

以下を新設する。

```text
packages/
  seisai-engine/
    src/seisai_engine/
      pipelines/
        fbpick/
          __init__.py
          common/
            __init__.py
            config.py
            artifacts.py
            io.py
            vis.py
          coarse/
            __init__.py
            config.py
            build_plan.py
            build_dataset.py
            build_model.py
            loss.py
            infer.py
            infer_segy2npz.py
            train.py
          fine/
            __init__.py
            config.py
            build_plan.py
            build_dataset.py
            build_window_dataset.py
            build_model.py
            loss.py
            infer.py
            infer_from_coarse.py
            train.py
          global_qc/
            __init__.py
            config.py
            build_candidates.py
            run.py
            export.py

  seisai-dataset/
    src/seisai_dataset/
      local_window_dataset.py
      builder/
        fb_local_ops.py

  seisai-pick/
    src/seisai_pick/
      global_qc/
        __init__.py
        geometry.py
        inversion_adapter.py
        arrival_band.py
        consistency.py
        confidence.py
        repick.py

cli/
  run_fbpick_coarse_train.py
  run_fbpick_coarse_infer.py
  run_fbpick_fine_train.py
  run_fbpick_fine_infer.py
  run_fbpick_global_qc.py

examples/
  config_train_fbpick_coarse.yaml
  config_infer_fbpick_coarse.yaml
  config_train_fbpick_fine.yaml
  config_infer_fbpick_fine.yaml
  config_fbpick_global_qc.yaml
```

---

## 4. アーキテクチャ概要

### 4.1 coarse

目的:

- survey 全体の gather から、**first-break 候補帯を外さない**粗い初動確率を出す。
- fine 用 local window の中心 seed を供給する。

入出力:

- 入力: `amplitude + offset + absolute time` の 3ch
- 出力: trace ごとの **coarse first-break probability**
- 派生出力: `pick_idx`, `confidence`, `valid mask`

### 4.2 fine

目的:

- coarse seed 周辺の局所窓を入力にして、**局所窓内の精密な確率分布**を出す。

入出力:

- 入力: `amplitude` の 1ch local window（初版）
- 出力: local window 内の **fine probability**
- 派生出力: `local_pick_idx`, `raw_pick_idx`, `fine_confidence`

### 4.3 global QC

目的:

- coarse/fine の pick 候補を、**3D 幾何・速度モデル・inversion 結果で再評価**する。
- reject / keep / adjust の判断を行う。

入出力:

- 入力: coarse/fine pick 候補 + geometry + optional inversion result
- 出力: `pick_global`, `confidence_global`, `reject_flag`

---

## 5. ファイル別責務一覧

## 5.1 `packages/seisai-engine/src/seisai_engine/pipelines/fbpick/common/`

### `__init__.py`
- `fbpick.common` で外に出す最小 API を export する。

### `config.py`
- stage 共通の小さい config dataclass を置く。
- 例:
  - artifact path 共通設定
  - pick 閾値共通設定
  - time window 共通設定
- 各 stage 固有 config はここに入れない。

### `artifacts.py`
- stage 間の artifact 契約を dataclass / constant key で定義する。
- 少なくとも次の構造を持つ。
  - `CoarseResult`
  - `FineResult`
  - `GlobalQcResult`
- NPZ キー名はここを唯一の定義元にする。

### `io.py`
- `artifacts.py` で定義した NPZ / JSON の read / write を行う。
- `save_*`, `load_*` をここに集約する。
- path 命名規約をここに固定する。

### `vis.py`
- gather + pick + probability を可視化するデバッグ関数を置く。
- matplotlib 依存の軽い関数にとどめる。

---

## 5.2 `packages/seisai-engine/src/seisai_engine/pipelines/fbpick/coarse/`

### `__init__.py`
- coarse pipeline の public API export。

### `config.py`
- coarse train / infer 用 dataclass と loader を実装する。
- `psn/config.py` をベースにしつつ次を変更する。
  - `model.in_chans == 3`
  - `model.out_chans == 1`
- 追加候補項目:
  - `train.subset_traces`
  - `train.use_label_valid_mask`
  - `infer.subset_traces`
  - `infer.use_offset_ch`
  - `infer.use_time_ch`
  - `target.psn_sigma` 相当の coarse sigma

### `build_plan.py`
- coarse 用 `BuildPlan` を作る。
- 必須構成:
  - amplitude: `x_view`
  - optional offset channel: `MakeOffsetChannel`
  - optional time channel: `MakeTimeChannel`
  - target: `FBGaussMap` もしくは等価の first-break 確率 target
- `blindtrace/build_plan.py` と `example_train_fbp.py` を流用する。

### `build_dataset.py`
- coarse train / eval 用 dataset を構築する。
- 第一候補は `SegyGatherPipelineDataset`。
- phase-pick CSR を直接使う必要がある場合のみ `SegyGatherPhasePipelineDataset` を使う。
- `fb gate` は既存 `gate_fblc.py` を優先活用する。

### `build_model.py`
- `EncDec2D` を使って coarse model を返す。
- 最初は backbone 追加をしない。
- `in_chans=3`, `out_chans=1` を前提にする。

### `loss.py`
- coarse probability に対する loss を定義する。
- 初版候補:
  - KL 系
  - BCE 系
- `example_train_fbp.py` の `FbSegKLLossView` と既存 loss 流儀を参考にする。

### `infer.py`
- coarse モデル推論 core を実装する。
- tiled inference は既存 `tiled_infer.py` を使用する。
- 返すもの:
  - `prob`
  - `pick_idx`
  - `confidence`
  - meta

### `infer_segy2npz.py`
- `SEG-Y -> coarse artifact` を行う stage 実行ファイル。
- `cli/run_fbpick_coarse_infer.py` から呼ばれる入口にする。
- 出力先は `fbpick/common/io.py` に従う。

### `train.py`
- coarse training の main 入口。
- `psn/train.py` の skeleton を流用する。
- config から dataset / model / loss / optimizer / ckpt を立ち上げる。

---

## 5.3 `packages/seisai-engine/src/seisai_engine/pipelines/fbpick/fine/`

### `__init__.py`
- fine pipeline の public API export。

### `config.py`
- fine train / infer config を定義する。
- 初版は amplitude-only 1ch を固定とする。
- 必須設定例:
  - `window.half_width_samples` または ms ベース幅
  - `window.target_len`
  - `coarse_artifact_root`
- 互換性のため `use_offset_channel` / `use_relative_time_channel` を schema に残してもよいが、初版では **必ず false** を検証する。

### `build_plan.py`
- fine 用 `BuildPlan` を作る。
- 初版入力は `amplitude` の 1ch local window。
- target は **local window 内の first-break probability**。

### `build_dataset.py`
- fine 学習用 dataset を構築する。
- GT pick を持つ raw gather から local window を作る訓練モードを担当する。

### `build_window_dataset.py`
- fine 推論用 local window dataset の構築を担当する。
- `coarse artifact` を seed にして raw gather から局所窓を切る。
- raw index と local index の対応を返す。
- `proc/jogsarar/stage2` の正式版に相当する。

### `build_model.py`
- `EncDec2D` を使って fine model を返す。
- 初版は `in_chans=1`, `out_chans=1`。

### `loss.py`
- local probability 用 loss を定義する。
- coarse 用と分けてよい。
- local window の sharpness を見たいので、loss 重み付けを切れるようにしておく。

### `infer.py`
- local window 上の fine 推論 core。
- `local_prob`, `local_pick_idx` を返す。

### `infer_from_coarse.py`
- coarse artifact を読み、local window を作り、fine 推論し、raw pick に戻す stage 実行ファイル。
- `cli/run_fbpick_fine_infer.py` の入口。
- `proc/jogsarar/stage4` の正式版に相当する。

### `train.py`
- fine training の main 入口。
- coarse と同様に train skeleton を流用する。

---

## 5.4 `packages/seisai-engine/src/seisai_engine/pipelines/fbpick/global_qc/`

### `__init__.py`
- global QC stage の export。

### `config.py`
- global QC 用 config を定義する。
- 必須設定例:
  - geometry path
  - inversion backend 種別
  - residual 許容範囲
  - arrival band 幅
  - reweight 係数
  - reject rule

### `build_candidates.py`
- coarse / fine artifact を読み、global QC に渡す候補集合を作る。
- 候補は少なくとも次を含む。
  - `pick_raw`
  - `probability around pick`
  - `confidence`
  - `trace index`
  - `ffid`
  - `offset`
  - geometry に紐づくキー

### `run.py`
- global QC の主実行ファイル。
- 流れ:
  1. candidate 構築
  2. geometry 読み込み
  3. inversion 呼び出し
  4. arrival band 構築
  5. probability reweight
  6. repick
  7. confidence 再計算
  8. export

### `export.py`
- final pick を外へ出す。
- 初版では NPZ / CSV まででよい。
- 将来 static correction に渡すための sidecar を設計しやすい形にする。

---

## 5.5 `packages/seisai-dataset/src/seisai_dataset/builder/fb_local_ops.py`

このファイルには、BuildPlan から再利用する **first-break 局所窓専用 op** を置く。

最低限追加するもの:

- `FBLocalGaussMap`
  - local window 内の pick index から局所 probability target を生成する。

将来拡張候補:

- `MakeRelativeTimeChannel`
  - local window の中心を 0 とする relative time channel を生成する。
- `NormalizeOffsetChannel`
  - fine を多チャンネル化するときに offset 正規化が必要なら追加する。

ルール:

- 汎用な builder op として書く。
- engine 側に builder op を書かない。

---

## 5.6 `packages/seisai-dataset/src/seisai_dataset/local_window_dataset.py`

責務:

- raw SEG-Y と coarse pick seed から **local window を列挙**する。
- 学習・推論の両方で使えるようにする。

最低限の機能:

- `gt_pick` から local window を作る training mode
- `coarse_pick` から local window を作る inference mode
- `window_start_i`, `window_end_i`, `seed_i`, `raw_pick_i`, `local_pick_i` の対応を返す
- variable W を扱うなら既存 collate の流儀に従う

---

## 5.7 `packages/seisai-pick/src/seisai_pick/global_qc/`

ここは **物理・幾何・pick 数学の純粋ロジック層**に限定する。

### `__init__.py`
- export のみ。

### `geometry.py`
- shot / receiver / elevation などの geometry を扱う。
- survey ごとの座標系差を吸収する最小ヘルパを置く。

### `inversion_adapter.py`
- inversion backend との I/O 境界。
- 初版は stub / protocol でもよい。
- engine 側から backend 実装を隠す。

### `arrival_band.py`
- inversion 結果または速度モデルから expected arrival band を作る。
- trace ごとの center と width を返せるようにする。

### `consistency.py`
- 3D 整合性スコアの計算。
- 近傍 shot / receiver と矛盾する pick を検出する純関数群を置く。

### `confidence.py`
- global QC 後の confidence を再計算する。
- 既存 `score/*` を組み合わせて書く。

### `repick.py`
- probability と arrival band を融合して final repick する。
- 最低限 `argmax(prob * prior)` 系の安全な実装を用意する。

---

## 5.8 `cli/`

CLI は **薄く保つ**。ロジックを書かない。

### 追加するファイル

- `cli/run_fbpick_coarse_train.py`
- `cli/run_fbpick_coarse_infer.py`
- `cli/run_fbpick_fine_train.py`
- `cli/run_fbpick_fine_infer.py`
- `cli/run_fbpick_global_qc.py`

各責務:

- `argparse` で `--config` を受ける
- 対応する `main()` を呼ぶだけ
- 既存 `run_psn_train.py`, `run_psn_infer.py` の薄さを踏襲する

---

## 5.9 `examples/`

### `config_train_fbpick_coarse.yaml`
- coarse 学習の最小例。
- 3ch 入力の設定を明示する。

### `config_infer_fbpick_coarse.yaml`
- coarse 推論の最小例。
- tile 設定を含める。

### `config_train_fbpick_fine.yaml`
- fine 学習の最小例。
- amplitude-only 1ch local window と local window 設定を含める。

### `config_infer_fbpick_fine.yaml`
- coarse artifact を入力にする fine 推論の最小例。
- amplitude-only 1ch local window を前提にする。

### `config_fbpick_global_qc.yaml`
- global QC の最小例。
- geometry / inversion / export を含める。

---

## 6. stage 間 artifact 契約

`proc/jogsarar/common/NPZ_CONTRACT.md` の思想を踏襲し、`fbpick/common/artifacts.py` を唯一の定義元にする。

初版で最低限必要な artifact は以下。

### 6.1 coarse artifact

必須キー:

- `prob`: `(n_traces, n_samples)` float32
- `pick_idx`: `(n_traces,)` int32
- `confidence`: `(n_traces,)` float32
- `trace_indices`: `(n_traces,)`
- `ffid_values`: `(n_traces,)`
- `offsets`: `(n_traces,)`
- `dt_sec`: scalar
- `n_samples_orig`: scalar

### 6.2 fine artifact

必須キー:

- `local_prob`: `(n_traces, w_local)` float32 または window ごとのまとまり
- `local_pick_idx`: `(n_traces,)` int32
- `raw_pick_idx`: `(n_traces,)` int32
- `window_start_i`: `(n_traces,)` int64
- `window_len`: scalar
- `fine_confidence`: `(n_traces,)` float32

### 6.3 global QC artifact

必須キー:

- `pick_global`: `(n_traces,)` int32
- `confidence_global`: `(n_traces,)` float32
- `reject_flag`: `(n_traces,)` bool
- `qc_score`: `(n_traces,)` float32
- `arrival_band_center`: `(n_traces,)` float32
- `arrival_band_width`: `(n_traces,)` float32

ルール:

- 実際の NPZ キー名は `artifacts.py` で定義し、コード中に直書きしない。
- 保存と読込は必ず `common/io.py` 経由にする。

---

## 7. config 設計方針

### 7.1 coarse

最低限の model 制約:

- `model.in_chans == 3`
- `model.out_chans == 1`

### 7.2 fine

最低限の model 制約:

- `model.in_chans == 1`
- `model.out_chans == 1`

### 7.3 global QC

- inversion backend の種別を文字列で指定可能にする。
- backend 未実装時は **即時エラー**にする。暗黙 noop にしない。

### 7.4 共通

- `paths.out_dir` の解決は既存 loader の流儀に合わせる。
- 相対パスの基準は既存 YAML loader に合わせる。

---

## 8. 既存コードからの流用方針

## 8.1 絶対に流用すべきもの

- `psn/train.py` の train skeleton
- `psn/config.py` の config loader パターン
- `blindtrace/build_plan.py` の 3ch 入力組み立てパターン
- `example_train_fbp.py` の first-break target と tile 推論発想
- `infer_window_dataset.py` の決定論 dataset 設計
- `residual_statics.py` と `score/*` の confidence 系ロジック

## 8.2 そのまま流用してはいけないもの

- `psn` の `out_chans == 3 (P/S/Noise)` 制約
- `proc/jogsarar` の固定窓長 / 固定 pad / 専用 path 前提
- stage 間 artifact の ad-hoc key 直書き

---

## 9. 実装順序

Codex は以下の順に PR を刻むこと。**一度に全部書かない**。

### PR-1: `fbpick/coarse` の最小実装

対象:

- `fbpick/common/*`
- `fbpick/coarse/*`
- `cli/run_fbpick_coarse_train.py`
- `cli/run_fbpick_coarse_infer.py`
- `examples/config_train_fbpick_coarse.yaml`
- `examples/config_infer_fbpick_coarse.yaml`

受け入れ条件:

- 学習 config が読み込める
- `in_chans=3`, `out_chans=1` が検証される
- 1 本の SEG-Y に対して coarse artifact NPZ が吐ける

### PR-2: local window dataset と amplitude-only fine の最小実装

対象:

- `seisai_dataset/builder/fb_local_ops.py`
- `seisai_dataset/local_window_dataset.py`
- `fbpick/fine/*`
- `cli/run_fbpick_fine_train.py`
- `cli/run_fbpick_fine_infer.py`
- `examples/config_train_fbpick_fine.yaml`
- `examples/config_infer_fbpick_fine.yaml`

受け入れ条件:

- coarse artifact から local window を作れる
- amplitude-only 1ch fine config が読み込める
- `in_chans=1`, `out_chans=1` が検証される
- fine 推論後に raw 軸の pick を返せる
- fine artifact が保存される

### PR-3: `seisai-pick/global_qc` と `fbpick/global_qc`

対象:

- `seisai_pick/global_qc/*`
- `fbpick/global_qc/*`
- `cli/run_fbpick_global_qc.py`
- `examples/config_fbpick_global_qc.yaml`

受け入れ条件:

- coarse/fine artifact を読み global QC を実行できる
- backend 未実装時に明示エラーで止まる
- final NPZ / CSV が出る

### PR-4: tests と docs 強化

対象:

- unit test
- smoke test
- docs 更新

---

## 10. テスト方針

最低限、次を追加すること。

### 10.1 unit test

- config loader の型検証
- `model.in_chans/out_chans` 制約
- artifact save/load round-trip
- local window raw/local index 変換
- global QC backend 未設定時の即時失敗

### 10.2 smoke test

- coarse train config が起動する
- coarse infer が 1 ファイルで走る
- fine infer が coarse artifact を読める
- global QC が artifact を読んで出力を書く

### 10.3 推奨テスト配置

- `tests/` 配下に pipeline 別で追加する
- `proc/jogsarar/tests` の流儀は参考にしてよいが、本体 tests は `tests/` に置く

---

## 11. 実装時の禁止事項

- `psn` の既存挙動を壊して first-break 対応を混ぜる
- builder op を engine 側に実装する
- artifact key をコード中にベタ書きする
- 失敗時に silent fallback する
- geometry/inversion 未設定時に適当に bypass する
- `proc/jogsarar` に本実装を追加し続ける

---

## 12. Codex への具体指示

Codex は次の順で実装すること。

1. 既存 `psn`, `blindtrace`, `example_train_fbp.py`, `proc/jogsarar` を読む。
2. `fbpick/common` と `fbpick/coarse` を新設する。
3. 既存 `EncDec2D` と tiled infer を流用して coarse を成立させる。
4. その後 `fb_local_ops.py` と `local_window_dataset.py` を追加する。
5. `fine` を coarse artifact 入力で成立させる。
6. 最後に `global_qc` を追加する。

PR ごとに必ず次を満たすこと。

- 新規ファイルは責務が単一である
- import 循環を作らない
- 既存 pipeline の public API を壊さない
- config load error は明確なメッセージで止める

---

## 13. 初版完了条件

初版は次を満たせば完了とする。

- `python cli/run_fbpick_coarse_train.py --config ...` が動く
- `python cli/run_fbpick_coarse_infer.py --config ...` が動く
- `python cli/run_fbpick_fine_train.py --config ...` が動く
- `python cli/run_fbpick_fine_infer.py --config ...` が動く
- `python cli/run_fbpick_global_qc.py --config ...` が動く
- 1 本の SEG-Y から最終 pick artifact を返せる

---

## 14. 将来拡張メモ

初版の後で検討してよいもの:

- fine の offset / relative time channel 再導入による多チャンネル化
- fine 専用 head の `seisai-models` 追加
- 3D inversion backend 実装の本格化
- pseudo-label self-training ループ
- static correction への接続
- GUI / viewer 統合
