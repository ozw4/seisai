# blindtrace 推論CLI 追加: 事前調査メモ（最小変更）

更新日: 2026-02-25

## 必須調査の結論

1. 既存SEGY→SEGY推論（`packages/seisai-engine/src/seisai_engine/infer/ffid_segy2segy.py`）
- Iterator は `FFIDGatherIterator` を使用。
- 各ファイルごとに `out_hw(n_traces,n_samples)` を確保し、`iter_gathers(file_indices=[fi])` の `trace_indices` で gather 推論結果を元trace順に再配置。
- `seen` で未充填traceを検出し、全traceが埋まらなければ即失敗。
- 最後に `write_segy_like_input` へ渡してファイル単位で書き出す流れ。

2. SEGY書き込み（`packages/seisai-utils/src/seisai_utils/segy_write.py`）
- `write_segy_like_input` は「入力ファイルを丸ごとコピーして trace data だけ上書き」なので、trace header/binary header/text header は保持される。
- ただし sample format は入力維持（出力dtypeは dst trace dtype にキャスト）で、float32固定（format=5）には未対応。
- Textual File Header (`text[0]`) 追記APIは未実装。
- `segyio.tools.create_text_header` が使えるため、80x40整形（C01..C40）で末尾行上書きは実装可能。

3. blindtrace モデル再構築（`packages/seisai-engine/src/seisai_engine/pipelines/blindtrace/build_model.py` ほか）
- `build_model` は `encdec2d_kwargs(dict)` を受けて `EncDec2D` を組み立てる薄いラッパ。
- 学習ckptには `model_sig` と `cfg` が保存される（`train_skeleton_checkpoint.py`）。
- 既存再構築例は `viewer/fbpick.py` にあり、`model_sig` からモデル再構築し、`infer_used_ema` と `ema_state_dict` があればEMA重みを優先している。

4. mask推論（`packages/seisai-transforms/src/seisai_transforms/mask_inference.py`）
- `cover_all_traces_predict_striped` の入出力は `x: (B,1,H,W) -> y: (B,1,H,W)`。
- 1パス内推論は `with autocast('cuda', enabled=use_amp): yb = model(xmb)`。
- 完全被覆は `hits` で合成しているが、現状は `clamp_min(1)` で割る実装（0 hitを明示的に失敗させていない）。
- 直接 `model(xmb)` 呼び出しのため、W方向tiledを差し込むには `predict_fn` 注入（または同等の拡張）が必要。

## 確定実装方針（最小差分）

- 既存の gather集約骨格（`FFIDGatherIterator` + trace_indices再配置）を再利用する。
- 既存 `write_segy_like_input` は壊さず拡張し、float32固定出力と `text[0]` 追記を追加する。
- 推論CLIは task別に `cli/run_blindtrace_infer.py` を新設し、実処理は `seisai_engine.pipelines.blindtrace` 側へ置く。
- 設定マージ順は `default < ckpt.cfg < infer.yaml < unknown overrides(KEY=VALUE)` を明示実装する。
- override は安全キーのみ許可し、範囲外は `allow_unsafe_override=true` のときだけ許可する。

## TODO（追加/修正予定ファイル）

- [ ] `cli/run_blindtrace_infer.py`（新規）
  - thin entrypoint を追加（`--config` と unknown overrides をパイプラインへ委譲）。

- [ ] `packages/seisai-engine/src/seisai_engine/pipelines/blindtrace/infer_segy2segy.py`（新規）
  - blindtrace推論の本体CLIロジックを実装。
  - ckpt読込、`cfg/model_sig` 復元、設定マージ、safe override検証、SEGY入出力、`*.mlmeta.json` 出力を担当。

- [ ] `packages/seisai-engine/src/seisai_engine/pipelines/blindtrace/__init__.py`（修正）
  - 新しい infer エントリ（main）を公開。

- [ ] `packages/seisai-engine/src/seisai_engine/infer/ffid_segy2segy.py`（修正）
  - 既存の gather→file 集約処理を blindtrace推論から再利用できるように最小抽象化（重複回避）。

- [ ] `packages/seisai-transforms/src/seisai_transforms/mask_inference.py`（修正）
  - 全trace被覆を未達時に即失敗させる。
  - tiled推論を差し込める拡張（`predict_fn` 追加）を入れる。

- [ ] `packages/seisai-utils/src/seisai_utils/segy_write.py`（修正）
  - format=5(float32)固定で書く経路を追加。
  - trace header全コピー・trace順維持を明示保証。
  - Textual File Header (`text[0]`) の末尾予約行（例: C37-C40）上書きヘルパを追加。

- [ ] `packages/seisai-engine/tests/test_blindtrace_infer_*`（新規）
  - 設定マージ優先順位と safe override 制約のテストを追加。

- [ ] `packages/seisai-transforms/tests/test_cover_all_traces_predict_striped.py`（修正）
  - 被覆未達の即失敗・`predict_fn` 経路のテストを追加。

- [ ] `packages/seisai-utils/tests/test_segy_write.py`（新規）
  - float32固定出力、trace header一致、text[0]追記、sidecar前提情報の検証を追加。
