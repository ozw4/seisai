# SegyGatherPairDataset 例外系テスト 技術メモ(保証内容と確認点)

本メモは `tests/test_segy_gather_pair_dataset_exceptions.py` に含まれる例外系 pytest の各テストが、**何を保証し**、**何を確認しているか**を整理したものです。
対象は `SegyGatherPairDataset` の「壊れた入力や契約違反を早期に検知し、明示的に失敗する」挙動です。

---

## 前提・補助関数の役割

### `_write_dummy_segy(path, n_traces, n_samples, dt_us)`
- **目的**: テスト用の最小SEG-Yを `tmp_path` 下に生成する。
- **生成物の性質**:
  - trace数 `n_traces`
  - サンプル数 `n_samples`
  - サンプリング間隔 `dt_us`(µs)
  - 振幅は全て0(IEEE float)
  - 最低限のトレースヘッダ(`FieldRecord`, `TraceNumber`)を埋める
- **保証**: SEGYの「形」だけ整え、例外系の条件(nsamples/dt/tracecount差)を意図通り発生させられる。

### `_make_identity_plan()`
- **目的**: PairDataset が返す `x_view_input`, `x_view_target` を、そのまま `input`, `target` に積む最小BuildPlanを提供する。
- **保証**: plan契約違反テスト以外で、planが原因で失敗しないこと(最小の正常系を固定)。

### `_BadTransformNon2D`
- **目的**: transform が **2D(H,W) numpy** を返す契約に違反させる。
- **保証**: Dataset 側の transform 入力検証(ndimチェック)が確実に走る状況を作る。

### `_BadTransformShapeMismatch`
- **目的**: inputとtargetで transform 結果の shape を意図的にずらす。
- **保証**: Dataset 側の「input/target transform後shape一致」検証が確実に走る状況を作る。

### `_NoopPlan`
- **目的**: plan が `input` / `target` を生成しない契約違反を作る。
- **保証**: Dataset 側の「planが必須キーを生やす」検証が確実に走る状況を作る。

---

## テスト一覧と保証内容

### 1. `test_pair_dataset_raises_on_empty_lists`
**目的**: 入力ファイルリストが空の場合の防御。

- **確認していること**
  - `input_segy_files=[]` かつ `target_segy_files=[]` が与えられた時、
    Dataset初期化が `ValueError` を投げること。
- **保証すること**
  - Datasetが「実行不能な設定(入力0件)」を黙って受理せず、**早期に明示的に失敗**する。
  - 以降のファイルIOやサンプル生成処理に進まない。

---

### 2. `test_pair_dataset_raises_on_length_mismatch`
**目的**: noisy/clean の「1対1対応」前提を破る設定の防御。

- **確認していること**
  - `len(input_segy_files) != len(target_segy_files)` のとき、
    Dataset初期化が `ValueError` を投げること。
- **保証すること**
  - PairDatasetが「ペア対応が定義できない設定」を許容せず、
    **入力段階で失敗**する。
  - 片側のみ余るなどの不定な対応関係を排除できる。

---

### 3. `test_pair_dataset_raises_on_nsamples_mismatch`(integration)
**目的**: input/target のサンプル数不一致(nsamples mismatch)を検知できること。

- **セットアップ**
  - input: `n_samples=1024`
  - target: `n_samples=2048`
  - trace数とdtは一致
- **確認していること**
  - 初期化で `ValueError` が投げられること。
- **保証すること**
  - PairDatasetが「時間軸長が異なる」ペアを受理せず、
    学習・推論で shape が破綻する前に **初期化段階で停止**する。
  - 「後段での静かなbroadcast/切り詰め/例外」などの不安定性を防止。

---

### 4. `test_pair_dataset_raises_on_tracecount_mismatch`(integration)
**目的**: input/target のトレース数不一致(tracecount mismatch)を検知できること。

- **セットアップ**
  - input: `n_traces=16`
  - target: `n_traces=17`
  - n_samplesとdtは一致
- **確認していること**
  - 初期化で `ValueError` が投げられること。
- **保証すること**
  - PairDatasetが「1対1のtrace対応」が破綻するペアを拒否し、
    index参照ずれや境界外アクセスの前に **安全に停止**する。

---

### 5. `test_pair_dataset_raises_on_dt_mismatch`(integration)
**目的**: input/target のサンプリング間隔不一致(dt mismatch)を検知できること。

- **セットアップ**
  - input: `dt_us=2000`
  - target: `dt_us=1000`
  - n_samplesとtrace数は一致
- **確認していること**
  - 初期化で `ValueError` が投げられること。
- **保証すること**
  - PairDatasetが「同じサンプル index が同じ物理時間を表さない」ペアを拒否し、
    訓練ターゲットの意味が崩れる状況を **初期化で遮断**する。

---

### 6. `test_pair_dataset_raises_on_transform_non_2d`(integration)
**目的**: transform が 2D(H,W) を返す契約を破った場合、明示的に失敗すること。

- **セットアップ**
  - transform が 1D array を返す `_BadTransformNon2D`
  - SEGYは正常ペア
- **確認していること**
  - `ds[0]`(サンプル取得時)に `ValueError` が投げられること。
- **保証すること**
  - transform 実装ミス(ndim違反)を Dataset が検知し、
    後続の builder/stack 処理で原因不明なエラーになる前に
    **責務境界で明確に停止**できる。

---

### 7. `test_pair_dataset_raises_on_transform_shape_mismatch`(integration)
**目的**: input/target で transform 後の shape が一致しない場合に失敗すること。

- **セットアップ**
  - transform が呼び出し回数によって返り shape を変える `_BadTransformShapeMismatch`
  - SEGYは正常ペア
- **確認していること**
  - `ds[0]`(サンプル取得時)に `ValueError` が投げられること。
- **保証すること**
  - input/target が同一窓(同一shape)で比較される前提を Dataset が守る。
  - train/infer が silent に misaligned になる(入力窓と教師窓がズレる)事故を、
    **サンプル生成時点で確実に検出**する。

---

### 8. `test_pair_dataset_raises_on_plan_not_populating_input_target`(integration)
**目的**: plan が `input` と `target` を生成する契約を満たさない場合に失敗すること。

- **セットアップ**
  - plan として `_NoopPlan` を渡し、`input` / `target` を作らない
  - transformはNone、SEGYは正常ペア
- **確認していること**
  - `ds[0]`(サンプル取得時)に `KeyError` が投げられること。
- **保証すること**
  - BuildPlan/Builder の設定ミスで「学習に必要なキーが欠落」する場合に、
    後段の train loop で曖昧なエラーになる前に
    **Datasetの責務として明示的に失敗**できる。

---

## integration マーカーの意義

`@pytest.mark.integration` が付与されたテストは、以下を意味します。

- **segyio が必要**(`pytest.importorskip("segyio")` により未導入環境ではスキップ)
- 一時ファイルに実SEG-Y互換ファイルを生成するため、pure unit より重い
- CIでは `-m "not integration"` で軽量テストのみ走らせ、必要時に `-m integration` を回す運用が可能

---

## これらのE系テストで保証できる品質特性

- **Fail Fast**: 破綻した入力/設定は早期に例外で停止する
- **診断容易性**: 原因のカテゴリ(リスト不備 / ペア整合 / transform契約 / plan契約)がテスト単位で明確
- **Pair前提の保護**: noisy/clean の 1対1対応の前提を nsamples/tracecount/dt の3軸で担保

---

## 次に足すと良いE系(未カバー)

- `max_trials` 到達で `RuntimeError` を出すケース(常に無効サンプルを返す transform 等で再現)
- SEGY読み込み自体の例外(ファイル破損/パス不正)のメッセージ整備とテスト
- input/target のヘッダ一致(ffid/chno/cmp/offset)を「検証する仕様」を入れた場合のテスト
