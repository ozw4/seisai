# PSN Comparison Configs

`proc/psn/configs/` には PSN 比較実験用の 6 設定を用意しています。
対象は `D3`, `D3noaa`, `A0`, `C1`, `D3-random`, `A0-random` です。
`*-random` は同一アーキテクチャで `train.init_ckpt` を持たないランダム初期化です。

python examples/example_train_psn.py --config proc/psn/configs/psn_D3.yaml
python examples/example_train_psn.py --config proc/psn/configs/psn_D3noaa.yaml
python examples/example_train_psn.py --config proc/psn/configs/psn_A0.yaml
python examples/example_train_psn.py --config proc/psn/configs/psn_C1.yaml
python examples/example_train_psn.py --config proc/psn/configs/psn_D3-random.yaml
python examples/example_train_psn.py --config proc/psn/configs/psn_A0-random.yaml

best 判定は `infer_loss` のみ (`ckpt.save_best_only=true`, `ckpt.metric=infer_loss`, `ckpt.mode=min`) です。
追加メトリクスは infer で `infer/p_*` と、条件を満たす場合に `infer/s_*` が記録されます。
