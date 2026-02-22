CUDA_VISIBLE_DEVICES=0 python examples/examples_train_blindtrace.py --config proc/mae/configs/aug_v1_A10_hflip_p05.yaml &
CUDA_VISIBLE_DEVICES=1 python examples/examples_train_blindtrace.py --config proc/mae/configs/aug_v1_A11_polarity_p05.yaml
CUDA_VISIBLE_DEVICES=0 python examples/examples_train_blindtrace.py --config proc/mae/configs/aug_v1_A12_space_weak.yaml &
CUDA_VISIBLE_DEVICES=1 python examples/examples_train_blindtrace.py --config proc/mae/configs/aug_v1_A13_time_weak.yaml
CUDA_VISIBLE_DEVICES=0 python examples/examples_train_blindtrace.py --config proc/mae/configs/aug_v1_A14_freq_weak_mix.yaml &

CUDA_VISIBLE_DEVICES=1 python examples/examples_train_blindtrace.py --config proc/mae/configs/aug_v1_A20_space_mid.yaml
CUDA_VISIBLE_DEVICES=0 python examples/examples_train_blindtrace.py --config proc/mae/configs/aug_v1_A21_time_mid.yaml &
CUDA_VISIBLE_DEVICES=1 python examples/examples_train_blindtrace.py --config proc/mae/configs/aug_v1_A22_freq_mid_mix.yaml
CUDA_VISIBLE_DEVICES=0 python examples/examples_train_blindtrace.py --config proc/mae/configs/aug_v1_A23_freq_mid_bandpass.yaml &
CUDA_VISIBLE_DEVICES=1 python examples/examples_train_blindtrace.py --config proc/mae/configs/aug_v1_A24_freq_mid_mix_restd.yaml
