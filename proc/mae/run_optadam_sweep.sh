CUDA_VISIBLE_DEVICES=0 python examples/examples_train_blindtrace.py --config proc/mae/configs/optim_v1_lr5e-5_wd1e-2.yaml &
CUDA_VISIBLE_DEVICES=1 python examples/examples_train_blindtrace.py --config proc/mae/configs/optim_v1_lr1e-4_wd1e-2.yaml
CUDA_VISIBLE_DEVICES=0 python examples/examples_train_blindtrace.py --config proc/mae/configs/optim_v1_lr2e-4_wd1e-2.yaml &
CUDA_VISIBLE_DEVICES=1 python examples/examples_train_blindtrace.py --config proc/mae/configs/optim_v1_lr3e-4_wd1e-2.yaml
CUDA_VISIBLE_DEVICES=0 python examples/examples_train_blindtrace.py --config proc/mae/configs/optim_v1_lr1e-4_wd0.yaml &
CUDA_VISIBLE_DEVICES=1 python examples/examples_train_blindtrace.py --config proc/mae/configs/optim_v1_lr1e-4_wd5e-2.yaml
CUDA_VISIBLE_DEVICES=0 python examples/examples_train_blindtrace.py --config proc/mae/configs/optim_v1_lr1e-4_wd1e-2_gn05.yaml &
CUDA_VISIBLE_DEVICES=1 python examples/examples_train_blindtrace.py --config proc/mae/configs/optim_v1_lr1e-4_wd1e-2_gn20.yaml
