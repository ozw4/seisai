CUDA_VISIBLE_DEVICES=0 python examples/examples_train_blindtrace.py --config proc/mae/configs/backbone_D3_convnext_small.yaml &
CUDA_VISIBLE_DEVICES=1 python examples/examples_train_blindtrace.py --config proc/mae/configs/backbone_D3_convnext_base.yaml &


