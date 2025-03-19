export CUDA_VISIBLE_DEVICES=0

model=ViT-B-16
num_tasks=8

python main.py method="scale_svd" model=${model} num_tasks=${num_tasks}