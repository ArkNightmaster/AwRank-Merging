export CUDA_VISIBLE_DEVICES=2

model=ViT-B-16
num_tasks=8

python main.py method="ties" model=${model} num_tasks=${num_tasks}