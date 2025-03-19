export CUDA_VISIBLE_DEVICES=1

model=ViT-B-16
num_tasks=8

python main.py method="sum" model=${model} num_tasks=${num_tasks}
