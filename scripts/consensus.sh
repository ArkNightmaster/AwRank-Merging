export CUDA_VISIBLE_DEVICES=3

model=ViT-B-16
num_tasks=8

python main.py method="consensus" model=${model} num_tasks=${num_tasks}