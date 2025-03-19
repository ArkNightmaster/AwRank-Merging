export CUDA_VISIBLE_DEVICES=4

model=ViT-B-16
num_tasks=8

python main.py method="TSVM" model=${model} num_tasks=${num_tasks}
