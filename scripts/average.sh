export CUDA_VISIBLE_DEVICES=0,1,2,3

model=ViT-B-16
num_tasks=8

python main.py method="average" model=${model} num_tasks=${num_tasks}