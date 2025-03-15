export CUDA_VISIBLE_DEVICES=2,3,4,5

model=ViT-B-16
num_tasks=8

# Collect layer statistics (run this first)
# python main.py method="collect_stats" model=${model} num_tasks=${num_tasks}

# # Merge and evaluate Iso-C
# python main.py method="iso_c" model=${model} num_tasks=${num_tasks}

# # Merge and evaluate Iso-CTS
# python main.py method="iso_cts" model=${model} num_tasks=${num_tasks} method.common_space_fraction=0.8

# Merge and evaluate Iso-IASAD
python main.py method="iso_iasad" model=${model} num_tasks=${num_tasks} method.num_iters=20 method.sparsity=0.1 method.rank_ratio=0.75