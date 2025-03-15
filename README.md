# No Task Left Behind: Isotropic Model Merging with Common and Task-Specific Subspaces
[![arXiv](https://img.shields.io/badge/arXiv-2502.04959-b31b1b.svg?style=flat)](https://arxiv.org/abs/2502.04959)

<div align="left"><img src="fig/teaser.png" width="100%" alt="schema"></div>

> Spectrum of singular values for a single layer weight update matrix obtained by merging using Task Arithmetic (top) compared to our approaches: Iso-C (middle) and Iso-CTS (bottom). Task Arithmetic sums the task-specific matrices, which result in a spectrum with a few dominant components. Iso-C instead replaces this spectrum with a uniform one, which results in significant performance improvement. Iso-CTS enhances the common subspace with task-specific subspaces and yields state-of-the-art model merging performance.


## 🚀 Setup

### Download fine-tuned checkpoints
Use the checkpoints provided by [Task Singular Vectors](https://drive.google.com/drive/folders/1UEM1Thcz1c7dc1nji1i5uTN53Kf6G3-e?usp=sharing) (which are the same as provided by [Tall Masks](https://drive.google.com/drive/folders/15ParSng4d5xSdaWdBFsg1617zPXT8Dae)). 

### Download the datasets
Most datasets being used should be downloaded automatically with `torchvision` or `huggingface`. For the datasets requiring manual preparation (like Cars, DTD, EuroSAT, SUN397), please follow the instructions in [this issue](https://github.com/mlfoundations/task_vectors/issues/1). Depending on the `torchvision` version, some issues might arise when downloading specific datasets like [here](https://github.com/basveeling/pcam/issues/4) or [here](https://github.com/pytorch/vision/issues/5662). In this case, using a different `torchvision` version might solve the issue. 

### Set data and models locations
Modify `model_location` and `data_location` in `config/config.yaml` before evaluation. 

### Prepare the environment
```bash
conda env create
conda activate iso-merging
```


## 🔄 Merging methods
### `Iso-C`: Isotropic Merging in Common Subspace
tldr ✅: Merge by Task Arithmetic (summation) and make the spectrum of singular values uniform.
### `Iso-CTS`: Isotropic Merging in Common and Task-Specific Subspaces
tldr ✅: Merge by Task Arithmetic (common subspace), replace the least significant singular vectors by task-specific ones (task-specific subspaces) and and make the spectrum of singular values uniform.
### `Iso-IASAD`: Isotropic Merging with Input-Aware Scaling and Alternating Decomposition
tldr ✅: Uses layer input statistics to scale parameters, decomposes them into low-rank (shared) and sparse (task-specific) components through iterative optimization, and makes the spectrum uniform.


## 🧪 Merge and eval
```bash
model=ViT-B-16
num_tasks=8

# Collect layer statistics (needed for Iso-IASAD)
python main.py method="collect_stats" model=${model} num_tasks=${num_tasks}

# Merge and evaluate Iso-C
python main.py method="iso_c" model=${model} num_tasks=${num_tasks}

# Merge and evaluate Iso-CTS
python main.py method="iso_cts" model=${model} num_tasks=${num_tasks} method.common_space_fraction=0.8

# Merge and evaluate Iso-IASAD
python main.py method="iso_iasad" model=${model} num_tasks=${num_tasks} method.num_iters=10 method.sparsity=0.1 method.rank_ratio=0.75
```

For more details on the Iso-IASAD algorithm, see [docs/iso_iasad.md](docs/iso_iasad.md).

## 📊 Evaluation Process

Our evaluation framework follows a two-stage process to ensure robust performance assessment of merged models:

### Validation Phase
1. **Task Vector Creation**: Based on the specified merging method (Iso-C, Iso-CTS, or Iso-IASAD), we create a merged task vector from individual fine-tuned models.
2. **Validation Set Evaluation**: The merged model is first evaluated on validation sets for all tasks.
3. **Optimal Scaling Factor**: For methods like Task Arithmetic, TIES, and Consensus Merging, we determine the optimal scaling coefficient (α) that maximizes the average normalized accuracy across all validation tasks.
4. **Mask Optimization**: For masking-based methods (TALL Mask, Magnitude Masking), we find the optimal mask configuration for each task.

### Test Phase
1. **Final Evaluation**: Using the optimal scaling factor or masks determined in the validation phase, we evaluate the merged model on the test sets.
2. **Performance Metrics**:
   - **Absolute Accuracy**: Raw classification accuracy on each task.
   - **Normalized Accuracy**: Accuracy normalized by the performance of individual task-specific fine-tuned models, calculated as: `test_accuracy / fine-tuned_accuracy`.
   - **Average Performance**: We report both the average absolute accuracy and average normalized accuracy across all tasks.

### Key Evaluation Metrics
- **Average Normalized Top-1**: This is our primary metric, representing how well the merged model preserves the performance of individual task-specific models.
- **Average Top-1**: The raw classification accuracy averaged across all tasks.

The evaluation results are automatically logged and can be visualized to compare different merging strategies. Our isotropic merging methods (Iso-C, Iso-CTS, Iso-IASAD) consistently achieve higher normalized accuracy compared to previous approaches, demonstrating their effectiveness in preserving task-specific knowledge while enabling multi-task capabilities.
