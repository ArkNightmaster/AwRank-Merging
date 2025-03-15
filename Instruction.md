# Isotropic Model Merging with Input-Aware Scaling and Alternating Decomposition

This repository contains the implementation of isotropic model merging techniques for Vision Transformer (ViT) models. The project focuses on merging multiple task-specific models into a single multi-task model while preserving performance across all tasks.

## Project Overview

The project implements several model merging algorithms:

1. **Iso-C**: Isotropic Merging in Common Subspace
   - Merges models by Task Arithmetic (summation) and makes the spectrum of singular values uniform.

2. **Iso-CTS**: Isotropic Merging in Common and Task-Specific Subspaces
   - Merges by Task Arithmetic (common subspace), replaces the least significant singular vectors by task-specific ones (task-specific subspaces), and makes the spectrum of singular values uniform.

3. **Iso-IASAD**: Isotropic Merging with Input-Aware Scaling and Alternating Decomposition (New)
   - Uses layer input statistics to scale parameters according to their importance
   - Decomposes parameters into low-rank (shared knowledge) and sparse (task-specific) components
   - Employs iterative optimization to refine both components

## Algorithm Details

### Iso-C and Iso-CTS

These algorithms focus on making the spectrum of singular values uniform after merging, which helps prevent dominant components from overshadowing others. Iso-CTS further enhances this by explicitly preserving task-specific subspaces.

### Iso-IASAD (New Algorithm)

The new algorithm extends the isotropic merging approach with three key improvements:

1. **Input-Aware Scaling**:
   - Uses layer input statistics to scale parameters according to their importance
   - Parameters that receive stronger input signals are considered more important

2. **Alternating Decomposition**:
   - Decomposes parameters into low-rank (shared knowledge) and sparse (task-specific) components
   - Low-rank component captures common knowledge across tasks
   - Sparse component preserves task-specific knowledge

3. **Iterative Optimization**:
   - Employs alternating least squares optimization to refine both components
   - Iteratively improves the decomposition to better balance common and task-specific knowledge

## Implementation

The implementation consists of three main components:

1. **Layer Statistics Collection**: Collects input statistics for each layer to determine parameter importance.
2. **Task-Specific Vector Extraction**: Extracts and scales task-specific vectors based on input statistics.
3. **Alternating Decomposition**: Decomposes parameters into low-rank and sparse components through iterative optimization.

## Usage

To use the new Iso-IASAD algorithm:

```bash
# First, collect layer statistics (if not already done)
python main.py method="collect_stats" model=ViT-B-16 num_tasks=8

# Then, merge and evaluate using Iso-IASAD
python main.py method="iso_iasad" model=ViT-B-16 num_tasks=8 method.num_iters=10 method.sparsity=0.1 method.rank_ratio=0.75
```

## Configuration Parameters

The Iso-IASAD algorithm can be configured with the following parameters:

- `num_iters`: Number of iterations for the alternating optimization (default: 10)
- `sparsity`: Sparsity level for the task-specific component (default: 0.1)
- `rank_ratio`: Ratio of the rank to use for the low-rank component (default: 0.75)

## Implementation Summary

The following files were created or modified to implement the Iso-IASAD algorithm:

1. **New Files**:
   - `src/utils/layer_stats.py`: Implements the layer statistics collection utilities
   - `src/utils/iso_iasad.py`: Implements the Iso-IASAD algorithm
   - `config/method/iso_iasad.yaml`: Configuration for the Iso-IASAD method
   - `config/method/collect_stats.yaml`: Configuration for statistics collection
   - `docs/iso_iasad.md`: Detailed documentation for the Iso-IASAD algorithm

2. **Modified Files**:
   - `src/eval/aggregation.py`: Updated to include the Iso-IASAD method and statistics collection
   - `scripts/merge_and_eval.sh`: Added commands for Iso-IASAD and statistics collection
   - `README.md`: Updated to include information about the Iso-IASAD algorithm

## Next Steps

To further improve the Iso-IASAD algorithm, consider the following:

1. **Hyperparameter Tuning**: Experiment with different values for `num_iters`, `sparsity`, and `rank_ratio` to find the optimal configuration for different tasks.

2. **Adaptive Sparsity**: Implement adaptive sparsity levels based on task difficulty or dataset characteristics.

3. **Layer-Specific Parameters**: Allow different hyperparameters for different layers, as some layers might benefit from higher sparsity or lower rank.

4. **Visualization Tools**: Develop tools to visualize the decomposition and understand how knowledge is shared across tasks.

5. **Performance Evaluation**: Conduct a comprehensive evaluation comparing Iso-IASAD with other merging methods across various tasks and model architectures.