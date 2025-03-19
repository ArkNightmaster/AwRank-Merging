# Model Merging Framework

This repository contains a framework for merging multiple fine-tuned models using various algorithms. The framework supports several merging methods including Task Arithmetic (TA), TIES, TALL, TSVM, Iso-C, Iso-CTS, and our newly added Scale-SVD method.

## Project Structure

- `src/`: Main source code
  - `models/`: Model definitions and task vector implementations
  - `utils/`: Utility functions and merging algorithms
  - `eval/`: Evaluation code
  - `datasets/`: Dataset implementations
- `config/`: Configuration files
- `scripts/`: Helper scripts
- `results/`: Results and outputs

## Merging Algorithms

### Existing Algorithms

1. **Task Arithmetic (TA)**: Simple addition of task vectors.
2. **TIES Merging**: Task vectors merging with interference reduction.
3. **TALL Mask**: Task vectors with layer-wise masking.
4. **TSVM**: Task Space Singular Value Merging - uses SVD to merge task vectors.
5. **Iso-C**: Isotropic Consolidation - makes singular values uniform.
6. **Iso-CTS**: Isotropic Common and Task-Specific spaces.

### New Algorithm: Scale-SVD

The Scale-SVD algorithm is a new method for task vector pruning and compression. It applies a low-rank plus sparse decomposition to task vectors, using input activations as scaling factors. This method:

1. Extracts a scale vector from the difference between a model and a merged model
2. Applies SVD with scaling based on layer inputs
3. Decomposes weights into low-rank and sparse components
4. Optimizes the trade-off between model size and performance

## ViT Model Support

The Scale-SVD algorithm has been adapted to work with Vision Transformer (ViT) models. The implementation:

1. Handles ViT-specific layer naming conventions and structures
2. Collects comprehensive layer statistics for all layer types:
   - **Attention Layers**: 
     - `attn.in_proj_weight`: Statistics for the combined QKV projection
     - `attn.out_proj.weight`: Statistics for the output projection
   - **MLP Layers**: 
     - `mlp.c_fc.weight`: Statistics for the first MLP layer (expansion)
     - `mlp.c_proj.weight`: Statistics for the second MLP layer (projection)
   - **LayerNorm Layers**: Statistics for normalization layers
   - **Convolutional Layers**: Statistics for patch embedding and other convolutions
   - **Linear Layers**: Statistics for all other linear projections
3. Uses these statistics to better scale the SVD decomposition for each layer

The layer statistics are organized in a hierarchical structure that matches the model's parameter names, making it easy to apply the appropriate scaling factors during merging.

## Usage

To use the Scale-SVD algorithm:

```python
from src.utils.scale_svd_utils import scale_svd_merging
from src.eval.aggregation import create_task_vector

# Configure the method in your config
config.method.name = "scale_svd"
config.method.sparsity = 0.1  # Sparsity level (0.0-1.0)
config.method.rank_ratio = 0.75  # Ratio for low-rank component

# Create and apply the task vector
task_vector, _ = create_task_vector(config)
```

### Collecting Layer Statistics

For optimal performance, you can collect comprehensive layer-specific activation statistics:

```bash
# Collect layer statistics for a specific dataset
python main.py method="collect_stats" model="ViT-B-16" method.batch_size=32 method.num_batches=50
```

Or use the test script:

```bash
# Test layer statistics collection
python scripts/test_scale_svd.py --layer-stats --config config/config.yaml
```

## Parameters

The Scale-SVD algorithm accepts the following parameters:

- `sparsity`: Controls the sparsity level (default: 0.1)
- `rank_ratio`: Controls the rank of the low-rank component (default: 0.75)
- `num_iters`: Number of iterations for alternating optimization (default: 10)
- `mask_strategy`: Strategy for masking ("row" or "global", default: "row")

## Implementation Details

The algorithm consists of three main components:

1. `extract_scale_vector`: Extracts a scale vector and applies SVD with scaling
2. `scale_svd`: Applies low-rank plus sparse decomposition to a tensor
3. `altern_ls`: Performs alternating optimization between low-rank and sparse components

See `src/utils/scale_svd_utils.py` for the full implementation. 