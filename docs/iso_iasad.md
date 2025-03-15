# Iso-IASAD: 基于输入感知缩放和交替分解的各向同性模型合并

## 概述

Iso-IASAD是一种先进的模型合并算法，它通过三个关键改进扩展了各向同性合并方法：

1. **输入感知缩放**：使用层输入统计信息根据参数的重要性进行缩放
2. **交替分解**：将参数分解为低秩（共享知识）和稀疏（任务特定）组件
3. **迭代优化**：采用迭代优化方法来优化这两个组件

## 算法详情

### 输入感知缩放

该算法使用层输入统计信息来确定每个参数的重要性。接收更强输入信号的参数被认为更重要，并相应地进行缩放。这基于这样的直觉：具有较大输入幅度的参数对模型输出的影响更大。

缩放的执行方式如下：
```python
scaled_weight = weight * torch.sqrt(diag_approx)
```
其中`diag_approx`包含每个神经元的输入统计信息。

### 交替分解

该算法将每个权重矩阵分解为两个组件：
1. 捕获跨任务共享知识的低秩组件
2. 保留任务特定知识的稀疏组件

这种分解是通过交替最小二乘优化来执行的，我们迭代地优化这两个组件，以更好地表示原始权重矩阵。

### 迭代优化

优化过程在以下步骤之间交替进行：
1. 计算残差（原始矩阵减去稀疏组件）的低秩近似
2. 计算稀疏组件作为残差（原始矩阵减去低秩组件）
3. 对稀疏组件应用稀疏化

这个迭代过程有助于更好地分离共享知识和任务特定知识。

## 实现细节

### 层统计信息收集

在使用Iso-IASAD之前，我们需要收集每一层的输入统计信息。这是通过以下步骤完成的：
1. 用钩子包装每一层，记录输入统计信息
2. 在训练数据的子集上运行模型
3. 计算每个神经元输入的平方L2范数

```python
# 统计信息收集示例
wrapped_layers = {}
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        wrapped_layers[name] = WrappedOATS(module, layer_name=name)

# 处理数据集
for batch in dataloader:
    outputs = model(**batch)

# 保存统计信息
layer_stats = {name: wrapper.scaler_row for name, wrapper in wrapped_layers.items()}
```

### 交替最小二乘优化

算法的核心是交替最小二乘优化：

```python
# 通过输入统计信息缩放权重
scaled_weight = weight * torch.sqrt(diag_approx)
sparse_component = torch.zeros_like(scaled_weight)

# 迭代优化
for iter_idx in range(num_iters):
    # 通过SVD进行低秩分解
    U, S, V = torch.linalg.svd(scaled_weight - sparse_component, full_matrices=False)
    S[target_rank:] = 0  # 只保留前k个奇异值
    low_rank_component = U @ torch.diag(S) @ V
    
    # 更新稀疏组件
    sparse_component = scaled_weight - low_rank_component
    
    # 对稀疏组件应用稀疏化
    # (只保留最小的元素)
    sparse_component = apply_sparsification(sparse_component, unstruct_sparse)
```

## 使用方法

要使用Iso-IASAD：

1. 首先，收集层统计信息：
```bash
python main.py method="collect_stats" model=ViT-B-16 num_tasks=8
```

2. 然后，使用Iso-IASAD进行合并和评估：
```bash
python main.py method="iso_iasad" model=ViT-B-16 num_tasks=8 method.num_iters=10 method.sparsity=0.1 method.rank_ratio=0.75
```

## 配置参数

该算法可以通过以下参数进行配置：

- `num_iters`：交替优化的迭代次数（默认：10）
- `sparsity`：任务特定组件的稀疏度（默认：0.1）
- `rank_ratio`：用于低秩组件的秩比例（默认：0.75）
- `prune_level`：剪枝策略（'row'或'element'）（默认：'row'）

## 优势

与之前的方法相比，Iso-IASAD提供了几个优势：

1. **更好的参数重要性估计**：通过使用输入统计信息，算法可以更好地估计每个参数的重要性。
2. **改进的知识分离**：交替分解有助于更好地分离共享知识和任务特定知识。
3. **增强的性能**：迭代优化过程导致在所有任务上都有更好的性能。
4. **鲁棒性**：该算法对任务难度和数据分布的变化更加鲁棒。

## 参考文献

此实现基于以下论文中描述的工作：
- "No Task Left Behind: Isotropic Model Merging with Common and Task-Specific Subspaces"（Marczak等，2025） 