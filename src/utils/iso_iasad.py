import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import os
from src.utils.layer_stats import get_layer_stats


@torch.no_grad()
def scale_svd(tensor, scaler_row, num_iters=10, sparsity=0.1, rank_ratio=0.75, prune_level="row"):
    """
    对张量应用输入感知缩放和交替分解。
    
    参数：
        tensor: 要处理的权重张量
        scaler_row: 用于缩放的输入统计信息
        num_iters: 交替优化的迭代次数
        sparsity: 任务特定组件的稀疏度
        rank_ratio: 用于低秩组件的秩比例
        prune_level: 剪枝策略（'row'或'element'）
        
    返回：
        具有低秩和稀疏组件的处理后的张量
    """
    # 准备参数
    dense_alloc = 1 - sparsity
    diag_approx = scaler_row.clone().to(tensor.device)
    orig_weight = tensor.clone().detach().float()
    d_out, d_in = orig_weight.shape
    
    # 根据张量大小和密度分配计算目标秩
    target_rank = int(rank_ratio * dense_alloc * (d_out*d_in)/(d_out + d_in))
    target_rank = max(1, min(target_rank, min(d_out, d_in) - 1))  # 确保有效秩
    unstruct_sparse = 1.0 - (1.0-rank_ratio)*dense_alloc
    
    # 应用交替最小二乘优化
    res = altern_ls(orig_weight, diag_approx, num_iters, target_rank, unstruct_sparse, prune_level)
    return res


@torch.no_grad()
def altern_ls(weight, diag_approx, num_iters, target_rank, unstruct_sparse, prune_level="row"):
    """
    执行交替最小二乘优化以分解权重矩阵。
    
    参数：
        weight: 要分解的权重矩阵
        diag_approx: 用于缩放的输入统计信息
        num_iters: 迭代次数
        target_rank: 低秩组件的目标秩
        unstruct_sparse: 稀疏组件的稀疏度
        prune_level: 剪枝策略（'row'或'element'）
        
    返回：
        分解后的权重矩阵（低秩+稀疏）
    """
    # 通过输入统计信息缩放权重
    if diag_approx.shape[0] != weight.shape[1]:
        # 处理维度不匹配（例如，对于注意力层）
        diag_approx = diag_approx.repeat(weight.shape[1] // diag_approx.shape[0])
    
    # 确保diag_approx具有正确的形状
    diag_approx = diag_approx[:weight.shape[1]]
    
    # 添加小的epsilon以避免除以零
    diag_approx = diag_approx + 1e-8
    
    # 通过输入统计信息缩放权重
    scaled_weight = weight * torch.sqrt(diag_approx)
    sparse_component = torch.zeros_like(scaled_weight)
    
    # 迭代优化
    for iter_idx in range(num_iters):
        # 通过SVD进行低秩分解
        try:
            U, S, V = torch.linalg.svd(scaled_weight - sparse_component, full_matrices=False)
            # 只保留前k个奇异值
            S_truncated = S.clone()
            S_truncated[target_rank:] = 0
            low_rank_component = U @ torch.diag(S_truncated) @ V
        except:
            # 如果SVD失败，则使用原始权重
            print(f"SVD失败，对迭代{iter_idx}使用原始权重")
            low_rank_component = scaled_weight - sparse_component
        
        # 更新稀疏组件
        sparse_component = scaled_weight - low_rank_component
        
        # 对稀疏组件应用稀疏化
        W_metric = sparse_component.abs()
        W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
        
        # 行级剪枝
        if prune_level == "row":
            # 对于每一行，只保留最小的元素（最多unstruct_sparse比例）
            num_to_keep = int(W_metric.shape[1] * unstruct_sparse)
            for i in range(W_metric.shape[0]):
                _, indices = torch.topk(W_metric[i], k=num_to_keep, largest=False)
                W_mask[i, indices] = True
        # 元素级剪枝
        else:
            # 整体上只保留最小的元素（最多unstruct_sparse比例）
            num_to_keep = int(W_metric.numel() * unstruct_sparse)
            _, indices = torch.topk(W_metric.view(-1), k=num_to_keep, largest=False)
            W_mask.view(-1)[indices] = True
        
        # 对稀疏组件应用掩码
        sparse_component[W_mask] = 0
    
    # 最终处理：重新缩放组件
    # 提取最终的低秩矩阵
    try:
        U, S, V = torch.linalg.svd(scaled_weight - sparse_component, full_matrices=False)
        S_truncated = S.clone()
        S_truncated[target_rank:] = 0
        low_rank_matrix = U @ torch.diag(S_truncated) @ V
    except:
        # 如果SVD失败，则使用最后一次迭代的结果
        print("最终SVD失败，使用最后一次迭代的结果")
        low_rank_matrix = low_rank_component
    
    # 重新缩放组件
    inv_scale = 1 / torch.sqrt(diag_approx)
    low_rank_matrix = low_rank_matrix * inv_scale
    sparse_comp = sparse_component * inv_scale
    
    # 组合组件
    final_matrix = low_rank_matrix + sparse_comp
    return final_matrix


@torch.no_grad()
def iso_iasad(task_vectors, config):
    """
    执行基于输入感知缩放和交替分解的各向同性合并。
    
    参数：
        task_vectors: 任务向量列表
        config: 配置参数
        
    返回：
        合并向量
    """
    # 记录原始设备，以便在计算结束后恢复
    original_devices = [next(iter(tv.vector.values())).device for tv in task_vectors]
    
    # 确保所有计算在GPU上进行
    device = config.device
    
    # 将所有任务向量移动到GPU
    for tv in task_vectors:
        for k in tv.vector:
            tv.vector[k] = tv.vector[k].to(device)
    
    new_vector = {}
    
    # 尝试为每个数据集加载层统计信息
    layer_stats_dict = {}
    stats_dir = os.path.join('results/layer_stats')
    os.makedirs(stats_dir, exist_ok=True)
    
    for dataset in config.DATASETS:
        try:
            stats_path = os.path.join(stats_dir, f"{config.model}_{dataset}_stats.pt")
            if os.path.exists(stats_path):
                # 将统计信息直接加载到GPU
                layer_stats_dict[dataset] = {k: v.to(device) for k, v in torch.load(stats_path).items()}
                print(f"已加载{dataset}的统计信息")
            else:
                print(f"警告：未找到{dataset}的统计信息文件。使用默认缩放。")
        except Exception as e:
            print(f"加载{dataset}的统计信息时出错：{e}")
    
    print("计算SVD...")
    # 预先计算合并模型（仅计算一次）
    from src.models.task_vectors import NonLinearTaskVector
    
    # 创建一个简单的平均作为合并模型
    merged_vector = {}
    for key in task_vectors[0].vector:
        merged_vector[key] = sum([tv.vector[key] for tv in task_vectors]) / len(task_vectors)
    
    # 创建一个虚拟任务向量对象来保存合并向量
    merged_model = NonLinearTaskVector(model_name=config.model, vector=merged_vector)
    
    # 处理每个键
    for key in task_vectors[0].vector:
        shape_ = task_vectors[0].vector[key].shape
        
        # 对于非矩阵参数或文本投影，使用简单平均
        is_2d_matrix = (len(shape_) == 2) and ("text_projection" not in key)
        if not is_2d_matrix:
            print(f"通过平均合并{key}...")
            for i, (task_vector, dataset) in enumerate(zip(task_vectors, config.DATASETS)):
                vec = task_vector.vector[key]
                if i == 0:
                    new_vector[key] = vec.clone()
                else:
                    new_vector[key] += (vec - new_vector[key]) / (i + 1)
            continue
        
        # 对于矩阵参数，应用我们的方法
        print(f"计算{key}的合并向量...")
        
        # 使用输入感知缩放处理每个任务向量
        processed_vectors = []
        for i, (task_vector, dataset) in enumerate(zip(task_vectors, config.DATASETS)):
            if dataset in layer_stats_dict:
                # 应用输入感知缩放和分解
                # 注意：这里我们只计算当前key的处理向量，而不是所有key
                vector = {}
                vector[key] = task_vector.vector[key] - merged_model.vector[key]
                
                # 查找相应的层统计信息
                layer_name = None
                for stat_key in layer_stats_dict[dataset]:
                    if stat_key.endswith(key.split('.')[-2]):
                        layer_name = stat_key
                        break
                
                if layer_name is not None:
                    # 应用缩放和分解
                    processed_vector = scale_svd(
                        vector[key], 
                        layer_stats_dict[dataset][layer_name],
                        num_iters=config.method.num_iters,
                        sparsity=config.method.sparsity,
                        rank_ratio=config.method.rank_ratio,
                        prune_level=config.method.prune_level
                    )
                    processed_vectors.append(processed_vector)
                else:
                    # 如果没有找到对应的层统计信息，使用原始向量
                    processed_vectors.append(vector[key])
            else:
                # 如果没有可用的统计信息，则回退到原始向量
                processed_vectors.append(task_vector.vector[key] - merged_model.vector[key])
        
        # 组合处理后的向量
        combined_w = sum(processed_vectors)
        
        # 应用SVD并使频谱均匀（类似于iso_c）
        u, s, v = torch.linalg.svd(combined_w, full_matrices=False)
        s_mean = torch.ones_like(s) * s.mean()
        
        new_vector[key] = torch.linalg.multi_dot(
            (
                u,
                torch.diag(s_mean),
                v,
            )
        )
    
    # 将任务向量恢复到原始设备
    for i, tv in enumerate(task_vectors):
        for k in tv.vector:
            tv.vector[k] = tv.vector[k].to(original_devices[i])
    
    # 将结果移回CPU（如果需要）
    if config.device != torch.device('cpu'):
        for k in new_vector:
            new_vector[k] = new_vector[k].cpu()
    
    return new_vector

