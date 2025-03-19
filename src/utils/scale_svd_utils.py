import torch
import math
import os

# Set device for computation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.inference_mode()
def scale_svd_merging(task_vectors, config):
    """
    Applies Scale-SVD merging to task vectors.
    
    This method applies the scale_svd algorithm to each task vector individually,
    processing each parameter tensor to create a low-rank plus sparse decomposition.
    
    Args:
        task_vectors (list): A list of task vector objects
        config (DictConfig): Configuration object with method parameters
        
    Returns:
        list: A list of task vectors after Scale-SVD processing
    """
    # Import NonLinearTaskVector here to avoid circular import
    from src.models.task_vectors import NonLinearTaskVector
    
    # 获取设备，支持字符串或torch.device对象
    if hasattr(config, "device"):
        device = config.device
        if isinstance(device, str):
            device = torch.device(device)
    else:
        device = DEVICE
    
    print(f"Computing Scale-SVD merging on device: {device}...")
    
    # Get sparsity and rank ratio from config
    sparsity = getattr(config.method, "sparsity", 0.1)
    rank_ratio = getattr(config.method, "rank_ratio", 0.75)
    num_iters = getattr(config.method, "num_iters", 10)
    mask_strategy = getattr(config.method, "mask_strategy", "row")
    
    print(f"Parameters: sparsity={sparsity}, rank_ratio={rank_ratio}, num_iters={num_iters}, mask_strategy={mask_strategy}")
    
    # 尝试加载层统计信息
    layer_stats = None
    try:
        stats_dir = os.path.join('results/layer_stats')
        if os.path.exists(stats_dir):
            stats_files = [f for f in os.listdir(stats_dir) if f.endswith('_stats.pt')]
            if stats_files:
                model_name = getattr(config, "model", "ViT-B-16")
                dataset_name = config.DATASETS[0] if hasattr(config, "DATASETS") and config.DATASETS else "default"
                
                # 尝试找到匹配的统计文件
                stats_path = os.path.join(stats_dir, f"{model_name}_{dataset_name}_stats.pt")
                if os.path.exists(stats_path):
                    print(f"Loading layer statistics from {stats_path}")
                    layer_stats = torch.load(stats_path)
                else:
                    # 如果找不到特定数据集的统计信息，使用第一个可用的统计文件
                    print(f"Statistics file for {model_name}_{dataset_name} not found, using first available file")
                    stats_path = os.path.join(stats_dir, stats_files[0])
                    print(f"Loading layer statistics from {stats_path}")
                    layer_stats = torch.load(stats_path)
                
                print(f"Loaded statistics with {len(layer_stats)} entries")
                
                # 打印统计信息的键名，用于调试
                print("Statistics keys (first 10):")
                for i, key in enumerate(layer_stats.keys()):
                    if i < 10:
                        if isinstance(layer_stats[key], dict):
                            print(f"  {key}: <dict> with {len(layer_stats[key])} keys")
                            for subkey, value in layer_stats[key].items():
                                print(f"    {subkey}: shape={value.shape if isinstance(value, torch.Tensor) else 'not a tensor'}")
                        elif isinstance(layer_stats[key], torch.Tensor):
                            print(f"  {key}: shape={layer_stats[key].shape}")
                        else:
                            print(f"  {key}: {type(layer_stats[key])}")
            else:
                print("No statistics files found in results/layer_stats, using default statistics")
        else:
            print("Statistics directory not found, using default statistics")
    except Exception as e:
        print(f"Error loading layer statistics: {e}")
        import traceback
        traceback.print_exc()
    
    processed_vectors = []
    with torch.no_grad():
        # Process each task vector individually
        for task_idx, task_vector in enumerate(task_vectors):
            print(f"\nProcessing task vector {task_idx + 1}/{len(task_vectors)}...")
            new_vector = {}
            
            for key in task_vector.vector:
                # Move tensor to device
                tensor = task_vector.vector[key].to(device)
                
                # Apply scale_svd to 2D matrices only
                if len(tensor.shape) == 2:
                    # 跳过特定的层，如text_projection
                    if "text_projection" in key:
                        new_vector[key] = tensor
                        continue
                    
                    # 获取该层的统计信息
                    scaler_row = None
                    
                    # 尝试直接从统计信息中获取
                    if layer_stats is not None and key in layer_stats:
                        scaler_row = layer_stats[key].to(device)
                        print(f"Using direct statistics for {key}")
                    
                    # 如果没有找到统计信息，使用默认的全1向量
                    if scaler_row is None:
                        scaler_row = torch.ones(tensor.shape[1], device=device)
                        print(f"Using default statistics for {key}")
                    
                    # 确保缩放因子的形状正确
                    if scaler_row.shape[0] != tensor.shape[1]:
                        print(f"Warning: Scaler shape mismatch for {key}. Expected {tensor.shape[1]}, got {scaler_row.shape[0]}. Using default.")
                        scaler_row = torch.ones(tensor.shape[1], device=device)
                    
                    # Apply scale_svd to the tensor
                    try:
                        new_vector[key] = scale_svd(
                            tensor,
                            scaler_row,
                            num_iters=num_iters,
                            sparsity=sparsity,
                            rank_ratio=rank_ratio,
                            prune_level=mask_strategy
                        )
                        # 计算稀疏度
                        zero_elements = (new_vector[key] == 0).sum().item()
                        total_elements = new_vector[key].numel()
                        actual_sparsity = zero_elements / total_elements
                        print(f"Applied scale_svd to {key}: shape={new_vector[key].shape}, sparsity={actual_sparsity:.4f}")
                    except Exception as e:
                        print(f"Error applying scale_svd to {key}: {e}")
                        import traceback
                        traceback.print_exc()
                        # 如果处理失败，保持原始张量不变
                        new_vector[key] = tensor
                else:
                    # 对于非2D张量，保持不变
                    new_vector[key] = tensor
            
            # Create a new NonLinearTaskVector with the processed vectors
            processed_vectors.append(NonLinearTaskVector(model_name=task_vector.model_name, vector=new_vector))
    
    return processed_vectors


@torch.inference_mode()
def extract_scale_vector(model, merged_model, mask_rate, mask_strategy, data_id):
    """
    Extracts a scale vector from the difference between a model and a merged model,
    and applies SVD with scaling based on layer inputs.
    
    Args:
        model: The fine-tuned model
        merged_model: The merged model
        mask_rate: Sparsity rate for masking
        mask_strategy: Strategy for masking
        data_id: ID of the dataset
        
    Returns:
        vector: The processed vector
    """
    print(f"Extracting scale vector with mask_rate={mask_rate}, mask_strategy={mask_strategy}, data_id={data_id}")
    
    # Get the difference between models
    vector = model - merged_model
    
    # 尝试加载层统计信息
    try:
        # 尝试从layer_stats目录加载统计信息
        stats_dir = os.path.join('results/layer_stats')
        if os.path.exists(stats_dir):
            stats_files = [f for f in os.listdir(stats_dir) if f.endswith('_stats.pt')]
            if stats_files:
                # 使用第data_id个统计文件
                if data_id < len(stats_files):
                    stats_file = stats_files[data_id]
                    print(f"Using statistics file: {stats_file}")
                    stats_path = os.path.join(stats_dir, stats_file)
                    layer_stats = torch.load(stats_path)
                    
                    # 处理所有2D参数
                    for key, param in vector.vector.items():
                        if len(param.shape) == 2 and "text_projection" not in key:
                            print(f"Processing parameter: {key}")
                            
                            # 获取对应的激活统计信息
                            input_activation = None
                            
                            # 尝试直接从统计信息中获取
                            if key in layer_stats:
                                input_activation = layer_stats[key].to(param.device)
                                print(f"  Using direct statistics for {key}")
                            # 尝试从复合键中获取
                            else:
                                # 尝试找到匹配的键
                                for stats_key in layer_stats.keys():
                                    # 检查键是否包含当前参数名称的一部分
                                    if key.split('.')[-1] in stats_key:
                                        input_activation = layer_stats[stats_key].to(param.device)
                                        print(f"  Using matched statistics from {stats_key}")
                                        break
                            
                            # 如果没有找到统计信息，使用默认的全1向量
                            if input_activation is None:
                                input_activation = torch.ones(param.shape[1], device=param.device)
                                print(f"  Using default statistics (ones)")
                            
                            # 确保统计信息的形状正确
                            if input_activation.shape[0] != param.shape[1]:
                                print(f"  Warning: Statistics shape mismatch. Expected {param.shape[1]}, got {input_activation.shape[0]}. Using default statistics.")
                                input_activation = torch.ones(param.shape[1], device=param.device)
                            
                            # 应用scale_svd
                            try:
                                vector.vector[key] = scale_svd(
                                    param, 
                                    input_activation,
                                    sparsity=mask_rate,
                                    prune_level=mask_strategy
                                )
                                print(f"  Successfully applied scale_svd")
                            except Exception as e:
                                print(f"  Error applying scale_svd: {e}")
                                import traceback
                                traceback.print_exc()
                else:
                    print(f"Data ID {data_id} is out of range for available statistics files: {stats_files}")
            else:
                print("No statistics files found in results/layer_stats")
        else:
            print("Statistics directory not found")
            
        # 如果没有找到统计信息，尝试使用旧的方法
        if not os.path.exists(stats_dir) or not stats_files:
            print("Falling back to old method using scale_matrix.pt...")
            if os.path.exists('scale_matrix.pt'):
                layer_inputs = torch.load('scale_matrix.pt')
                dataset_names = list(layer_inputs.keys())
                print(f"Available datasets: {dataset_names}")
                
                if data_id < len(dataset_names):
                    data_name = dataset_names[data_id]
                    print(f"Using statistics for dataset: {data_name}")
                    layer_input = layer_inputs[data_name]
                    
                    # 处理所有2D参数
                    for key, param in vector.vector.items():
                        if len(param.shape) == 2 and "text_projection" not in key:
                            print(f"Processing parameter: {key}")
                            
                            # 获取对应的激活统计信息
                            input_activation = None
                            
                            # 尝试从旧格式的统计信息中获取
                            parts = key.split('.')
                            if len(parts) >= 3:
                                block_key = '.'.join(parts[:-1])
                                param_key = parts[-1]
                                
                                if block_key in layer_input and param_key in layer_input[block_key]:
                                    input_activation = layer_input[block_key][param_key].to(param.device)
                                    print(f"  Using collected statistics from old format")
                            
                            # 如果没有找到统计信息，使用默认的全1向量
                            if input_activation is None:
                                input_activation = torch.ones(param.shape[1], device=param.device)
                                print(f"  Using default statistics (ones)")
                            
                            # 确保统计信息的形状正确
                            if input_activation.shape[0] != param.shape[1]:
                                print(f"  Warning: Statistics shape mismatch. Expected {param.shape[1]}, got {input_activation.shape[0]}. Using default statistics.")
                                input_activation = torch.ones(param.shape[1], device=param.device)
                            
                            # 应用scale_svd
                            try:
                                vector.vector[key] = scale_svd(
                                    param, 
                                    input_activation,
                                    sparsity=mask_rate,
                                    prune_level=mask_strategy
                                )
                                print(f"  Successfully applied scale_svd")
                            except Exception as e:
                                print(f"  Error applying scale_svd: {e}")
                                import traceback
                                traceback.print_exc()
                else:
                    print(f"Data ID {data_id} is out of range for available datasets: {dataset_names}")
            else:
                print("scale_matrix.pt not found, using default statistics")
                
                # 使用默认的全1向量处理所有2D参数
                for key, param in vector.vector.items():
                    if len(param.shape) == 2 and "text_projection" not in key:
                        print(f"Processing parameter: {key}")
                        input_activation = torch.ones(param.shape[1], device=param.device)
                        
                        # 应用scale_svd
                        try:
                            vector.vector[key] = scale_svd(
                                param, 
                                input_activation,
                                sparsity=mask_rate,
                                prune_level=mask_strategy
                            )
                            print(f"  Successfully applied scale_svd with default statistics")
                        except Exception as e:
                            print(f"  Error applying scale_svd: {e}")
                            import traceback
                            traceback.print_exc()
    except Exception as e:
        print(f"Error in extract_scale_vector: {e}")
        import traceback
        traceback.print_exc()
        print("Continuing without scaling...")
    
    return vector


def altern_ls(weight, diag_approx, num_iters, target_rank, unstruct_sparse, prune_level="row", prune_n=0, prune_m=0):
    """
    Performs alternating optimization between low-rank and sparse components.
    
    Args:
        weight: The weight tensor to decompose
        diag_approx: Diagonal approximation for scaling
        num_iters: Number of iterations
        target_rank: Target rank for the low-rank component
        unstruct_sparse: Sparsity level for the sparse component
        prune_level: Strategy for pruning ("row" or "global")
        prune_n: N value for N:M sparsity
        prune_m: M value for N:M sparsity
        
    Returns:
        tensor: The decomposed tensor
    """
    if diag_approx.isnan().any():
        print("Outliers have NaN. Exiting!")
        raise ValueError("NaN values in diagonal approximation")

    # Scale the weight
    scaled_weight = weight * torch.sqrt(diag_approx)  # d_out x d_in
    sparse_component = torch.zeros_like(scaled_weight).to(weight.device)
    
    # Alternating optimization
    for iter_idx in range(num_iters): 
        # Apply SVD
        U, S, V = torch.linalg.svd(scaled_weight - sparse_component, full_matrices=False)
        S[target_rank:] = 0
        low_rank_component = U @ torch.diag(S) @ V
        sparse_component = scaled_weight - low_rank_component

        # Prune the weight
        W_metric = sparse_component.clone()
        W_mask = (torch.zeros_like(W_metric) == 1)  # Initialize a mask to be all False
        
        if prune_n != 0:
            print("Applying N:M Sparsity")
            # Structured n:m sparsity
            W_metric = torch.abs(W_metric)
            for ii in range(W_metric.shape[1]):
                if ii % prune_m == 0:
                    tmp = W_metric[:,ii:(ii+prune_m)].float()
                    W_mask.scatter_(1, ii+torch.topk(tmp, prune_m - prune_n, dim=1, largest=False)[1], True)
        elif prune_level == "row":
            # Row-wise pruning
            sort_res = torch.sort(torch.abs(W_metric), dim=-1, stable=True)
            indices = sort_res[1][:,:int(W_metric.shape[1] * unstruct_sparse)]
            W_mask.scatter_(1, indices, True)
        elif prune_level == "global":
            # Global pruning
            sort_res = torch.sort(torch.flatten(torch.abs(W_metric)), stable=True)
            indices = sort_res[1][:int(W_metric.numel() * unstruct_sparse)]

            W_mask = torch.flatten(W_mask)
            W_mask[indices] = True
            W_mask = torch.unflatten(W_mask, 0, (W_metric.shape[0], W_metric.shape[1]))
        else:
            raise ValueError(f"Unknown pruning level: {prune_level}")
            
        # Apply mask to sparse component
        sparse_component[W_mask] = 0
    
    # Compute final matrices
    low_rank_matrix = U[:, :target_rank] @ torch.diag(S[:target_rank]) @ V[:target_rank, :]
    low_rank_matrix = low_rank_matrix * (1/torch.sqrt(diag_approx))
    sparse_comp = sparse_component * (1/torch.sqrt(diag_approx))
    final_matrix = low_rank_matrix + sparse_comp
    
    return final_matrix


def scale_svd(
    tensor, 
    scaler_row,
    num_iters=10,
    sparsity=0.1,
    rank_ratio=0.75,
    prune_level="row",
    prune_n=0,
    prune_m=0,
):
    """
    Applies low-rank plus sparse decomposition to a tensor.
    
    Args:
        tensor: The tensor to decompose
        scaler_row: Scaling factor for each row
        num_iters: Number of iterations for alternating optimization
        sparsity: Sparsity level (0.0-1.0)
        rank_ratio: Ratio for the low-rank component
        prune_level: Strategy for pruning
        prune_n: N value for N:M sparsity
        prune_m: M value for N:M sparsity
        
    Returns:
        tensor: The decomposed tensor
    """
    # 确保张量和缩放因子在同一设备上
    device = tensor.device
    dense_alloc = 1 - sparsity
    diag_approx = scaler_row.clone().to(device)
    orig_weight = tensor.clone().detach().float().to(device)
    d_out = orig_weight.shape[0]
    d_in = orig_weight.shape[1]
    
    # Calculate target rank based on rank_ratio and tensor dimensions
    target_rank = max(1, int(rank_ratio * dense_alloc * (d_out*d_in)/(d_out + d_in)))
    unstruct_sparse = 1.0 - (1.0-rank_ratio)*dense_alloc
    
    # 确保目标秩不超过张量的最小维度
    target_rank = min(target_rank, min(d_out, d_in))
    
    # Apply alternating optimization
    res = altern_ls(
        orig_weight, 
        diag_approx, 
        num_iters, 
        target_rank, 
        unstruct_sparse,
        prune_level=prune_level,
        prune_n=prune_n,
        prune_m=prune_m
    )
    
    return res 