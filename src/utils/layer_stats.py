import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import os
from tqdm import tqdm


class WrappedOATS:
    """
    包装线性层以收集输入统计信息。
    """
    def __init__(self, module, layer_name):
        self.module = module
        self.layer_name = layer_name
        self.scaler_row = None
        self.hook = module.register_forward_pre_hook(self._hook_fn)
        self.num_samples = 0
    
    def _hook_fn(self, module, input_tensor):
        # 确保输入是张量
        if isinstance(input_tensor, tuple):
            input_tensor = input_tensor[0]
        
        # 计算行级别的统计信息（每个输入特征的方差）
        if self.scaler_row is None:
            self.scaler_row = torch.zeros(input_tensor.shape[-1], device=input_tensor.device)
        
        # 计算每个特征的方差并累积
        batch_var = torch.var(input_tensor, dim=0)
        if len(batch_var.shape) > 1:
            batch_var = batch_var.mean(dim=0)  # 对所有维度取平均，得到每个特征的方差
        
        # 更新累积统计信息
        self.scaler_row = (self.scaler_row * self.num_samples + batch_var) / (self.num_samples + 1)
        self.num_samples += 1
    
    def remove_hook(self):
        self.hook.remove()


class ViTAttentionHook:
    """
    为ViT注意力层收集输入统计信息的钩子。
    
    在ViT模型中，MultiheadAttention模块的结构如下：
    - 输入首先通过in_proj_weight进行投影，生成q, k, v
    - 然后进行注意力计算
    - 最后通过out_proj进行输出投影
    
    由于in_proj_weight不是一个单独的模块，而是MultiheadAttention的一个属性，
    我们需要通过主模块的前向钩子来收集其输入统计信息。
    
    对于out_proj，通过分析PyTorch的multi_head_attention_forward函数源码，
    我们确认MultiheadAttention模块的输出就是out_proj的输入。因此，我们使用
    MultiheadAttention的输出钩子来收集out_proj的输入统计信息。
    """
    def __init__(self, module, layer_name):
        self.module = module
        self.layer_name = layer_name
        self.in_proj_stats = None
        self.out_proj_stats = None
        self.num_samples = 0
        self.out_proj_samples = 0
        
        # 注册主要的前向钩子 - 用于收集输入到整个注意力模块的统计信息
        self.hook = module.register_forward_pre_hook(self._hook_fn)
        
        # 为整个MultiheadAttention添加一个输出钩子，用于收集out_proj的输入统计信息
        self.attn_output_hook = module.register_forward_hook(self._attn_output_hook_fn)
    
    def _hook_fn(self, module, input_tensor):
        try:
            # 确保输入是张量
            if isinstance(input_tensor, tuple):
                input_tensor = input_tensor[0]
            
            # 初始化统计信息
            if self.in_proj_stats is None:
                self.in_proj_stats = torch.zeros(input_tensor.shape[-1], device=input_tensor.device)
            
            # 计算输入投影的统计信息
            batch_var = torch.var(input_tensor, dim=0)
            if len(batch_var.shape) > 1:
                batch_var = batch_var.mean(dim=0)
            
            # 更新累积统计信息
            self.in_proj_stats = (self.in_proj_stats * self.num_samples + batch_var) / (self.num_samples + 1)
            self.num_samples += 1
        except Exception as e:
            print(f"Error in ViTAttentionHook._hook_fn for {self.layer_name}: {e}")
            import traceback
            traceback.print_exc()
    
    def _attn_output_hook_fn(self, module, input_tensor, output_tensor):
        """
        收集MultiheadAttention的输出，作为out_proj的输入统计信息。
        """
        try:
            # output_tensor是一个元组，第一个元素是注意力的输出
            if isinstance(output_tensor, tuple):
                attn_output = output_tensor[0]
                
                # 初始化统计信息（如果尚未初始化）
                if self.out_proj_stats is None:
                    feat_dim = attn_output.shape[-1]
                    self.out_proj_stats = torch.zeros(feat_dim, device=attn_output.device)
                
                # 计算方差
                batch_var = torch.var(attn_output, dim=0)
                if len(batch_var.shape) > 1:
                    batch_var = batch_var.mean(dim=0)
                
                # 更新累积统计信息
                self.out_proj_stats = (self.out_proj_stats * self.out_proj_samples + batch_var) / (self.out_proj_samples + 1)
                self.out_proj_samples += 1
        except Exception as e:
            print(f"Error in ViTAttentionHook._attn_output_hook_fn for {self.layer_name}: {e}")
            import traceback
            traceback.print_exc()
    
    def remove_hook(self):
        self.hook.remove()
        self.attn_output_hook.remove()
    
    def get_stats(self):
        """
        返回收集的统计信息。
        注意：这里的键名需要与task_vector中的键名匹配。
        """
        stats = {}
        if self.in_proj_stats is not None:
            stats["attn.in_proj_weight"] = self.in_proj_stats
        if self.out_proj_stats is not None:
            stats["attn.out_proj.weight"] = self.out_proj_stats
        return stats


class ViTMLPHook:
    """
    为ViT MLP层收集输入统计信息的钩子。
    
    在ViT模型中，MLP模块的结构如下：
    - 输入首先通过c_fc进行投影（扩展维度）
    - 然后通过GELU激活函数
    - 最后通过c_proj进行输出投影（压缩维度）
    
    我们需要分别收集c_fc和c_proj的输入统计信息。
    """
    def __init__(self, module, layer_name):
        self.module = module
        self.layer_name = layer_name
        self.c_fc_stats = None
        self.c_proj_stats = None
        self.num_samples = 0
        
        # 为整个MLP添加钩子（主要用于计数）
        self.hook = module.register_forward_pre_hook(self._hook_fn)
        
        # 为c_fc添加钩子
        if hasattr(module, 'c_fc'):
            self.c_fc_hook = module.c_fc.register_forward_pre_hook(self._c_fc_hook_fn)
        else:
            self.c_fc_hook = None
        
        # 为c_proj添加钩子
        if hasattr(module, 'c_proj'):
            self.c_proj_hook = module.c_proj.register_forward_pre_hook(self._c_proj_hook_fn)
        else:
            self.c_proj_hook = None
    
    def _hook_fn(self, module, input_tensor):
        # 这个钩子主要用于记录样本数量
        if isinstance(input_tensor, tuple):
            input_tensor = input_tensor[0]
        self.num_samples += 1
    
    def _c_fc_hook_fn(self, module, input_tensor):
        try:
            # 确保输入是张量
            if isinstance(input_tensor, tuple):
                input_tensor = input_tensor[0]
            
            # 初始化统计信息
            if self.c_fc_stats is None:
                self.c_fc_stats = torch.zeros(input_tensor.shape[-1], device=input_tensor.device)
            
            # 计算c_fc的统计信息
            batch_var = torch.var(input_tensor, dim=0)
            if len(batch_var.shape) > 1:
                batch_var = batch_var.mean(dim=0)
            
            # 更新累积统计信息
            self.c_fc_stats = (self.c_fc_stats * (self.num_samples - 1) + batch_var) / self.num_samples
        except Exception as e:
            print(f"Error in ViTMLPHook._c_fc_hook_fn for {self.layer_name}: {e}")
            import traceback
            traceback.print_exc()
    
    def _c_proj_hook_fn(self, module, input_tensor):
        try:
            # 确保输入是张量
            if isinstance(input_tensor, tuple):
                input_tensor = input_tensor[0]
            
            # 初始化统计信息
            if self.c_proj_stats is None:
                self.c_proj_stats = torch.zeros(input_tensor.shape[-1], device=input_tensor.device)
            
            # 计算c_proj的统计信息
            batch_var = torch.var(input_tensor, dim=0)
            if len(batch_var.shape) > 1:
                batch_var = batch_var.mean(dim=0)
            
            # 更新累积统计信息
            self.c_proj_stats = (self.c_proj_stats * (self.num_samples - 1) + batch_var) / self.num_samples
        except Exception as e:
            print(f"Error in ViTMLPHook._c_proj_hook_fn for {self.layer_name}: {e}")
            import traceback
            traceback.print_exc()
    
    def remove_hook(self):
        self.hook.remove()
        if self.c_fc_hook is not None:
            self.c_fc_hook.remove()
        if self.c_proj_hook is not None:
            self.c_proj_hook.remove()
    
    def get_stats(self):
        """
        返回收集的统计信息。
        注意：这里的键名需要与task_vector中的键名匹配。
        """
        stats = {}
        if self.c_fc_stats is not None:
            stats["mlp.c_fc.weight"] = self.c_fc_stats
        if self.c_proj_stats is not None:
            stats["mlp.c_proj.weight"] = self.c_proj_stats
        return stats


class ViTLayerNormHook:
    """
    为ViT LayerNorm层收集输入统计信息的钩子。
    """
    def __init__(self, module, layer_name):
        self.module = module
        self.layer_name = layer_name
        self.stats = None
        self.num_samples = 0
        self.hook = module.register_forward_pre_hook(self._hook_fn)
    
    def _hook_fn(self, module, input_tensor):
        try:
            # 确保输入是张量
            if isinstance(input_tensor, tuple):
                input_tensor = input_tensor[0]
            
            # 初始化统计信息
            if self.stats is None:
                self.stats = torch.zeros(input_tensor.shape[-1], device=input_tensor.device)
            
            # 计算输入的统计信息
            batch_var = torch.var(input_tensor, dim=0)
            if len(batch_var.shape) > 1:
                batch_var = batch_var.mean(dim=0)
            
            # 更新累积统计信息
            self.stats = (self.stats * self.num_samples + batch_var) / (self.num_samples + 1)
            self.num_samples += 1
        except Exception as e:
            print(f"Error in ViTLayerNormHook._hook_fn for {self.layer_name}: {e}")
            import traceback
            traceback.print_exc()
    
    def remove_hook(self):
        self.hook.remove()
    
    def get_stats(self):
        """
        返回收集的统计信息。
        """
        return {"weight": self.stats} if self.stats is not None else {}


class ViTConvHook:
    """
    为ViT卷积层收集输入统计信息的钩子。
    """
    def __init__(self, module, layer_name):
        self.module = module
        self.layer_name = layer_name
        self.stats = None
        self.num_samples = 0
        self.hook = module.register_forward_pre_hook(self._hook_fn)
    
    def _hook_fn(self, module, input_tensor):
        try:
            # 确保输入是张量
            if isinstance(input_tensor, tuple):
                input_tensor = input_tensor[0]
            
            # 初始化统计信息 - 对于卷积层，我们关注的是输入通道
            if self.stats is None:
                self.stats = torch.zeros(input_tensor.shape[1], device=input_tensor.device)
            
            # 计算每个通道的方差
            # 对于卷积层，我们需要在空间维度上计算方差
            batch_var = torch.var(input_tensor, dim=[0, 2, 3])
            
            # 更新累积统计信息
            self.stats = (self.stats * self.num_samples + batch_var) / (self.num_samples + 1)
            self.num_samples += 1
        except Exception as e:
            print(f"Error in ViTConvHook._hook_fn for {self.layer_name}: {e}")
            import traceback
            traceback.print_exc()
    
    def remove_hook(self):
        self.hook.remove()
    
    def get_stats(self):
        """
        返回收集的统计信息。
        """
        return {"weight": self.stats} if self.stats is not None else {}


class ViTLinearHook:
    """
    为ViT线性层收集输入统计信息的钩子。
    """
    def __init__(self, module, layer_name):
        self.module = module
        self.layer_name = layer_name
        self.stats = None
        self.num_samples = 0
        self.hook = module.register_forward_pre_hook(self._hook_fn)
    
    def _hook_fn(self, module, input_tensor):
        try:
            # 确保输入是张量
            if isinstance(input_tensor, tuple):
                input_tensor = input_tensor[0]
            
            # 初始化统计信息
            if self.stats is None:
                self.stats = torch.zeros(input_tensor.shape[-1], device=input_tensor.device)
            
            # 计算输入的统计信息
            batch_var = torch.var(input_tensor, dim=0)
            if len(batch_var.shape) > 1:
                batch_var = batch_var.mean(dim=0)
            
            # 更新累积统计信息
            self.stats = (self.stats * self.num_samples + batch_var) / (self.num_samples + 1)
            self.num_samples += 1
        except Exception as e:
            print(f"Error in ViTLinearHook._hook_fn for {self.layer_name}: {e}")
            import traceback
            traceback.print_exc()
    
    def remove_hook(self):
        self.hook.remove()
    
    def get_stats(self):
        """
        返回收集的统计信息。
        """
        return {"weight": self.stats} if self.stats is not None else {}

@torch.inference_mode()
def collect_layer_statistics(model, dataloader, num_batches=50, save_path='scale_matrix.pt'):
    """
    收集模型中所有层的输入统计信息。
    
    当config.method="collect_stats"时，此函数将收集ViT模型中所有层的输入激活值统计信息，
    并将其保存到指定路径。
    
    参数：
        model: 要收集统计信息的模型
        dataloader: 提供数据批次的DataLoader
        num_batches: 要处理的批次数
        save_path: 保存收集的统计信息的路径
        
    返回：
        将层名称映射到其输入统计信息的字典
    """
    print(f"Starting layer statistics collection for {num_batches} batches...")
    
    # 初始化用于收集统计信息的层包装器
    vit_attention_hooks = {}
    vit_mlp_hooks = {}
    vit_layernorm_hooks = {}
    vit_conv_hooks = {}
    vit_linear_hooks = {}
    
    # 为ViT模型的所有层添加钩子
    for name, module in model.named_modules():
        # 检查是否为ViT的注意力层
        if isinstance(module, nn.MultiheadAttention):
            try:
                vit_attention_hooks[name] = ViTAttentionHook(module, name)
                print(f"Added attention hook for {name}")
            except Exception as e:
                print(f"Error adding attention hook for {name}: {e}")
                import traceback
                traceback.print_exc()
        
        # 检查是否为ViT的MLP层
        elif name.endswith('.mlp'):
            try:
                vit_mlp_hooks[name] = ViTMLPHook(module, name)
                print(f"Added MLP hook for {name}")
            except Exception as e:
                print(f"Error adding MLP hook for {name}: {e}")
                import traceback
                traceback.print_exc()
        
        # 检查是否为LayerNorm层
        elif isinstance(module, nn.LayerNorm):
            try:
                vit_layernorm_hooks[name] = ViTLayerNormHook(module, name)
                print(f"Added LayerNorm hook for {name}")
            except Exception as e:
                print(f"Error adding LayerNorm hook for {name}: {e}")
                import traceback
                traceback.print_exc()
        
        # 检查是否为卷积层
        elif isinstance(module, nn.Conv2d):
            try:
                vit_conv_hooks[name] = ViTConvHook(module, name)
                print(f"Added Conv2d hook for {name}")
            except Exception as e:
                print(f"Error adding Conv2d hook for {name}: {e}")
                import traceback
                traceback.print_exc()
    
    # 处理数据集以收集统计信息
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        try:
            for i, batch in enumerate(tqdm(dataloader, desc="Collecting layer statistics")):
                if i >= num_batches:
                    break
                
                # 将批次移动到与模型相同的设备
                try:
                    if isinstance(batch, dict):
                        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                for k, v in batch.items()}
                        outputs = model(**batch)
                    else:
                        if isinstance(batch, (list, tuple)) and len(batch) > 0:
                            batch_data = batch[0]
                        else:
                            batch_data = batch
                        
                        batch_data = batch_data.to(device)
                        outputs = model(batch_data)
                except Exception as e:
                    print(f"Error processing batch {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        except Exception as e:
            print(f"Error during data processing: {e}")
            import traceback
            traceback.print_exc()
    
    # 提取统计信息并清理钩子
    print("Extracting collected statistics...")
    layer_stats = {}
    
    # 处理ViT注意力层的统计信息
    for block_name, hook in vit_attention_hooks.items():
        try:
            attn_stats = hook.get_stats()
            for key, value in attn_stats.items():
                if 'in_proj_weight' in key:
                    key = 'in_proj_weight'
                elif 'out_proj.weight' in key:
                    key = 'out_proj.weight'
                key = '.'.join([block_name, key])
                layer_stats[key] = value.detach().cpu()
            
            hook.remove_hook()
        except Exception as e:
            print(f"Error processing attention hook for {block_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # 处理ViT MLP层的统计信息
    for block_name, hook in vit_mlp_hooks.items():
        try:
            mlp_stats = hook.get_stats()
            for key, value in mlp_stats.items():
                if 'c_fc.weight' in key:
                    key = 'c_fc.weight'
                elif 'c_proj.weight' in key:
                    key = 'c_proj.weight'
                key = '.'.join([block_name, key])
                layer_stats[key] = value.detach().cpu()
            hook.remove_hook()
        except Exception as e:
            print(f"Error processing MLP hook for {block_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # 处理LayerNorm层的统计信息
    for block_name, hook in vit_layernorm_hooks.items():
        try:
            ln_stats = hook.get_stats()
            for key, value in ln_stats.items():
                key = '.'.join([block_name, key])
                layer_stats[key] = value.detach().cpu()
            hook.remove_hook()
        except Exception as e:
            print(f"Error processing LayerNorm hook for {block_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # 处理卷积层的统计信息
    for block_name, hook in vit_conv_hooks.items():
        try:
            conv_stats = hook.get_stats()
            for key, value in conv_stats.items():
                key = '.'.join([block_name, key])
                layer_stats[key] = value.detach().cpu()
            hook.remove_hook()
        except Exception as e:
            print(f"Error processing Conv2d hook for {block_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存统计信息
    print(f"Saving statistics to {save_path}...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(layer_stats, save_path)
    print(f"Statistics saved successfully with {len(layer_stats)} entries.")
    
    return layer_stats


def get_layer_stats(model_name, dataset_name, stats_dir='results/layer_stats'):
    """
    加载特定模型和数据集的层统计信息。
    
    参数：
        model_name: 模型名称
        dataset_name: 数据集名称
        stats_dir: 包含统计信息文件的目录
        
    返回：
        将层名称映射到其输入统计信息的字典
    """
    stats_path = os.path.join(stats_dir, f"{model_name}_{dataset_name}_stats.pt")
    if not os.path.exists(stats_path):
        # 尝试使用默认的scale_matrix.pt路径
        stats_path = 'scale_matrix.pt'
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"未找到统计信息文件：{stats_path}和scale_matrix.pt")
    
    print(f"Loading layer statistics from {stats_path}")
    return torch.load(stats_path)


def is_collecting_stats(config):
    """
    检查是否处于收集统计信息模式。
    
    参数：
        config: 配置对象
        
    返回：
        如果config.method为"collect_stats"，则返回True，否则返回False
    """
    return hasattr(config, 'method') and config.method == "collect_stats" 