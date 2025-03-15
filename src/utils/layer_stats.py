import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import os


class WrappedOATS:
    """
    用于收集神经网络层输入统计信息的包装类。
    
    该类包装一个层并在前向传播过程中收集有关其输入的统计信息。
    这些统计信息用于确定层中参数的重要性。
    
    参数：
        layer: 要包装的层（通常是nn.Linear）
        layer_id: 层的可选标识符
        layer_name: 层的可选名称
    """
    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]
        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0
        self.layer_id = layer_id
        self.layer_name = layer_name
        
        # 注册前向钩子以收集统计信息
        self.hook = self.layer.register_forward_hook(self._hook_fn)
    
    def _hook_fn(self, module, inp, out):
        """收集输入统计信息的前向钩子函数。"""
        self.add_batch(inp[0], out)
    
    def add_batch(self, inp, out):
        """
        将一批输入添加到统计信息中。
        
        参数：
            inp: 层的输入张量
            out: 层的输出张量
        """
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        # 使用加权平均更新运行统计信息
        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp
        self.scaler_row += torch.norm(inp.clone(), p=2, dim=1) ** 2 / self.nsamples
    
    def remove_hook(self):
        """移除前向钩子。"""
        self.hook.remove()


def collect_layer_statistics(model, dataloader, num_batches=50, save_path='scale_matrix.pt'):
    """
    收集模型中所有线性层的输入统计信息。
    
    参数：
        model: 要收集统计信息的模型
        dataloader: 提供数据批次的DataLoader
        num_batches: 要处理的批次数
        save_path: 保存收集的统计信息的路径
        
    返回：
        将层名称映射到其输入统计信息的字典
    """
    # 初始化用于收集统计信息的层包装器
    wrapped_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            wrapped_layers[name] = WrappedOATS(module, layer_name=name)
    
    # 处理数据集以收集统计信息
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            # 将批次移动到与模型相同的设备
            if isinstance(batch, dict):
                batch = {k: v.to(next(model.parameters()).device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch.items()}
                outputs = model(**batch)
            else:
                batch = batch[0].to(next(model.parameters()).device)
                outputs = model(batch)
    
    # 提取统计信息并清理钩子
    layer_stats = {}
    for name, wrapper in wrapped_layers.items():
        layer_stats[name] = wrapper.scaler_row.detach().cpu()
        wrapper.remove_hook()
    
    # 保存统计信息
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(layer_stats, save_path)
    
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
        raise FileNotFoundError(f"未找到统计信息文件：{stats_path}")
    
    return torch.load(stats_path) 