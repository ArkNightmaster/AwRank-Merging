import os
import sys
import torch
import argparse
from tqdm import tqdm

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.task_vectors import ImageEncoder
from src.utils.layer_stats import collect_layer_statistics
from src.datasets import get_dataset
from src.datasets.common import get_dataloader


def test_layer_stats_collection(args):
    """
    测试增强的层统计信息收集功能。
    
    参数:
        args: 命令行参数
    """
    print(f"Testing layer statistics collection with model: {args.model}")
    
    # 创建模型
    model = ImageEncoder(args.model)
    model.eval()
    
    # 将模型移动到指定设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 创建数据集和数据加载器
    dataset = get_dataset(
        args.dataset,
        model.train_preprocess,
        args.data_dir,
        batch_size=args.batch_size,
    )
    
    dataloader = get_dataloader(
        dataset,
        is_train=True,
        args=args
    )
    
    # 创建保存目录
    stats_dir = os.path.join('results/layer_stats')
    os.makedirs(stats_dir, exist_ok=True)
    stats_path = os.path.join(stats_dir, f"{args.model}_{args.dataset}_dummy_stats.pt")
    
    # 收集层统计信息
    print(f"Collecting statistics for {args.num_batches} batches...")
    layer_stats = collect_layer_statistics(
        model,
        dataloader,
        num_batches=args.num_batches,
        save_path=stats_path
    )
    
    # 打印统计信息摘要
    print(f"\nStatistics summary:")
    print(f"Total number of layers with statistics: {len(layer_stats)}")
    
    # 按层类型分类统计信息
    layer_types = {
        'attention': 0,
        'mlp': 0,
        'layernorm': 0,
        'conv': 0,
        'linear': 0,
        'other': 0
    }
    
    for key in layer_stats.keys():
        if 'attn' in key:
            layer_types['attention'] += 1
        elif 'mlp' in key:
            layer_types['mlp'] += 1
        elif 'ln' in key or 'layernorm' in key:
            layer_types['layernorm'] += 1
        elif 'conv' in key:
            layer_types['conv'] += 1
        elif 'weight' in key and not any(t in key for t in ['attn', 'mlp', 'ln', 'layernorm', 'conv']):
            layer_types['linear'] += 1
        else:
            layer_types['other'] += 1
    
    print("\nStatistics by layer type:")
    for layer_type, count in layer_types.items():
        print(f"  {layer_type}: {count}")
    
    # 打印一些示例统计信息
    print("\nSample statistics (first 5):")
    for i, (key, value) in enumerate(layer_stats.items()):
        if i >= 5:
            break
        print(f"  {key}: shape={value.shape}, mean={value.mean().item():.6f}, std={value.std().item():.6f}")
    
    print(f"\nStatistics saved to {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test layer statistics collection")
    parser.add_argument("--model", type=str, default="ViT-B-16", help="Model name")
    parser.add_argument("--dataset", type=str, default="DTD", help="Dataset name")
    parser.add_argument("--data-dir", type=str, default="datasets", help="Data directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-batches", type=int, default=10, help="Number of batches to process")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    test_layer_stats_collection(args) 