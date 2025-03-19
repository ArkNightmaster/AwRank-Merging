import torch
import torch.nn as nn
from src.models.task_vectors import ImageEncoder
import argparse

def print_model_structure(model_name="ViT-B-16"):
    """
    打印ViT模型的结构，包括所有模块的类型和路径。
    
    Args:
        model_name: 模型名称，默认为"ViT-B-16"
    """
    # 创建模型
    model = ImageEncoder(model_name)
    model.eval()
    
    # 打印模型结构
    print(f"Model structure for {model_name}:")
    print("=" * 80)
    
    # 收集所有模块类型
    module_types = {}
    
    # 遍历所有模块
    for name, module in model.named_modules():
        module_type = type(module).__name__
        if module_type not in module_types:
            module_types[module_type] = []
        module_types[module_type].append(name)
    
    # 打印模块类型统计
    print("Module types summary:")
    for module_type, paths in sorted(module_types.items()):
        print(f"{module_type}: {len(paths)} instances")
    
    print("\nDetailed module paths by type:")
    print("=" * 80)
    
    # 打印每种类型的详细路径
    for module_type, paths in sorted(module_types.items()):
        if len(paths) > 0:
            print(f"\n{module_type}:")
            # 只打印前5个和最后1个路径，如果路径太多
            if len(paths) > 6:
                for path in paths[:5]:
                    print(f"  - {path}")
                print(f"  ... and {len(paths) - 5} more")
                print(f"  - {paths[-1]}")
            else:
                for path in paths:
                    print(f"  - {path}")
    
    # 特别关注Transformer块的结构
    print("\nTransformer block structure:")
    print("=" * 80)
    
    # 找到第一个Transformer块
    for name, module in model.named_modules():
        if "resblocks.0" in name:
            parent_name = name.split(".resblocks.0")[0]
            block = model
            for part in parent_name.split("."):
                if part:
                    block = getattr(block, part)
            block = block.resblocks[0]
            
            print(f"Structure of first Transformer block ({parent_name}.resblocks.0):")
            for subname, submodule in block.named_modules():
                if subname:  # Skip the empty name (the block itself)
                    print(f"  - {subname}: {type(submodule).__name__}")
            
            # 打印参数形状
            print("\nParameter shapes in first Transformer block:")
            for param_name, param in block.named_parameters():
                print(f"  - {param_name}: {param.shape}")
            
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print ViT model structure")
    parser.add_argument("--model", type=str, default="ViT-B-16", help="Model name")
    args = parser.parse_args()
    
    print_model_structure(args.model) 