
"""
创建小型数据集用于快速训练和测试
"""

import json
import os
import argparse

def create_small_dataset(input_path, output_path, num_samples=1000):
    """
    从大型数据集中提取指定数量的样本创建小型数据集
    
    Args:
        input_path: 输入数据集路径
        output_path: 输出数据集路径
        num_samples: 样本数量
    """
    print(f"从 {input_path} 加载数据集...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"数据集大小: {len(data)}")
    
    small_data = data[:num_samples]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"保存 {len(small_data)} 个样本到 {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(small_data, f, ensure_ascii=False, indent=2)
    
    print("完成!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="创建小型数据集用于快速训练和测试")
    
    parser.add_argument("--input_path", type=str, required=True,
                        help="输入数据集路径")
    parser.add_argument("--output_path", type=str, required=True,
                        help="输出数据集路径")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="样本数量")
    
    args = parser.parse_args()
    create_small_dataset(args.input_path, args.output_path, args.num_samples)
