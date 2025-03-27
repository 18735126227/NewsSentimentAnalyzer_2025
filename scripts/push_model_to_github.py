
"""
将训练好的模型推送到GitHub
"""

import os
import argparse
import subprocess
import shutil
import json
from datetime import datetime

def create_model_release_dir(model_dir, release_dir):
    """
    创建模型发布目录
    
    Args:
        model_dir: 模型目录
        release_dir: 发布目录
    """
    os.makedirs(release_dir, exist_ok=True)
    
    best_model_dir = None
    for subdir in os.listdir(model_dir):
        if subdir.startswith('best_sentiment_checkpoint_epoch'):
            best_model_dir = os.path.join(model_dir, subdir)
            break
    
    if best_model_dir is None:
        for subdir in os.listdir(model_dir):
            if subdir.startswith('best_loss_checkpoint_epoch'):
                best_model_dir = os.path.join(model_dir, subdir)
                break
    
    if best_model_dir is None:
        raise ValueError(f"在 {model_dir} 中未找到最佳模型")
    
    for file in os.listdir(best_model_dir):
        src_file = os.path.join(best_model_dir, file)
        dst_file = os.path.join(release_dir, file)
        shutil.copy2(src_file, dst_file)
    
    for file in os.listdir(model_dir):
        if file.endswith('.json'):
            src_file = os.path.join(model_dir, file)
            dst_file = os.path.join(release_dir, file)
            shutil.copy2(src_file, dst_file)
    
    print(f"模型文件已复制到 {release_dir}")

def create_model_card(model_dir, release_dir):
    """
    创建模型卡片
    
    Args:
        model_dir: 模型目录
        release_dir: 发布目录
    """
    config_path = os.path.join(model_dir, 'training_config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    metrics = {}
    latest_epoch = 0
    for file in os.listdir(model_dir):
        if file.startswith('metrics_epoch_'):
            epoch = int(file.split('_')[-1].split('.')[0])
            if epoch > latest_epoch:
                latest_epoch = epoch
                metrics_path = os.path.join(model_dir, file)
    
    if metrics_path:
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
    
    model_card = f"""# BertSum抽取式摘要模型


这是一个基于BERT的中文抽取式摘要模型，使用多任务学习框架同时优化摘要生成和情感分类任务，以保证生成的摘要与原文具有一致的情感倾向。


- 预训练模型: {config.get('model_name', 'bert-base-chinese')}
- 数据集: LCSTS (中文短文本摘要数据集)
- 批次大小: {config.get('batch_size', 4)}
- 学习率: {config.get('learning_rate', 2e-5)}
- 训练轮数: {config.get('num_epochs', 2)}
- 情感一致性训练: {'启用' if config.get('use_sentiment', True) else '禁用'}
- 情感损失权重: {config.get('sentiment_loss_weight', 1.0)}



- 损失: {metrics.get('train', {}).get('loss', 'N/A')}
- 情感准确率: {metrics.get('train', {}).get('sentiment_accuracy', 'N/A')}
- 情感F1: {metrics.get('train', {}).get('sentiment_f1', 'N/A')}


- 损失: {metrics.get('eval', {}).get('eval_loss', 'N/A')}
- 情感准确率: {metrics.get('eval', {}).get('sentiment_accuracy', 'N/A')}
- 情感F1: {metrics.get('eval', {}).get('sentiment_f1', 'N/A')}


```python
from models.bertsum.extractor import BertSumExtractor

model = BertSumExtractor.from_pretrained("models/bertsum_model")

summary = model.generate_summary(text, max_length=3)
```


{datetime.now().strftime('%Y-%m-%d')}
"""
    
    model_card_path = os.path.join(release_dir, 'README.md')
    with open(model_card_path, 'w', encoding='utf-8') as f:
        f.write(model_card)
    
    print(f"模型卡片已保存到 {model_card_path}")

def push_to_github(release_dir, branch_name, commit_message):
    """
    推送到GitHub
    
    Args:
        release_dir: 发布目录
        branch_name: 分支名称
        commit_message: 提交信息
    """
    subprocess.run(['git', 'add', release_dir], check=True)
    
    subprocess.run(['git', 'commit', '-m', commit_message], check=True)
    
    subprocess.run(['git', 'push', 'origin', branch_name], check=True)
    
    print(f"模型已推送到GitHub分支 {branch_name}")

def main(args):
    create_model_release_dir(args.model_dir, args.release_dir)
    
    create_model_card(args.model_dir, args.release_dir)
    
    if args.push:
        push_to_github(args.release_dir, args.branch_name, args.commit_message)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将训练好的模型推送到GitHub")
    
    parser.add_argument("--model_dir", type=str, required=True,
                        help="模型目录")
    parser.add_argument("--release_dir", type=str, required=True,
                        help="发布目录")
    parser.add_argument("--branch_name", type=str, default="migration-to-new-owner",
                        help="分支名称")
    parser.add_argument("--commit_message", type=str, default="Add trained BertSum model with sentiment consistency",
                        help="提交信息")
    parser.add_argument("--push", action="store_true",
                        help="是否推送到GitHub")
    
    args = parser.parse_args()
    main(args)
