
"""
测试抽取式摘要模型并计算ROUGE评分
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from rouge import Rouge
import torch
from transformers import BertTokenizerFast

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.bertsum.modeling import BertSumExtractor
from src.sentiment.analyzer import SentimentAnalyzer

def load_dataset(data_path, num_samples=1000):
    """
    加载数据集
    
    Args:
        data_path: 数据集路径
        num_samples: 样本数量
        
    Returns:
        样本列表
    """
    print(f"正在加载数据集: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    num_samples = min(num_samples, len(data))
    samples = data[:num_samples]
    print(f"已加载 {len(samples)} 条样本")
    return samples

def generate_summaries(model, tokenizer, samples, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    使用抽取式模型生成摘要
    
    Args:
        model: BertSum模型
        tokenizer: BERT分词器
        samples: 样本列表
        device: 设备
        
    Returns:
        原文和摘要的列表
    """
    results = []
    model.to(device)
    model.eval()
    
    for sample in tqdm(samples, desc="生成摘要"):
        original_text = sample['content']
        reference_summary = sample['title']
        
        sentences = [s.strip() for s in original_text.split('。') if s.strip()]
        
        inputs = tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            scores = model(**inputs)
            
        top_indices = scores.argsort(descending=True)[:3].cpu().numpy()
        top_indices = sorted(top_indices)
        
        generated_summary = '。'.join([sentences[i] for i in top_indices]) + '。'
        
        results.append({
            'original_text': original_text,
            'reference_summary': reference_summary,
            'generated_summary': generated_summary
        })
    
    return results

def calculate_rouge(results):
    """
    计算ROUGE评分
    
    Args:
        results: 结果列表
        
    Returns:
        ROUGE评分
    """
    rouge = Rouge()
    rouge_scores = []
    
    for item in tqdm(results, desc="计算ROUGE评分"):
        try:
            score = rouge.get_scores(item['generated_summary'], item['reference_summary'])[0]
            rouge_scores.append(score)
        except Exception as e:
            print(f"计算ROUGE评分时出错: {e}")
    
    avg_rouge = {
        'rouge-1': {
            'f': np.mean([score['rouge-1']['f'] for score in rouge_scores]),
            'p': np.mean([score['rouge-1']['p'] for score in rouge_scores]),
            'r': np.mean([score['rouge-1']['r'] for score in rouge_scores])
        },
        'rouge-2': {
            'f': np.mean([score['rouge-2']['f'] for score in rouge_scores]),
            'p': np.mean([score['rouge-2']['p'] for score in rouge_scores]),
            'r': np.mean([score['rouge-2']['r'] for score in rouge_scores])
        },
        'rouge-l': {
            'f': np.mean([score['rouge-l']['f'] for score in rouge_scores]),
            'p': np.mean([score['rouge-l']['p'] for score in rouge_scores]),
            'r': np.mean([score['rouge-l']['r'] for score in rouge_scores])
        }
    }
    
    return avg_rouge

def calculate_sentiment_consistency(results):
    """
    计算情感一致性
    
    Args:
        results: 结果列表
        
    Returns:
        情感一致性评分
    """
    analyzer = SentimentAnalyzer()
    consistency_scores = []
    
    for item in tqdm(results, desc="计算情感一致性"):
        try:
            consistency = analyzer.compute_consistency(
                item['original_text'], 
                item['generated_summary']
            )
            consistency_scores.append(consistency)
        except Exception as e:
            print(f"计算情感一致性时出错: {e}")
    
    avg_consistency = {
        'label_consistency': np.mean([score['label_consistency'] for score in consistency_scores]),
        'score_consistency': np.mean([score['score_consistency'] for score in consistency_scores])
    }
    
    return avg_consistency

def save_results(results, output_path):
    """
    保存结果
    
    Args:
        results: 结果列表
        output_path: 输出路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到: {output_path}")

def main(args):
    samples = load_dataset(args.data_path, args.num_samples)
    
    print(f"正在加载模型: {args.model_path}")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    model = BertSumExtractor.from_pretrained(args.model_path)
    
    results = generate_summaries(model, tokenizer, samples, args.device)
    
    save_results(results, args.output_path)
    
    rouge_scores = calculate_rouge(results)
    print("\nROUGE评分:")
    print(f"ROUGE-1: {rouge_scores['rouge-1']['f']:.4f}")
    print(f"ROUGE-2: {rouge_scores['rouge-2']['f']:.4f}")
    print(f"ROUGE-L: {rouge_scores['rouge-l']['f']:.4f}")
    
    consistency_scores = calculate_sentiment_consistency(results)
    print("\n情感一致性评分:")
    print(f"标签一致性: {consistency_scores['label_consistency']:.4f}")
    print(f"分数一致性: {consistency_scores['score_consistency']:.4f}")
    
    metrics = {
        'rouge': rouge_scores,
        'sentiment_consistency': consistency_scores
    }
    
    metrics_path = os.path.join(os.path.dirname(args.output_path), 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    print(f"评分已保存到: {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试抽取式摘要模型并计算ROUGE评分")
    
    parser.add_argument("--data_path", type=str, required=True,
                        help="数据集路径，JSON格式")
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型路径")
    parser.add_argument("--output_path", type=str, default="results/extractive_summaries.json",
                        help="输出路径")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="样本数量")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="设备")
    
    args = parser.parse_args()
    main(args)
