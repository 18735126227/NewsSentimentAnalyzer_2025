import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Any
import os
import json
from tqdm import tqdm
from rouge import Rouge
from src.sentiment.analyzer import SentimentAnalyzer

class SummaryEvaluator:
    """摘要评估器，计算Rouge指标和情感一致性"""
    
    def __init__(self, output_dir: str = "results"):
        """
        初始化评估器
        
        Args:
            output_dir: 输出结果目录
        """
        self.rouge = Rouge()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def evaluate_summaries(self, original_texts: List[str], summaries: List[str], model_name: str) -> Dict[str, Any]:
        """
        评估摘要质量和情感一致性
        
        Args:
            original_texts: 原文列表
            summaries: 摘要列表
            model_name: 模型名称，用于结果标识
            
        Returns:
            包含评估结果的字典
        """
        assert len(original_texts) == len(summaries), "原文和摘要数量必须相同"
        
        rouge_scores = []
        for orig, summ in tqdm(zip(original_texts, summaries), total=len(original_texts), desc="计算Rouge指标"):
            try:
                score = self.rouge.get_scores(summ, orig)[0]
                rouge_scores.append(score)
            except Exception as e:
                print(f"计算Rouge指标时出错: {e}")
                rouge_scores.append({
                    'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                    'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                    'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}
                })
        
        sentiment_scores = []
        for orig, summ in tqdm(zip(original_texts, summaries), total=len(original_texts), desc="计算情感一致性"):
            try:
                sentiment_result = self.sentiment_analyzer.compute_consistency(orig, summ)
                sentiment_scores.append(sentiment_result)
            except Exception as e:
                print(f"计算情感一致性时出错: {e}")
                sentiment_scores.append({
                    'label_consistency': 0.0,
                    'score_consistency': 0.0,
                    'original_sentiment': {'score': 0.0, 'label': 'neutral'},
                    'summary_sentiment': {'score': 0.0, 'label': 'neutral'}
                })
        
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
        
        avg_sentiment = {
            'label_consistency': np.mean([score['label_consistency'] for score in sentiment_scores]),
            'score_consistency': np.mean([score['score_consistency'] for score in sentiment_scores])
        }
        
        results = {
            'model_name': model_name,
            'sample_count': len(original_texts),
            'rouge_scores': avg_rouge,
            'sentiment_consistency': avg_sentiment,
            'detailed_scores': {
                'rouge': rouge_scores,
                'sentiment': sentiment_scores
            }
        }
        
        self._save_results(results, model_name)
        
        return results
    
    def _save_results(self, results: Dict[str, Any], model_name: str):
        """保存评估结果到文件"""
        model_dir = os.path.join(self.output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        with open(os.path.join(model_dir, 'evaluation_results.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        rouge_df = pd.DataFrame({
            'rouge-1-f': [results['rouge_scores']['rouge-1']['f']],
            'rouge-1-p': [results['rouge_scores']['rouge-1']['p']],
            'rouge-1-r': [results['rouge_scores']['rouge-1']['r']],
            'rouge-2-f': [results['rouge_scores']['rouge-2']['f']],
            'rouge-2-p': [results['rouge_scores']['rouge-2']['p']],
            'rouge-2-r': [results['rouge_scores']['rouge-2']['r']],
            'rouge-l-f': [results['rouge_scores']['rouge-l']['f']],
            'rouge-l-p': [results['rouge_scores']['rouge-l']['p']],
            'rouge-l-r': [results['rouge_scores']['rouge-l']['r']],
            'sentiment_label_consistency': [results['sentiment_consistency']['label_consistency']],
            'sentiment_score_consistency': [results['sentiment_consistency']['score_consistency']]
        })
        rouge_df.to_csv(os.path.join(model_dir, 'metrics.csv'), index=False)
    
    def visualize_results(self, model_names: List[str]):
        """
        可视化多个模型的评估结果对比
        
        Args:
            model_names: 模型名称列表
        """
        results = []
        
        for model_name in model_names:
            result_path = os.path.join(self.output_dir, model_name, 'evaluation_results.json')
            if os.path.exists(result_path):
                with open(result_path, 'r', encoding='utf-8') as f:
                    model_results = json.load(f)
                    results.append(model_results)
            else:
                print(f"警告: 找不到模型 {model_name} 的评估结果")
        
        if not results:
            print("没有找到任何评估结果，无法生成可视化")
            return
        
        vis_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        rouge_1_f = [r['rouge_scores']['rouge-1']['f'] for r in results]
        rouge_2_f = [r['rouge_scores']['rouge-2']['f'] for r in results]
        rouge_l_f = [r['rouge_scores']['rouge-l']['f'] for r in results]
        sentiment_label = [r['sentiment_consistency']['label_consistency'] for r in results]
        sentiment_score = [r['sentiment_consistency']['score_consistency'] for r in results]
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        x = np.arange(len(model_names))
        width = 0.25
        
        plt.bar(x - width, rouge_1_f, width, label='Rouge-1-F1')
        plt.bar(x, rouge_2_f, width, label='Rouge-2-F1')
        plt.bar(x + width, rouge_l_f, width, label='Rouge-L-F1')
        
        plt.xlabel('模型')
        plt.ylabel('Rouge F1分数')
        plt.title('不同模型的Rouge指标对比')
        plt.xticks(x, model_names)
        plt.legend()
        
        plt.subplot(2, 1, 2)
        
        plt.bar(x - width/2, sentiment_label, width, label='标签一致性')
        plt.bar(x + width/2, sentiment_score, width, label='分数一致性')
        
        plt.xlabel('模型')
        plt.ylabel('情感一致性分数')
        plt.title('不同模型的情感一致性对比')
        plt.xticks(x, model_names)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'model_comparison.png'))
        plt.close()
        
        metrics = {
            'Rouge-1-F1': rouge_1_f,
            'Rouge-2-F1': rouge_2_f,
            'Rouge-L-F1': rouge_l_f,
            '情感标签一致性': sentiment_label,
            '情感分数一致性': sentiment_score
        }
        
        df = pd.DataFrame(metrics, index=model_names)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(df, annot=True, cmap='YlGnBu', fmt='.3f')
        plt.title('模型性能指标热力图')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'metrics_heatmap.png'))
        plt.close()
        
        print(f"可视化结果已保存到 {vis_dir}")
