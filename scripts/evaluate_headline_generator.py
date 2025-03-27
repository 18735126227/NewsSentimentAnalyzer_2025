"""
Evaluation script for HeadlineGenerator using ROUGE metrics
"""

import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
from rouge import Rouge

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.bertsum.headline_generator import HeadlineGenerator
from src.sentiment.analyzer import SentimentAnalyzer

def load_dataset(file_path, max_samples=1000):
    """Load dataset from JSON file"""
    print(f"Loading dataset from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if max_samples > 0 and len(data) > max_samples:
        data = data[:max_samples]
    
    print(f"Loaded {len(data)} samples")
    return data

def evaluate_generator(generator, dataset, output_file, max_samples=1000):
    """Evaluate generator on dataset"""
    print("Evaluating HeadlineGenerator...")
    
    rouge = Rouge()
    
    sentiment_analyzer = SentimentAnalyzer()
    
    results = []
    rouge_scores = []
    sentiment_consistency = []
    
    for i, sample in enumerate(tqdm(dataset[:max_samples])):
        content = sample.get('content', '')
        reference = sample.get('title', '')
        
        if not content or not reference:
            continue
        
        headline = generator.generate_headline(content)
        
        try:
            scores = rouge.get_scores(headline, reference)[0]
            rouge_scores.append(scores)
        except Exception as e:
            print(f"Error calculating ROUGE for sample {i}: {e}")
            continue
        
        try:
            content_sentiment = sentiment_analyzer.analyze(content)
            headline_sentiment = sentiment_analyzer.analyze(headline)
            
            label_consistency = 1 if content_sentiment['label'] == headline_sentiment['label'] else 0
            
            score_consistency = 1 - abs(content_sentiment['score'] - headline_sentiment['score'])
            
            sentiment_consistency.append({
                'label_consistency': label_consistency,
                'score_consistency': score_consistency
            })
        except Exception as e:
            print(f"Error calculating sentiment for sample {i}: {e}")
            continue
        
        results.append({
            'content': content,
            'reference': reference,
            'headline': headline,
            'rouge': scores,
            'sentiment_consistency': {
                'label_consistency': label_consistency,
                'score_consistency': score_consistency
            }
        })
    
    avg_rouge = {
        'rouge-1': {
            'f': np.mean([s['rouge-1']['f'] for s in rouge_scores]),
            'p': np.mean([s['rouge-1']['p'] for s in rouge_scores]),
            'r': np.mean([s['rouge-1']['r'] for s in rouge_scores])
        },
        'rouge-2': {
            'f': np.mean([s['rouge-2']['f'] for s in rouge_scores]),
            'p': np.mean([s['rouge-2']['p'] for s in rouge_scores]),
            'r': np.mean([s['rouge-2']['r'] for s in rouge_scores])
        },
        'rouge-l': {
            'f': np.mean([s['rouge-l']['f'] for s in rouge_scores]),
            'p': np.mean([s['rouge-l']['p'] for s in rouge_scores]),
            'r': np.mean([s['rouge-l']['r'] for s in rouge_scores])
        }
    }
    
    avg_sentiment_consistency = {
        'label_consistency': np.mean([s['label_consistency'] for s in sentiment_consistency]),
        'score_consistency': np.mean([s['score_consistency'] for s in sentiment_consistency])
    }
    
    print("\nEvaluation Results:")
    print(f"Number of samples: {len(results)}")
    print("\nAverage ROUGE Scores:")
    print(f"ROUGE-1 F1: {avg_rouge['rouge-1']['f']:.4f}")
    print(f"ROUGE-2 F1: {avg_rouge['rouge-2']['f']:.4f}")
    print(f"ROUGE-L F1: {avg_rouge['rouge-l']['f']:.4f}")
    print("\nAverage Sentiment Consistency:")
    print(f"Label Consistency: {avg_sentiment_consistency['label_consistency']:.4f}")
    print(f"Score Consistency: {avg_sentiment_consistency['score_consistency']:.4f}")
    
    metrics_file = os.path.join(os.path.dirname(output_file), 'headline_evaluation_metrics.json')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump({
            'rouge': avg_rouge,
            'sentiment_consistency': avg_sentiment_consistency
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(f"Metrics saved to {metrics_file}")
    
    return avg_rouge, avg_sentiment_consistency

def main():
    """Main function"""
    os.makedirs('results/evaluation', exist_ok=True)
    
    dataset_path = 'data/lcsts/lcsts_data.json'
    dataset = load_dataset(dataset_path, max_samples=1000)
    
    generator = HeadlineGenerator(bert_model_name="bert-base-chinese", use_sentiment=False)
    generator.eval()
    
    output_file = 'results/evaluation/headline_evaluation_results.json'
    evaluate_generator(generator, dataset, output_file, max_samples=1000)

if __name__ == "__main__":
    main()
