from torch.utils.data import Dataset
import json
import torch
from typing import Dict, List, Optional
import os
from src.sentiment.analyzer import SentimentAnalyzer

class SentimentAwareLCSTSDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_source_length: int = 512, max_target_length: int = 128, cache_dir: Optional[str] = None):
        """
        初始化带情感标注的LCSTS数据集
        Args:
            data_path: LCSTS数据集路径
            tokenizer: HuggingFace分词器
            max_source_length: 源文本最大长度
            max_target_length: 目标摘要最大长度
            cache_dir: 缓存情感标注的目录
        """
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.analyzer = SentimentAnalyzer()
        
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.sentiment_cache_path = os.path.join(cache_dir, "sentiment_labels.json") if cache_dir else None
        if self.sentiment_cache_path and os.path.exists(self.sentiment_cache_path):
            with open(self.sentiment_cache_path, 'r', encoding='utf-8') as f:
                self.sentiment_labels = json.load(f)
        else:
            self._generate_sentiment_labels()
    
    def _generate_sentiment_labels(self):
        """生成情感标注并缓存"""
        self.sentiment_labels = []
        
        for item in self.data:
            content_sentiment = self.analyzer.analyze(item['content'])
            title_sentiment = self.analyzer.analyze(item['title'])
            
            sentiment_map = {"negative": 0, "neutral": 1, "positive": 2}
            
            self.sentiment_labels.append({
                "content_sentiment": sentiment_map[content_sentiment["label"]],
                "title_sentiment": sentiment_map[title_sentiment["label"]]
            })
            
        if self.sentiment_cache_path:
            os.makedirs(os.path.dirname(self.sentiment_cache_path), exist_ok=True)
            with open(self.sentiment_cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.sentiment_labels, f)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        sentiment_item = self.sentiment_labels[idx]
        
        source_encoding = self.tokenizer(
            item['content'],
            max_length=self.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            item['title'],
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': source_encoding['input_ids'].squeeze(),
            'attention_mask': source_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze(),
            'decoder_attention_mask': target_encoding['attention_mask'].squeeze(),
            'content_sentiment': torch.tensor(sentiment_item['content_sentiment'], dtype=torch.long),
            'title_sentiment': torch.tensor(sentiment_item['title_sentiment'], dtype=torch.long)
        }
