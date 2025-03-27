import re
from typing import Dict, Union, List
from snownlp import SnowNLP

class SentimentAnalyzer:
    """
    中文情感分析器，使用SnowNLP进行情感分析
    """
    
    def __init__(self):
        """初始化情感分析器"""
        pass
    
    def analyze(self, text: str) -> Dict[str, Union[str, float]]:
        """
        分析文本情感
        
        Args:
            text: 待分析的文本
            
        Returns:
            包含情感标签和分数的字典
        """
        if not text or len(text.strip()) == 0:
            return {"label": "neutral", "score": 0.5}
        
        try:
            s = SnowNLP(text)
            sentiment_score = s.sentiments
            
            if sentiment_score > 0.7:
                label = "positive"
            elif sentiment_score < 0.3:
                label = "negative"
            else:
                label = "neutral"
                
            return {
                "label": label,
                "score": sentiment_score
            }
        except Exception as e:
            print(f"情感分析出错: {e}")
            return {"label": "neutral", "score": 0.5}
    
    def compute_consistency(self, original_text: str, summary: str) -> Dict[str, Union[float, Dict]]:
        """
        计算原文和摘要之间的情感一致性
        
        Args:
            original_text: 原文
            summary: 摘要
            
        Returns:
            包含情感一致性指标的字典
        """
        original_sentiment = self.analyze(original_text)
        summary_sentiment = self.analyze(summary)
        
        label_consistency = 1.0 if original_sentiment["label"] == summary_sentiment["label"] else 0.0
        score_diff = abs(original_sentiment["score"] - summary_sentiment["score"])
        score_consistency = 1.0 - min(score_diff, 1.0)
        
        return {
            "label_consistency": label_consistency,
            "score_consistency": score_consistency,
            "original_sentiment": original_sentiment,
            "summary_sentiment": summary_sentiment
        }
    
    def batch_analyze(self, texts: List[str]) -> List[Dict[str, Union[str, float]]]:
        """
        批量分析文本情感
        
        Args:
            texts: 待分析的文本列表
            
        Returns:
            情感分析结果列表
        """
        return [self.analyze(text) for text in texts]
