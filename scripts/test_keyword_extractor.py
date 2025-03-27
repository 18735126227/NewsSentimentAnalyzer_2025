"""
Test script for KeywordBertSumExtractor
"""

import os
import sys
import torch
import numpy as np
from transformers import BertTokenizerFast

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.bertsum.keyword_extractor import KeywordBertSumExtractor

def test_keyword_extractor():
    """Test the keyword extractor implementation"""
    print("Creating KeywordBertSumExtractor...")
    extractor = KeywordBertSumExtractor(bert_model_name="bert-base-chinese", use_sentiment=False)
    extractor.eval()
    
    test_texts = [
        "A meeting was held yesterday. The president announced a new economic policy that will affect millions of citizens. The meeting lasted for two hours.",
        
        "A major earthquake struck the coastal region this morning. Initial reports indicate minimal damage to buildings. However, at least 15 people were injured and thousands have been evacuated from the area.",
        
        "Scientists have discovered a new treatment for cancer. The treatment has shown 80% effectiveness in clinical trials. Side effects appear to be minimal compared to traditional treatments."
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\nTest text {i+1}: {text}")
        
        keywords = extractor.extract_keywords(text, top_k=5)
        print(f"Keywords: {keywords}")
        
        sentences = []
        for s in text.replace('!', '.').replace('?', '.').split('.'):
            if s.strip():
                sentences.append(s.strip())
        
        scores = extractor.score_sentences(sentences, keywords)
        print("Sentence scores:")
        for j, (sentence, score) in enumerate(zip(sentences, scores)):
            print(f"  Sentence {j+1} ({score:.4f}): {sentence}")
        
        summary = extractor.generate_summary(text, max_length=2)
        print(f"Generated summary (max_length=2): {summary}")
        
        summary_short = extractor.generate_summary(text, max_length=1)
        print(f"Generated summary (max_length=1): {summary_short}")
    
    chinese_texts = [
        "中国经济持续增长。去年GDP增长率达到6.1%。专家预测今年将继续保持稳定增长。",
        "北京举办了一场国际会议。来自50个国家的代表参加了会议。会议讨论了气候变化问题。",
        "新冠疫苗研发取得重大突破。临床试验显示疫苗有效率达到95%。预计明年初可以开始大规模接种。"
    ]
    
    for i, text in enumerate(chinese_texts):
        print(f"\n中文测试文本 {i+1}: {text}")
        
        keywords = extractor.extract_keywords(text, top_k=5)
        print(f"关键词: {keywords}")
        
        summary = extractor.generate_summary(text, max_length=2)
        print(f"生成的摘要 (max_length=2): {summary}")
        
        summary_short = extractor.generate_summary(text, max_length=1)
        print(f"生成的摘要 (max_length=1): {summary_short}")

if __name__ == "__main__":
    test_keyword_extractor()
