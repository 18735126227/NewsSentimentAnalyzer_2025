"""
Test script for TitleFocusedExtractor
"""

import os
import sys
import torch
import numpy as np
from transformers import BertTokenizerFast

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.bertsum.title_focused_extractor import TitleFocusedExtractor

def test_title_focused_extractor():
    """Test the title focused extractor implementation"""
    print("Creating TitleFocusedExtractor...")
    extractor = TitleFocusedExtractor(bert_model_name="bert-base-chinese", use_sentiment=False)
    extractor.eval()
    
    test_texts = [
        "A meeting was held yesterday. The president announced a new economic policy that will affect millions of citizens. The meeting lasted for two hours.",
        
        "A major earthquake struck the coastal region this morning. Initial reports indicate minimal damage to buildings. However, at least 15 people were injured and thousands have been evacuated from the area.",
        
        "Scientists have discovered a new treatment for cancer. The treatment has shown 80% effectiveness in clinical trials. Side effects appear to be minimal compared to traditional treatments."
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\nTest text {i+1}: {text}")
        
        keywords = extractor.extract_keywords(text)
        entities = extractor.extract_entities(text)
        print(f"Keywords: {keywords}")
        print(f"Entities: {entities}")
        
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
        
        keywords = extractor.extract_keywords(text)
        entities = extractor.extract_entities(text)
        print(f"关键词: {keywords}")
        print(f"实体: {entities}")
        
        summary = extractor.generate_summary(text, max_length=2)
        print(f"生成的摘要 (max_length=2): {summary}")
        
        summary_short = extractor.generate_summary(text, max_length=1)
        print(f"生成的摘要 (max_length=1): {summary_short}")
    
    lcsts_examples = [
        {
            "title": "中国经济增长6.1%",
            "content": "中国经济持续增长。去年GDP增长率达到6.1%。专家预测今年将继续保持稳定增长。多个行业都表现出良好的发展势头。"
        },
        {
            "title": "新冠疫苗研发取得突破",
            "content": "新冠疫苗研发取得重大突破。临床试验显示疫苗有效率达到95%。预计明年初可以开始大规模接种。这将有助于控制疫情蔓延。"
        }
    ]
    
    print("\n测试LCSTS数据集示例:")
    for i, example in enumerate(lcsts_examples):
        print(f"\n示例 {i+1}:")
        print(f"标题: {example['title']}")
        print(f"内容: {example['content']}")
        
        summary = extractor.generate_summary(example['content'], max_length=1)
        print(f"生成的摘要: {summary}")
        
        from rouge import Rouge
        rouge = Rouge()
        try:
            scores = rouge.get_scores(summary, example['title'])[0]
            print(f"ROUGE-1 F1: {scores['rouge-1']['f']:.4f}")
            print(f"ROUGE-2 F1: {scores['rouge-2']['f']:.4f}")
            print(f"ROUGE-L F1: {scores['rouge-l']['f']:.4f}")
        except Exception as e:
            print(f"Error calculating ROUGE: {e}")

if __name__ == "__main__":
    test_title_focused_extractor()
