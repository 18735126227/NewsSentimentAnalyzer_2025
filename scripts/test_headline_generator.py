"""
Test script for HeadlineGenerator
"""

import os
import sys
import torch
import numpy as np
from transformers import BertTokenizerFast
from rouge import Rouge

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.bertsum.headline_generator import HeadlineGenerator

def test_headline_generator():
    """Test the headline generator implementation"""
    print("Creating HeadlineGenerator...")
    generator = HeadlineGenerator(bert_model_name="bert-base-chinese", use_sentiment=False)
    generator.eval()
    
    chinese_texts = [
        "中国经济持续增长。去年GDP增长率达到6.1%。专家预测今年将继续保持稳定增长。",
        "北京举办了一场国际会议。来自50个国家的代表参加了会议。会议讨论了气候变化问题。",
        "新冠疫苗研发取得重大突破。临床试验显示疫苗有效率达到95%。预计明年初可以开始大规模接种。"
    ]
    
    for i, text in enumerate(chinese_texts):
        print(f"\n中文测试文本 {i+1}: {text}")
        
        headline = generator.generate_headline(text)
        print(f"生成的标题: {headline}")
    
    lcsts_examples = [
        {
            "title": "中国经济增长6.1%",
            "content": "中国经济持续增长。去年GDP增长率达到6.1%。专家预测今年将继续保持稳定增长。多个行业都表现出良好的发展势头。"
        },
        {
            "title": "新冠疫苗研发取得突破",
            "content": "新冠疫苗研发取得重大突破。临床试验显示疫苗有效率达到95%。预计明年初可以开始大规模接种。这将有助于控制疫情蔓延。"
        },
        {
            "title": "北京举办国际气候变化会议",
            "content": "北京举办了一场国际会议。来自50个国家的代表参加了会议。会议讨论了气候变化问题。与会者就减排目标达成了初步共识。"
        },
        {
            "title": "苹果发布新款iPhone",
            "content": "苹果公司今日发布了新款iPhone手机。新机型采用了全新的A15处理器。屏幕尺寸比上一代增加了0.3英寸。售价从999美元起。"
        },
        {
            "title": "中国女排获世界杯冠军",
            "content": "中国女排在世界杯决赛中以3:1战胜美国队。这是中国女排第五次获得世界杯冠军。朱婷获得了最有价值球员称号。全国球迷为此欢欣鼓舞。"
        }
    ]
    
    print("\n测试LCSTS数据集示例:")
    rouge = Rouge()
    total_rouge1 = 0
    total_rouge2 = 0
    total_rougeL = 0
    
    for i, example in enumerate(lcsts_examples):
        print(f"\n示例 {i+1}:")
        print(f"标题: {example['title']}")
        print(f"内容: {example['content']}")
        
        headline = generator.generate_headline(example['content'])
        print(f"生成的标题: {headline}")
        
        try:
            scores = rouge.get_scores(headline, example['title'])[0]
            print(f"ROUGE-1 F1: {scores['rouge-1']['f']:.4f}")
            print(f"ROUGE-2 F1: {scores['rouge-2']['f']:.4f}")
            print(f"ROUGE-L F1: {scores['rouge-l']['f']:.4f}")
            
            total_rouge1 += scores['rouge-1']['f']
            total_rouge2 += scores['rouge-2']['f']
            total_rougeL += scores['rouge-l']['f']
        except Exception as e:
            print(f"Error calculating ROUGE: {e}")
    
    avg_rouge1 = total_rouge1 / len(lcsts_examples)
    avg_rouge2 = total_rouge2 / len(lcsts_examples)
    avg_rougeL = total_rougeL / len(lcsts_examples)
    
    print("\n平均ROUGE分数:")
    print(f"ROUGE-1 F1: {avg_rouge1:.4f}")
    print(f"ROUGE-2 F1: {avg_rouge2:.4f}")
    print(f"ROUGE-L F1: {avg_rougeL:.4f}")

if __name__ == "__main__":
    test_headline_generator()
