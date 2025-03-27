import argparse
import os
import json
import torch
import logging
from tqdm import tqdm
from typing import List, Dict, Any
from transformers import AutoTokenizer

from src.training.trainer import SummaryTrainer
from src.evaluation.evaluator import SummaryEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_test_data(file_path: str) -> List[Dict[str, str]]:
    """加载测试数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def generate_summaries(
    model_path: str,
    test_data: List[Dict[str, str]],
    model_type: str = "pegasus",
    use_sentiment: bool = False,
    max_length: int = 128,
    batch_size: int = 8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> List[Dict[str, Any]]:
    """使用模型生成摘要"""
    logger.info(f"使用模型 {model_path} 生成摘要")
    
    trainer = SummaryTrainer(
        model_name=model_path,
        model_type=model_type,
        use_sentiment=use_sentiment,
        device=device
    )
    
    results = []
    for item in tqdm(test_data, desc="生成摘要"):
        original_text = item["content"]
        reference_summary = item["title"]
        
        if use_sentiment:
            generation_result = trainer.generate_summary(
                original_text, 
                max_length=max_length,
                return_sentiment=True
            )
            if isinstance(generation_result, dict):
                generated_summary = generation_result.get("summary", "")
                sentiment_consistency = generation_result.get("sentiment_consistency", None)
            else:
                generated_summary = str(generation_result)
                sentiment_consistency = None
        else:
            generated_summary = trainer.generate_summary(
                original_text, 
                max_length=max_length
            )
            sentiment_consistency = None
        
        results.append({
            "original_text": original_text,
            "reference_summary": reference_summary,
            "generated_summary": generated_summary,
            "sentiment_consistency": sentiment_consistency
        })
    
    return results

def main(args):
    test_data = load_test_data(args.test_data)
    logger.info(f"加载了 {len(test_data)} 条测试数据")
    
    if args.num_samples > 0:
        test_data = test_data[:args.num_samples]
        logger.info(f"使用 {len(test_data)} 条测试数据进行评估")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    evaluator = SummaryEvaluator(output_dir=args.output_dir)
    
    model_results = {}
    for model_config in args.models:
        model_path, model_type, use_sentiment = model_config.split(',')
        use_sentiment = use_sentiment.lower() == 'true'
        
        model_name = os.path.basename(model_path)
        
        summaries = generate_summaries(
            model_path=model_path,
            test_data=test_data,
            model_type=model_type,
            use_sentiment=use_sentiment,
            max_length=args.max_length,
            batch_size=args.batch_size
        )
        
        summaries_path = os.path.join(args.output_dir, f"{model_name}_summaries.json")
        with open(summaries_path, 'w', encoding='utf-8') as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)
        
        original_texts = [item["original_text"] for item in summaries]
        generated_summaries = [item["generated_summary"] for item in summaries]
        
        evaluation_results = evaluator.evaluate_summaries(
            original_texts=original_texts,
            summaries=generated_summaries,
            model_name=model_name
        )
        
        model_results[model_name] = evaluation_results
    
    if len(model_results) > 1:
        evaluator.visualize_results(list(model_results.keys()))
        logger.info(f"已生成模型对比可视化结果，保存在 {os.path.join(args.output_dir, 'visualizations')}")
    
    logger.info("评估完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="中文新闻摘要评估脚本")
    
    parser.add_argument("--test_data", type=str, required=True,
                        help="测试数据路径，JSON格式")
    parser.add_argument("--num_samples", type=int, default=-1,
                        help="评估样本数量，-1表示使用全部数据")
    
    parser.add_argument("--models", type=str, nargs='+', required=True,
                        help="要评估的模型，格式：'模型路径,模型类型,是否使用情感'，例如：'models/pegasus_sentiment,pegasus,true'")
    
    parser.add_argument("--max_length", type=int, default=128,
                        help="生成摘要的最大长度")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="批次大小")
    
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="评估结果输出目录")
    
    args = parser.parse_args()
    main(args)
