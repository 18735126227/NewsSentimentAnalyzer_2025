
"""
训练BertSum抽取式摘要模型，支持情感一致性
"""

import torch
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizerFast, AdamW, get_linear_schedule_with_warmup, set_seed
import argparse
import os
import json
import logging
from tqdm import tqdm
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.bertsum.modeling import BertSumExt
from models.bertsum.extractor import BertSumExtractor
from src.data.sentiment_dataset import SentimentAwareLCSTSDataset
from src.sentiment.analyzer import SentimentAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BertSumTrainer:
    """BertSum训练器"""
    
    def __init__(
        self,
        model_name="bert-base-chinese",
        learning_rate=2e-5,
        weight_decay=0.01,
        use_sentiment=True,
        sentiment_loss_weight=1.0,
        device=None
    ):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_sentiment = use_sentiment
        self.sentiment_loss_weight = sentiment_loss_weight
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"初始化BertSum训练器，使用设备: {self.device}")
        logger.info(f"使用预训练模型: {model_name}")
        logger.info(f"情感一致性训练: {'启用' if use_sentiment else '禁用'}")
        
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        
        self.model = BertSumExt(
            bert_model_name=model_name,
            dropout=0.1,
            use_sentiment=use_sentiment
        )
        self.model.to(self.device)
        
        self.optimizer = None
        self.scheduler = None
    
    def prepare_optimizer(self, num_training_steps, warmup_steps=0):
        """准备优化器和学习率调度器"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
    
    def train_epoch(self, dataloader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        sentiment_preds = []
        sentiment_labels = []
        
        progress_bar = tqdm(dataloader, desc="训练")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            sentence_mask = (input_ids == self.tokenizer.cls_token_id).float()
            
            batch_size = input_ids.size(0)
            max_sentences = sentence_mask.sum(dim=1).max().int().item()
            
            labels = torch.zeros(batch_size, max_sentences, device=self.device)
            
            for i in range(batch_size):
                num_sentences = min(3, int(sentence_mask[i].sum().item()))
                labels[i, :num_sentences] = 1.0
            
            if self.use_sentiment and "content_sentiment" in batch:
                sentiment_labels_batch = batch["content_sentiment"].to(self.device)
            else:
                sentiment_labels_batch = None
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                sentence_mask=sentence_mask,
                labels=labels,
                sentiment_labels=sentiment_labels_batch
            )
            
            loss = outputs["loss"]
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            
            if self.use_sentiment and outputs["sentiment_logits"] is not None and sentiment_labels_batch is not None:
                sentiment_logits = outputs["sentiment_logits"]
                sentiment_preds.extend(sentiment_logits.argmax(dim=-1).cpu().numpy())
                sentiment_labels.extend(sentiment_labels_batch.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        
        metrics = {"loss": avg_loss}
        if self.use_sentiment and sentiment_labels:
            sentiment_accuracy = accuracy_score(sentiment_labels, sentiment_preds)
            sentiment_f1 = f1_score(sentiment_labels, sentiment_preds, average='weighted')
            metrics.update({
                "sentiment_accuracy": sentiment_accuracy,
                "sentiment_f1": sentiment_f1
            })
        
        return metrics
    
    def evaluate(self, dataloader):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        all_sent_scores = []
        all_labels = []
        sentiment_preds = []
        sentiment_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="评估"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                sentence_mask = (input_ids == self.tokenizer.cls_token_id).float()
                
                batch_size = input_ids.size(0)
                max_sentences = sentence_mask.sum(dim=1).max().int().item()
                
                labels = torch.zeros(batch_size, max_sentences, device=self.device)
                
                for i in range(batch_size):
                    num_sentences = min(3, int(sentence_mask[i].sum().item()))
                    labels[i, :num_sentences] = 1.0
                
                if self.use_sentiment and "content_sentiment" in batch:
                    sentiment_labels_batch = batch["content_sentiment"].to(self.device)
                else:
                    sentiment_labels_batch = None
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    sentence_mask=sentence_mask,
                    labels=labels,
                    sentiment_labels=sentiment_labels_batch
                )
                
                loss = outputs["loss"]
                sent_scores = outputs["sent_scores"]
                
                total_loss += loss.item()
                
                all_sent_scores.extend(sent_scores.cpu().numpy())
                all_labels.extend(labels.view(-1).cpu().numpy())
                
                if self.use_sentiment and outputs["sentiment_logits"] is not None and sentiment_labels_batch is not None:
                    sentiment_logits = outputs["sentiment_logits"]
                    sentiment_preds.extend(sentiment_logits.argmax(dim=-1).cpu().numpy())
                    sentiment_labels.extend(sentiment_labels_batch.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        
        metrics = {"eval_loss": avg_loss}
        
        if self.use_sentiment and sentiment_labels:
            sentiment_accuracy = accuracy_score(sentiment_labels, sentiment_preds)
            sentiment_f1 = f1_score(sentiment_labels, sentiment_preds, average='weighted')
            metrics.update({
                "sentiment_accuracy": sentiment_accuracy,
                "sentiment_f1": sentiment_f1
            })
        
        return metrics
    
    def save_checkpoint(self, output_dir):
        """保存检查点"""
        os.makedirs(output_dir, exist_ok=True)
        
        model_path = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(self.model.state_dict(), model_path)
        
        self.tokenizer.save_pretrained(output_dir)
        
        extractor = BertSumExtractor(
            bert_model_name=self.model_name,
            use_sentiment=self.use_sentiment
        )
        extractor.model.load_state_dict(self.model.state_dict())
        extractor_path = os.path.join(output_dir, "extractor.bin")
        torch.save(extractor.state_dict(), extractor_path)
        
        logger.info(f"模型已保存到: {output_dir}")
        
        return output_dir

def prepare_dataset(data_path, tokenizer, max_source_length, max_target_length, use_sentiment, cache_dir=None):
    """准备数据集"""
    logger.info(f"加载数据集: {data_path}")
    
    if use_sentiment:
        dataset = SentimentAwareLCSTSDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            cache_dir=cache_dir
        )
    else:
        from src.data.dataset import LCSTSDataset
        dataset = LCSTSDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_source_length=max_source_length,
            max_target_length=max_target_length
        )
    
    logger.info(f"数据集大小: {len(dataset)}")
    return dataset

def main(args):
    set_seed(args.seed)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"bertsum_{'sentiment' if args.use_sentiment else 'base'}_{timestamp}"
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    config = vars(args)
    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    trainer = BertSumTrainer(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_sentiment=args.use_sentiment,
        sentiment_loss_weight=args.sentiment_loss_weight
    )
    
    dataset = prepare_dataset(
        data_path=args.data_path,
        tokenizer=trainer.tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        use_sentiment=args.use_sentiment,
        cache_dir=args.cache_dir
    )
    
    if args.small_dataset_size > 0:
        dataset_size = min(args.small_dataset_size, len(dataset))
        indices = torch.randperm(len(dataset))[:dataset_size].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)
        logger.info(f"使用小数据集进行训练，大小: {len(dataset)}")
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(val_dataset)}")
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    num_training_steps = len(train_dataloader) * args.num_epochs
    warmup_steps = int(num_training_steps * 0.1)
    trainer.prepare_optimizer(num_training_steps, warmup_steps)
    
    best_loss = float('inf')
    best_sentiment_accuracy = 0.0
    
    for epoch in range(args.num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        train_metrics = trainer.train_epoch(train_dataloader)
        logger.info(f"训练损失: {train_metrics['loss']:.4f}")
        
        if "sentiment_accuracy" in train_metrics:
            logger.info(f"训练情感准确率: {train_metrics['sentiment_accuracy']:.4f}")
            logger.info(f"训练情感F1: {train_metrics['sentiment_f1']:.4f}")
        
        eval_metrics = trainer.evaluate(val_dataloader)
        eval_loss = eval_metrics['eval_loss']
        logger.info(f"验证损失: {eval_loss:.4f}")
        
        if "sentiment_accuracy" in eval_metrics:
            sentiment_accuracy = eval_metrics["sentiment_accuracy"]
            logger.info(f"验证情感准确率: {sentiment_accuracy:.4f}")
            logger.info(f"验证情感F1: {eval_metrics['sentiment_f1']:.4f}")
        else:
            sentiment_accuracy = 0.0
        
        metrics_path = os.path.join(output_dir, f"metrics_epoch_{epoch+1}.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump({
                "epoch": epoch + 1,
                "train": train_metrics,
                "eval": eval_metrics
            }, f, ensure_ascii=False, indent=2)
        
        if eval_loss < best_loss:
            best_loss = eval_loss
            checkpoint_dir = os.path.join(
                output_dir,
                f"best_loss_checkpoint_epoch_{epoch+1}"
            )
            trainer.save_checkpoint(checkpoint_dir)
            logger.info(f"保存最佳损失模型到 {checkpoint_dir}")
        
        if args.use_sentiment and sentiment_accuracy > best_sentiment_accuracy:
            best_sentiment_accuracy = sentiment_accuracy
            sentiment_checkpoint_dir = os.path.join(
                output_dir,
                f"best_sentiment_checkpoint_epoch_{epoch+1}"
            )
            trainer.save_checkpoint(sentiment_checkpoint_dir)
            logger.info(f"保存最佳情感模型到 {sentiment_checkpoint_dir}")
        
        if args.save_every_epoch:
            epoch_checkpoint_dir = os.path.join(
                output_dir,
                f"checkpoint_epoch_{epoch+1}"
            )
            trainer.save_checkpoint(epoch_checkpoint_dir)
            logger.info(f"保存Epoch {epoch+1}检查点到 {epoch_checkpoint_dir}")
    
    logger.info(f"训练完成！最佳验证损失: {best_loss:.4f}")
    if args.use_sentiment:
        logger.info(f"最佳情感准确率: {best_sentiment_accuracy:.4f}")
    
    final_checkpoint_dir = os.path.join(output_dir, "final_model")
    trainer.save_checkpoint(final_checkpoint_dir)
    logger.info(f"保存最终模型到 {final_checkpoint_dir}")
    
    return final_checkpoint_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BertSum抽取式摘要训练脚本，支持情感一致性")
    
    parser.add_argument("--model_name", type=str, default="bert-base-chinese",
                        help="预训练模型名称或路径")
    
    parser.add_argument("--data_path", type=str, required=True,
                        help="数据集路径，JSON格式")
    parser.add_argument("--cache_dir", type=str, default="cache",
                        help="缓存目录，用于存储情感标签等")
    
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录，用于保存模型和结果")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="权重衰减")
    parser.add_argument("--max_source_length", type=int, default=512,
                        help="源文本最大长度")
    parser.add_argument("--max_target_length", type=int, default=128,
                        help="目标摘要最大长度")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="数据加载器工作进程数")
    
    parser.add_argument("--use_sentiment", action="store_true",
                        help="是否使用情感一致性训练")
    parser.add_argument("--sentiment_loss_weight", type=float, default=1.0,
                        help="情感损失权重")
    
    parser.add_argument("--small_dataset_size", type=int, default=-1,
                        help="小数据集大小，用于快速测试，-1表示使用全部数据")
    
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--save_every_epoch", action="store_true",
                        help="是否每个epoch都保存检查点")
    
    args = parser.parse_args()
    main(args)
