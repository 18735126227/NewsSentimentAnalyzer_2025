"""
训练改进的BertSum模型用于抽取式摘要
使用小型数据集进行快速训练和测试
"""

import os
import sys
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, AdamW, get_linear_schedule_with_warmup

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.bertsum.modeling import BertSumExt
from models.bertsum.improved_extractor import ImprovedBertSumExtractor
from src.sentiment.analyzer import SentimentAnalyzer

class LCSTSExtractiveDataset(Dataset):
    """LCSTS数据集用于抽取式摘要"""
    
    def __init__(self, data_path, tokenizer, max_length=512):
        """
        初始化数据集
        
        Args:
            data_path: 数据集路径
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"从 {data_path} 加载数据...")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"加载了 {len(self.data)} 个样本")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """获取数据样本"""
        sample = self.data[idx]
        
        content = sample.get('content', '')
        title = sample.get('title', '')
        
        if not content:
            content = "空内容"
        
        sentences = []
        for s in content.replace('!', '。').replace('！', '。').replace('?', '。').replace('？', '。').split('。'):
            if s.strip():
                sentences.append(s.strip())
        
        if not sentences:
            sentences = ["空内容"]
        
        inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        labels = torch.zeros(len(sentences))
        
        title_words = set(title)
        for i, sentence in enumerate(sentences):
            score = 0.0
            
            position_score = 1.0 if i == 0 else 0.0
            
            match_score = 0.0
            for word in title_words:
                if word in sentence:
                    match_score += 0.1
            
            length = len(sentence)
            length_score = 1.0 if 10 <= length <= 50 else 0.5
            
            score = position_score * 0.3 + match_score * 0.5 + length_score * 0.2
            
            labels[i] = min(score, 1.0)  # 限制在0-1之间
        
        if torch.sum(labels) < 0.5:
            labels[0] = 1.0
        
        sentence_mask = torch.ones(len(sentences))
        
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'token_type_ids': inputs.get('token_type_ids', None),
            'sentence_mask': sentence_mask,
            'labels': labels,
            'content': content,
            'title': title,
            'sentences': sentences
        }

def collate_fn(batch):
    """数据批次整理函数"""
    max_sentences = max([item['input_ids'].size(0) for item in batch])
    
    batch_input_ids = []
    batch_attention_mask = []
    batch_token_type_ids = []
    batch_sentence_mask = []
    batch_labels = []
    batch_content = []
    batch_title = []
    batch_sentences = []
    
    for item in batch:
        num_sentences = item['input_ids'].size(0)
        
        padded_input_ids = torch.zeros((max_sentences, item['input_ids'].size(1)), dtype=torch.long)
        padded_attention_mask = torch.zeros((max_sentences, item['attention_mask'].size(1)), dtype=torch.long)
        padded_sentence_mask = torch.zeros(max_sentences, dtype=torch.float)
        padded_labels = torch.zeros(max_sentences, dtype=torch.float)
        
        padded_input_ids[:num_sentences] = item['input_ids']
        padded_attention_mask[:num_sentences] = item['attention_mask']
        padded_sentence_mask[:num_sentences] = item['sentence_mask']
        padded_labels[:num_sentences] = item['labels']
        
        batch_input_ids.append(padded_input_ids)
        batch_attention_mask.append(padded_attention_mask)
        batch_sentence_mask.append(padded_sentence_mask)
        batch_labels.append(padded_labels)
        batch_content.append(item['content'])
        batch_title.append(item['title'])
        batch_sentences.append(item['sentences'])
        
        if item['token_type_ids'] is not None:
            padded_token_type_ids = torch.zeros((max_sentences, item['token_type_ids'].size(1)), dtype=torch.long)
            padded_token_type_ids[:num_sentences] = item['token_type_ids']
            batch_token_type_ids.append(padded_token_type_ids)
    
    batch_data = {
        'input_ids': torch.stack(batch_input_ids),
        'attention_mask': torch.stack(batch_attention_mask),
        'sentence_mask': torch.stack(batch_sentence_mask),
        'labels': torch.stack(batch_labels),
        'content': batch_content,
        'title': batch_title,
        'sentences': batch_sentences
    }
    
    if batch_token_type_ids:
        batch_data['token_type_ids'] = torch.stack(batch_token_type_ids)
    
    return batch_data

def train(args):
    """训练模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"improved_bertsum_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    tokenizer = BertTokenizerFast.from_pretrained(args.bert_model_name)
    
    train_dataset = LCSTSExtractiveDataset(
        data_path=args.train_data_path,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )
    
    model = BertSumExt(
        bert_model_name=args.bert_model_name,
        dropout=args.dropout,
        use_sentiment=args.use_sentiment
    )
    model.to(device)
    
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    total_steps = len(train_dataloader) * args.num_epochs
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    
    sentiment_analyzer = None
    if args.use_sentiment:
        sentiment_analyzer = SentimentAnalyzer()
    
    print(f"开始训练，共 {args.num_epochs} 个 epoch...")
    
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentence_mask = batch['sentence_mask'].to(device)
            labels = batch['labels'].to(device)
            
            token_type_ids = None
            if 'token_type_ids' in batch:
                token_type_ids = batch['token_type_ids'].to(device)
            
            sentiment_labels = None
            if args.use_sentiment and sentiment_analyzer is not None:
                sentiment_labels = []
                for content in batch['content']:
                    sentiment = sentiment_analyzer.analyze(content)
                    if sentiment['label'] == 'positive':
                        sentiment_labels.append(0)
                    elif sentiment['label'] == 'neutral':
                        sentiment_labels.append(1)
                    else:  # negative
                        sentiment_labels.append(2)
                sentiment_labels = torch.tensor(sentiment_labels, dtype=torch.long).to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                sentence_mask=sentence_mask,
                labels=labels,
                sentiment_labels=sentiment_labels
            )
            
            loss = outputs["loss"]
            
            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            
            global_step += 1
            
            if global_step % args.save_steps == 0:
                checkpoint_path = os.path.join(save_dir, f"checkpoint-{global_step}")
                os.makedirs(checkpoint_path, exist_ok=True)
                
                torch.save(model.state_dict(), os.path.join(checkpoint_path, "pytorch_model.bin"))
                
                tokenizer.save_pretrained(checkpoint_path)
                
                print(f"保存检查点到 {checkpoint_path}")
        
        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{args.num_epochs} 平均损失: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(save_dir, "best_model")
            os.makedirs(best_model_path, exist_ok=True)
            
            torch.save(model.state_dict(), os.path.join(best_model_path, "pytorch_model.bin"))
            
            tokenizer.save_pretrained(best_model_path)
            
            print(f"保存最佳模型到 {best_model_path}")
    
    final_model_path = os.path.join(save_dir, "final_model")
    os.makedirs(final_model_path, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(final_model_path, "pytorch_model.bin"))
    
    tokenizer.save_pretrained(final_model_path)
    
    print(f"保存最终模型到 {final_model_path}")
    
    improved_extractor = ImprovedBertSumExtractor(
        bert_model_name=args.bert_model_name,
        dropout=args.dropout,
        use_sentiment=args.use_sentiment
    )
    improved_extractor.model.load_state_dict(model.state_dict())
    
    improved_model_path = os.path.join(save_dir, "improved_extractor")
    os.makedirs(improved_model_path, exist_ok=True)
    
    torch.save(improved_extractor.state_dict(), os.path.join(improved_model_path, "pytorch_model.bin"))
    
    tokenizer.save_pretrained(improved_model_path)
    
    print(f"保存ImprovedBertSumExtractor到 {improved_model_path}")
    
    return save_dir

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="训练改进的BertSum模型用于抽取式摘要")
    
    parser.add_argument("--train_data_path", type=str, default="data/lcsts/lcsts_small.json",
                        help="训练数据路径")
    parser.add_argument("--max_length", type=int, default=512,
                        help="最大序列长度")
    
    parser.add_argument("--bert_model_name", type=str, default="bert-base-chinese",
                        help="BERT模型名称")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout率")
    parser.add_argument("--use_sentiment", action="store_true",
                        help="是否使用情感一致性")
    
    parser.add_argument("--batch_size", type=int, default=4,
                        help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="权重衰减")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="最大梯度范数")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="数据加载器工作进程数")
    
    parser.add_argument("--save_dir", type=str, default="models/checkpoints",
                        help="保存目录")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="保存检查点的步数间隔")
    
    args = parser.parse_args()
    
    save_dir = train(args)
    
    print(f"训练完成，模型保存在 {save_dir}")

if __name__ == "__main__":
    main()
