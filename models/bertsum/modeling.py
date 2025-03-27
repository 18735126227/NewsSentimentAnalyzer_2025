import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig

class BertSumExt(nn.Module):
    """
    改进的BertSum模型用于抽取式摘要，基于bert-base-chinese预训练模型
    添加了更复杂的句子评分机制，以提高摘要质量
    """
    
    def __init__(self, bert_model_name="bert-base-chinese", dropout=0.1, use_sentiment=False):
        super(BertSumExt, self).__init__()
        
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        
        hidden_size = self.bert.config.hidden_size
        self.sent_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=2,
            bidirectional=True,
            dropout=dropout,
            batch_first=True
        )
        
        self.sent_layer_norm = nn.LayerNorm(hidden_size)
        self.sent_fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.sent_fc2 = nn.Linear(hidden_size // 2, 1)
        
        self.use_sentiment = use_sentiment
        if use_sentiment:
            self.sentiment_classifier = nn.Linear(hidden_size, 3)  # 3分类：积极、中性、消极
            self.sentiment_weight = 0.2  # 降低情感任务的权重
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        sentence_mask=None,
        labels=None,
        sentiment_labels=None,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
        
        cls_indices = torch.nonzero(input_ids == 101, as_tuple=False)  # 101 is the CLS token ID for BERT
        batch_indices = cls_indices[:, 0]
        seq_indices = cls_indices[:, 1]
        
        cls_output = sequence_output[batch_indices, seq_indices]  # [num_cls, hidden_size]
        
        batch_size = input_ids.size(0) if input_ids is not None else 1
        num_sentences_per_doc = cls_indices.size(0) // batch_size
        cls_output = cls_output.view(batch_size, num_sentences_per_doc, -1)
        
        lstm_output, _ = self.sent_lstm(cls_output)
        
        norm_output = self.sent_layer_norm(lstm_output)
        
        fc1_output = F.relu(self.sent_fc1(self.dropout(norm_output)))
        
        raw_scores = self.sent_fc2(self.dropout(fc1_output))  # [batch_size, num_sentences, 1]
        
        batch_size, num_sentences, _ = raw_scores.size()
        position_bias = torch.zeros_like(raw_scores)
        
        for i in range(num_sentences):
            if i == 0:
                position_bias[:, i, 0] = -1.0
            else:
                normalized_pos = ((i-1) / max(1, num_sentences-2)) * 2 - 1  # -1 to 1
                position_bias[:, i, 0] = -0.3 * (normalized_pos ** 2) + 0.5  # parabola, max at middle
        
        sent_scores = torch.sigmoid(raw_scores + position_bias)  # [batch_size, num_sentences, 1]
        
        sent_scores = sent_scores.squeeze(-1)  # [batch_size, num_sentences]
        
        loss = None
        if labels is not None:
            loss_fct = nn.MarginRankingLoss(margin=0.5)
            
            total_loss = 0.0
            num_pairs = 0
            
            for i in range(batch_size):
                doc_scores = sent_scores[i]  # [num_sentences]
                doc_labels = labels[i]  # [num_sentences]
                
                for j in range(num_sentences_per_doc):
                    for k in range(j+1, num_sentences_per_doc):
                        if doc_labels[j] != doc_labels[k]:  # 只有当标签不同时才计算损失
                            target = torch.sign(doc_labels[j] - doc_labels[k])
                            pair_loss = loss_fct(doc_scores[j].unsqueeze(0), 
                                               doc_scores[k].unsqueeze(0), 
                                               target.unsqueeze(0))
                            total_loss += pair_loss
                            num_pairs += 1
            
            if num_pairs > 0:
                loss = total_loss / num_pairs
            else:
                loss = F.mse_loss(sent_scores, labels.float())
        
        sentiment_logits = None
        if self.use_sentiment and pooled_output is not None:
            sentiment_logits = self.sentiment_classifier(self.dropout(pooled_output))
            
            if sentiment_labels is not None and loss is not None:
                sentiment_loss_fct = nn.CrossEntropyLoss()
                sentiment_loss = sentiment_loss_fct(sentiment_logits, sentiment_labels)
                
                if loss is not None:
                    loss = loss + self.sentiment_weight * sentiment_loss
                else:
                    loss = sentiment_loss
        
        return {
            "loss": loss,
            "sent_scores": sent_scores,
            "sentiment_logits": sentiment_logits,
        }
