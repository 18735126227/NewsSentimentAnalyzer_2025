import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class BertSumExt(nn.Module):
    """
    BertSum模型用于抽取式摘要，基于bert-base-chinese预训练模型
    """
    
    def __init__(self, bert_model_name="bert-base-chinese", dropout=0.1, use_sentiment=False):
        super(BertSumExt, self).__init__()
        
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        self.ext_layer = nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
        self.use_sentiment = use_sentiment
        if use_sentiment:
            self.sentiment_classifier = nn.Linear(self.bert.config.hidden_size, 3)  # 3分类：积极、中性、消极
    
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
        
        batch_size = input_ids.size(0) if input_ids is not None else 0
        hidden_size = sequence_output.size(2) if sequence_output is not None else 0
        max_sentences = sentence_mask.size(1) if sentence_mask is not None else 0
        
        expanded_cls = torch.zeros(
            batch_size, max_sentences, hidden_size, 
            dtype=sequence_output.dtype, 
            device=sequence_output.device
        )
        
        device = input_ids.device if input_ids is not None else sequence_output.device
        sentence_indices = torch.arange(max_sentences, device=device).unsqueeze(0).expand(batch_size, -1)
        valid_indices = sentence_mask.bool() if sentence_mask is not None else torch.zeros(batch_size, max_sentences, dtype=torch.bool, device=device)
        
        sent_scores = self.sigmoid(self.ext_layer(self.dropout(cls_output)))  # [num_cls, 1]
        
        expanded_scores = torch.zeros(
            batch_size, max_sentences, 
            dtype=sent_scores.dtype, 
            device=sent_scores.device
        )
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCELoss(reduction='none')
            sent_scores_flat = sent_scores.view(-1)
            labels_flat = labels.float().view(-1)
            
            min_len = min(sent_scores_flat.size(0), labels_flat.size(0))
            per_sentence_loss = loss_fct(sent_scores_flat[:min_len], labels_flat[:min_len])
            
            if sentence_mask is not None:
                mask_flat = sentence_mask.view(-1)
                if mask_flat.size(0) != min_len:
                    mask_flat = mask_flat[:min_len] if mask_flat.size(0) > min_len else torch.cat([mask_flat, torch.zeros(min_len - mask_flat.size(0), device=mask_flat.device)])
                
                masked_loss = per_sentence_loss * mask_flat
                loss = masked_loss.sum() / (mask_flat.sum() + 1e-10)
            else:
                loss = per_sentence_loss.mean()
        
        sentiment_logits = None
        if self.use_sentiment and pooled_output is not None:
            sentiment_logits = self.sentiment_classifier(self.dropout(pooled_output))
            
            if sentiment_labels is not None and loss is not None:
                sentiment_loss_fct = nn.CrossEntropyLoss()
                sentiment_loss = sentiment_loss_fct(sentiment_logits, sentiment_labels)
                
                loss = loss + sentiment_loss
        
        return {
            "loss": loss,
            "sent_scores": sent_scores,
            "sentiment_logits": sentiment_logits,
        }
