import torch
import torch.nn as nn
from transformers import PegasusForConditionalGeneration

class SentimentAwarePegasus(nn.Module):
    """
    情感感知Pegasus模型，双通道联合训练
    结合文本内容和情感特征进行摘要生成
    """
    
    def __init__(self, model_name: str = "IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese-V1"):
        super().__init__()
        self.pegasus = PegasusForConditionalGeneration.from_pretrained(model_name)
        
        self.sentiment_classifier = nn.Linear(
            self.pegasus.config.d_model, 
            3  # 3分类：积极、中性、消极
        )
        
        self.sentiment_attention = nn.Linear(self.pegasus.config.d_model, 1)
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
        sentiment_labels=None,
        return_dict=None,
    ):
        outputs = self.pegasus(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            return_dict=True,
        )
        
        encoder_hidden_states = outputs.encoder_last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        attention_weights = torch.softmax(
            self.sentiment_attention(encoder_hidden_states), 
            dim=1
        )
        sentence_repr = torch.sum(
            attention_weights * encoder_hidden_states, 
            dim=1
        )  # [batch_size, hidden_size]
        
        sentiment_logits = self.sentiment_classifier(sentence_repr)  # [batch_size, 3]
        
        loss = outputs.loss  # 基础摘要损失
        
        if sentiment_labels is not None:
            sentiment_loss_fct = nn.CrossEntropyLoss()
            sentiment_loss = sentiment_loss_fct(sentiment_logits, sentiment_labels)
            
            total_loss = loss + sentiment_loss
        else:
            total_loss = loss
        
        return {
            "loss": total_loss,
            "logits": outputs.logits,
            "sentiment_logits": sentiment_logits,
        }
    
    def generate(self, **kwargs):
        """封装原始的generate方法"""
        return self.pegasus.generate(**kwargs)
