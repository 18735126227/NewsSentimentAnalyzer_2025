"""
Improved BertSumExtractor with forced diversity in sentence selection
"""

import os
import torch
import numpy as np
import torch.nn as nn
from transformers import BertTokenizerFast, PreTrainedModel
from .modeling import BertSumExt

class ImprovedBertSumExtractor(PreTrainedModel):
    """
    Improved BertSum extractor for extractive summarization
    Forces diversity in sentence selection
    """
    
    def __init__(self, config=None, bert_model_name="bert-base-chinese", dropout=0.1, use_sentiment=False):
        if config is None:
            from transformers import BertConfig
            config = BertConfig.from_pretrained(bert_model_name)
        super().__init__(config)
        self.model = BertSumExt(bert_model_name=bert_model_name, dropout=dropout, use_sentiment=use_sentiment)
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)
    
    def forward(self, **kwargs):
        """Forward pass"""
        outputs = self.model(**kwargs)
        return outputs["sent_scores"].squeeze(-1)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load from pretrained"""
        model = cls(*model_args, **kwargs)
        
        try:
            if os.path.isdir(pretrained_model_name_or_path):
                model_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
            else:
                model_path = pretrained_model_name_or_path
                
            state_dict = torch.load(model_path, map_location="cpu")
            
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("bert."):
                    new_key = "model." + key
                    new_state_dict[new_key] = value
                elif key.startswith("ext_layer."):
                    new_key = "model." + key
                    new_state_dict[new_key] = value
                elif key.startswith("sentiment_classifier."):
                    new_key = "model." + key
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
            
            if len(missing_keys) > 0:
                print(f"Missing keys: {missing_keys}")
            if len(unexpected_keys) > 0:
                print(f"Unexpected keys: {unexpected_keys}")
                
        except Exception as e:
            print(f"Error loading pretrained model: {e}")
            print("Using uninitialized model")
        
        return model
    
    def save_pretrained(self, save_directory):
        """Save model"""
        torch.save(self.state_dict(), f"{save_directory}/pytorch_model.bin")
    
    def generate_summary(self, text, max_length=3):
        """
        Generate summary from text with forced diversity
        
        Args:
            text: Input text
            max_length: Maximum number of sentences in summary
            
        Returns:
            Summary text
        """
        is_chinese = '。' in text or '！' in text or '？' in text
        
        sentences = []
        if is_chinese:
            for s in text.replace('!', '。').replace('！', '。').replace('?', '。').replace('？', '。').split('。'):
                if s.strip():
                    sentences.append(s.strip())
            end_mark = '。'
        else:
            for s in text.replace('!', '.').replace('?', '.').split('.'):
                if s.strip():
                    sentences.append(s.strip())
            end_mark = '.'
        
        if not sentences:
            return ""
        
        if len(sentences) <= 1:
            return sentences[0] + end_mark
            
        if len(sentences) <= max_length:
            return end_mark.join(sentences) + end_mark
        
        inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        sentence_mask = torch.ones(1, len(sentences))
        
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        sentence_mask = sentence_mask.to(device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                token_type_ids=inputs.get('token_type_ids', None),
                sentence_mask=sentence_mask
            )
            
            sent_scores = outputs["sent_scores"]
            
            if len(sent_scores.shape) == 3:
                sent_scores = sent_scores.squeeze(-1)
            
            selected_indices = []
            
            raw_scores = sent_scores[0].cpu().numpy()
            
            adjusted_scores = np.zeros_like(raw_scores)
            for i in range(len(raw_scores)):
                if i == 0:
                    adjusted_scores[i] = raw_scores[i] * 1.2
                elif i < len(raw_scores) // 3:
                    adjusted_scores[i] = raw_scores[i] * 1.1
                elif i < 2 * len(raw_scores) // 3:
                    adjusted_scores[i] = raw_scores[i]
                else:
                    adjusted_scores[i] = raw_scores[i] * 0.9
            
            keywords = ['重要', '关键', '突破', '宣布', '发现', '决定', '最新', '首次', 
                       'important', 'key', 'breakthrough', 'announced', 'discovered', 'decided']
            
            for i, sentence in enumerate(sentences):
                for keyword in keywords:
                    if keyword in sentence:
                        adjusted_scores[i] *= 1.2
                        break
                
                if any(c.isdigit() for c in sentence):
                    adjusted_scores[i] *= 1.1
            
            if max_length == 1:
                selected_indices = [np.argmax(adjusted_scores)]
            else:
                top_indices = np.argsort(adjusted_scores)[::-1][:max_length]
                selected_indices = top_indices.tolist()
                
                if 0 not in selected_indices and len(selected_indices) >= max_length:
                    min_score_idx = np.argmin([adjusted_scores[i] for i in selected_indices])
                    selected_indices[min_score_idx] = 0
            
            selected_indices = sorted(list(set(selected_indices)))
        
        selected_sentences = [sentences[i] for i in selected_indices if i < len(sentences)]
        
        if not selected_sentences:
            return sentences[0] + end_mark
        
        if end_mark == '.':
            summary = '. '.join(selected_sentences) + '.'
        else:
            summary = end_mark.join(selected_sentences) + end_mark
        
        return summary
