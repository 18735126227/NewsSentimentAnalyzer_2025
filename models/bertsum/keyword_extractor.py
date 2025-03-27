"""
Keyword-based BertSumExtractor with improved sentence selection
"""

import os
import re
import torch
import numpy as np
import jieba
import torch.nn as nn
from transformers import BertTokenizerFast, PreTrainedModel
from .modeling import BertSumExt

class KeywordBertSumExtractor(PreTrainedModel):
    """
    Keyword-based BertSum extractor for extractive summarization
    Uses keyword matching and position-based scoring to select important sentences
    """
    
    def __init__(self, config=None, bert_model_name="bert-base-chinese", dropout=0.1, use_sentiment=False):
        if config is None:
            from transformers import BertConfig
            config = BertConfig.from_pretrained(bert_model_name)
        super().__init__(config)
        self.model = BertSumExt(bert_model_name=bert_model_name, dropout=dropout, use_sentiment=use_sentiment)
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)
        
        jieba.initialize()
    
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
    
    def extract_keywords(self, text, top_k=10):
        """
        Extract keywords from text
        
        Args:
            text: Input text
            top_k: Number of top keywords to extract
            
        Returns:
            List of keywords
        """
        is_chinese = '。' in text or '！' in text or '？' in text
        
        chinese_stopwords = {'的', '了', '和', '是', '在', '我', '有', '这', '个', '你', '们', '就', '也', '都', '要', '把', '与', '他', '她', '它', '这个', '那个', '一个', '不', '没有', '很', '这样', '那样', '如果', '因为', '所以', '但是', '可是', '不过', '而且', '并且', '或者', '然后', '接着', '其实', '其中', '一些', '一样', '一直', '只是', '只有', '这些', '那些', '自己', '什么', '怎么', '为什么', '哪里', '谁', '哪个', '哪些', '多少', '几个', '这里', '那里', '现在', '曾经', '已经', '正在', '将要', '应该', '可能', '必须', '需要', '能够', '可以', '不能', '不可以', '不要', '不必', '不用', '没有', '没什么', '没关系', '不管', '无论', '虽然', '尽管', '即使', '不论', '除了', '只要', '只有', '不仅', '而且', '不但', '并且', '然而', '但是', '可是', '不过', '因此', '所以', '因为', '由于', '如果', '假如', '假使', '假若', '若是', '如此', '这么', '那么', '如何', '怎样', '怎么', '何时', '何地', '何人', '何物', '何故', '何况', '如此', '这般', '这样', '那样', '一般', '一样', '一起', '一直', '一定', '一会儿', '一旦', '一来', '一切', '一下', '一点', '一些', '一种', '一番', '一阵', '一批', '一群', '一对', '一双', '一套', '一份', '一系列', '一家', '一部分', '一边', '一面', '一时', '一刻', '一天', '一年', '一生', '一世', '一辈子'}
        english_stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'may', 'might', 'must', 'can', 'could', 'of', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'would', 'should', 'could', 'ought', 'i\'m', 'you\'re', 'he\'s', 'she\'s', 'it\'s', 'we\'re', 'they\'re', 'i\'ve', 'you\'ve', 'we\'ve', 'they\'ve', 'i\'d', 'you\'d', 'he\'d', 'she\'d', 'we\'d', 'they\'d', 'i\'ll', 'you\'ll', 'he\'ll', 'she\'ll', 'we\'ll', 'they\'ll', 'isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t', 'hasn\'t', 'haven\'t', 'hadn\'t', 'doesn\'t', 'don\'t', 'didn\'t', 'won\'t', 'wouldn\'t', 'shan\'t', 'shouldn\'t', 'can\'t', 'cannot', 'couldn\'t', 'mustn\'t', 'let\'s', 'that\'s', 'who\'s', 'what\'s', 'here\'s', 'there\'s', 'when\'s', 'where\'s', 'why\'s', 'how\'s'}
        
        if is_chinese:
            words = jieba.lcut(text)
            stopwords = chinese_stopwords
        else:
            words = re.findall(r'\b\w+\b', text.lower())
            stopwords = english_stopwords
        
        word_freq = {}
        for word in words:
            if word not in stopwords and len(word) > 1:
                if word in word_freq:
                    word_freq[word] += 1
                else:
                    word_freq[word] = 1
        
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, _ in sorted_words[:top_k]]
    
    def score_sentences(self, sentences, keywords):
        """
        Score sentences based on keyword matching and position
        
        Args:
            sentences: List of sentences
            keywords: List of keywords
            
        Returns:
            List of sentence scores
        """
        scores = []
        
        for i, sentence in enumerate(sentences):
            score = 0.0
            
            position_score = 0.0
            if i == 0:
                position_score = 1.0
            elif i < len(sentences) // 3:
                position_score = 0.8
            elif i < 2 * len(sentences) // 3:
                position_score = 0.5
            else:
                position_score = 0.3
            
            keyword_score = 0.0
            for keyword in keywords:
                if keyword in sentence:
                    keyword_score += 1.0
            
            if keywords:
                keyword_score = keyword_score / len(keywords) * 5.0  # Scale up for importance
            
            length = len(sentence)
            length_score = 0.0
            if 10 <= length <= 50:
                length_score = 1.0
            elif length < 10:
                length_score = length / 10.0
            else:
                length_score = 50.0 / length
            
            number_score = 0.0
            if any(c.isdigit() for c in sentence):
                number_score = 0.5
            
            score = (position_score * 0.3) + (keyword_score * 0.5) + (length_score * 0.1) + (number_score * 0.1)
            scores.append(score)
        
        return scores
    
    def generate_summary(self, text, max_length=3):
        """
        Generate summary from text using keyword extraction and sentence scoring
        
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
        
        keywords = self.extract_keywords(text, top_k=10)
        
        scores = self.score_sentences(sentences, keywords)
        
        if max_length == 1:
            selected_indices = [np.argmax(scores)]
        else:
            selected_indices = np.argsort(scores)[::-1][:max_length].tolist()
            
            if 0 not in selected_indices and len(selected_indices) >= max_length:
                min_score_idx = selected_indices[np.argmin([scores[i] for i in selected_indices])]
                replace_idx = selected_indices.index(min_score_idx)
                selected_indices[replace_idx] = 0
        
        selected_indices = sorted(selected_indices)
        
        selected_sentences = [sentences[i] for i in selected_indices if i < len(sentences)]
        
        if not selected_sentences:
            return sentences[0] + end_mark
        
        if end_mark == '.':
            summary = '. '.join(selected_sentences) + '.'
        else:
            summary = end_mark.join(selected_sentences) + end_mark
        
        return summary
