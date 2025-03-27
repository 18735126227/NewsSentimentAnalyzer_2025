"""
Title-focused extractor for Chinese news summarization
Specifically designed to generate summaries that better match reference titles in LCSTS dataset
"""

import os
import re
import torch
import numpy as np
import jieba
import jieba.analyse
from transformers import BertTokenizerFast, PreTrainedModel
from .modeling import BertSumExt

class TitleFocusedExtractor(PreTrainedModel):
    """
    Title-focused extractor for Chinese news summarization
    Specifically designed to generate summaries that better match reference titles in LCSTS dataset
    """
    
    def __init__(self, config=None, bert_model_name="bert-base-chinese", dropout=0.1, use_sentiment=False):
        if config is None:
            from transformers import BertConfig
            config = BertConfig.from_pretrained(bert_model_name)
        super().__init__(config)
        self.model = BertSumExt(bert_model_name=bert_model_name, dropout=dropout, use_sentiment=use_sentiment)
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)
        
        self.stopwords = set()
        try:
            stopwords_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/stopwords/chinese_stopwords.txt")
            if os.path.exists(stopwords_path):
                with open(stopwords_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        self.stopwords.add(line.strip())
        except Exception as e:
            print(f"Warning: Could not load stopwords: {e}")
    
    def forward(self, **kwargs):
        """Forward pass"""
        outputs = self.model(**kwargs)
        return outputs["sent_scores"].squeeze(-1)
    
    def extract_keywords(self, text, top_k=10, allow_pos=('n', 'nr', 'ns', 'nt', 'nz', 'vn')):
        """
        Extract keywords from text using TF-IDF and TextRank
        
        Args:
            text: Input text
            top_k: Number of keywords to extract
            allow_pos: Allowed part-of-speech tags
            
        Returns:
            List of keywords
        """
        tfidf_keywords = jieba.analyse.extract_tags(text, topK=top_k*2, allowPOS=allow_pos)
        
        textrank_keywords = jieba.analyse.textrank(text, topK=top_k*2, allowPOS=allow_pos)
        
        combined_keywords = []
        for keyword in tfidf_keywords + textrank_keywords:
            if keyword not in self.stopwords and len(keyword) > 1:
                combined_keywords.append(keyword)
        
        unique_keywords = list(dict.fromkeys(combined_keywords))[:top_k]
        
        return unique_keywords
    
    def extract_entities(self, text):
        """
        Extract named entities from text
        
        Args:
            text: Input text
            
        Returns:
            List of entities
        """
        entities = []
        
        date_patterns = [
            r'\d{4}年\d{1,2}月\d{1,2}日',
            r'\d{4}年\d{1,2}月',
            r'\d{1,2}月\d{1,2}日',
            r'昨天',
            r'今天',
            r'明天',
            r'前天',
            r'后天'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            entities.extend(matches)
        
        number_patterns = [
            r'\d+\.?\d*%',
            r'\d+\.?\d*亿',
            r'\d+\.?\d*万',
            r'\d+\.?\d*元',
            r'\d+\.?\d*美元',
            r'\d+\.?\d*人'
        ]
        
        for pattern in number_patterns:
            matches = re.findall(pattern, text)
            entities.extend(matches)
        
        words = jieba.posseg.cut(text)
        for word, flag in words:
            if flag in ('ns', 'nt', 'nr') and len(word) > 1 and word not in self.stopwords:
                entities.append(word)
        
        return list(set(entities))
    
    def score_sentences(self, sentences, keywords, entities):
        """
        Score sentences based on multiple factors
        
        Args:
            sentences: List of sentences
            keywords: List of keywords
            entities: List of entities
            
        Returns:
            List of scores
        """
        scores = []
        
        sentence_lengths = [len(s) for s in sentences]
        avg_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
        
        for i, sentence in enumerate(sentences):
            score = 0.0
            
            position_score = 1.0 if i == 0 else 0.0
            
            length_ratio = min(sentence_lengths[i] / avg_length, 2.0) if avg_length > 0 else 1.0
            length_score = 1.0 if 0.5 <= length_ratio <= 1.5 else 0.5
            
            keyword_count = 0
            for keyword in keywords:
                if keyword in sentence:
                    keyword_count += 1
            keyword_score = min(keyword_count / 3.0, 1.0) if keywords else 0.0
            
            entity_count = 0
            for entity in entities:
                if entity in sentence:
                    entity_count += 1
            entity_score = min(entity_count / 2.0, 1.0) if entities else 0.0
            
            title_features = 0
            
            if '？' in sentence or '?' in sentence:
                title_features += 0.5
            
            announcement_patterns = ['宣布', '发布', '推出', '公布', '表示', '称']
            for pattern in announcement_patterns:
                if pattern in sentence:
                    title_features += 0.3
                    break
            
            news_patterns = ['首次', '突破', '创新', '最新', '重要', '关键', '最大', '最高']
            for pattern in news_patterns:
                if pattern in sentence:
                    title_features += 0.3
                    break
            
            title_score = min(title_features, 1.0)
            
            score = (
                position_score * 0.2 +
                length_score * 0.1 +
                keyword_score * 0.3 +
                entity_score * 0.2 +
                title_score * 0.2
            )
            
            scores.append(score)
        
        return scores
    
    def generate_summary(self, text, max_length=3):
        """
        Generate summary from text
        
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
        
        keywords = self.extract_keywords(text)
        entities = self.extract_entities(text)
        
        scores = self.score_sentences(sentences, keywords, entities)
        
        if max_length == 1:
            adjusted_scores = []
            for i, score in enumerate(scores):
                length_penalty = min(1.0, 30 / max(10, len(sentences[i])))
                adjusted_scores.append(score * length_penalty)
            
            selected_indices = [np.argmax(adjusted_scores)]
        else:
            selected_indices = []
            
            first_sentence_score = scores[0]
            
            sorted_indices = np.argsort(scores)[::-1]
            
            selected_keywords = set()
            
            for idx in sorted_indices:
                if len(selected_indices) >= max_length:
                    break
                
                if idx == 0 and first_sentence_score > 0.3:
                    selected_indices.append(idx)
                    for keyword in keywords:
                        if keyword in sentences[idx]:
                            selected_keywords.add(keyword)
                    continue
                
                sentence_keywords = [kw for kw in keywords if kw in sentences[idx]]
                new_keywords = [kw for kw in sentence_keywords if kw not in selected_keywords]
                
                if new_keywords or scores[idx] > 0.7:
                    selected_indices.append(idx)
                    for keyword in sentence_keywords:
                        selected_keywords.add(keyword)
            
            if len(selected_indices) < max_length:
                for idx in sorted_indices:
                    if idx not in selected_indices:
                        selected_indices.append(idx)
                        if len(selected_indices) >= max_length:
                            break
            
            selected_indices = sorted(selected_indices)
        
        selected_sentences = [sentences[i] for i in selected_indices if i < len(sentences)]
        
        if not selected_sentences:
            return sentences[0] + end_mark
        
        if max_length == 1 and len(selected_sentences[0]) > 30:
            parts = selected_sentences[0].split('，')
            if len(parts) > 1:
                best_part = parts[0]
                best_score = 0
                
                for part in parts:
                    part_score = sum(1 for kw in keywords if kw in part)
                    if part_score > best_score or (part_score == best_score and len(part) < len(best_part)):
                        best_score = part_score
                        best_part = part
                
                if best_score > 0:
                    selected_sentences[0] = best_part
        
        if end_mark == '.':
            summary = '. '.join(selected_sentences) + '.'
        else:
            summary = end_mark.join(selected_sentences) + end_mark
        
        return summary
