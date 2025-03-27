"""
Headline Generator for Chinese news summarization
Specifically designed to generate title-like summaries for LCSTS dataset
"""

import os
import re
import torch
import numpy as np
import jieba
import jieba.analyse
from transformers import BertTokenizerFast, PreTrainedModel
from .modeling import BertSumExt

class HeadlineGenerator(PreTrainedModel):
    """
    Headline Generator for Chinese news summarization
    Specifically designed to generate title-like summaries for LCSTS dataset
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
    
    def generate_headline(self, text):
        """
        Generate headline from text
        
        Args:
            text: Input text
            
        Returns:
            Headline text
        """
        keywords = self.extract_keywords(text)
        entities = self.extract_entities(text)
        
        sentences = []
        for s in text.replace('!', '。').replace('！', '。').replace('?', '。').replace('？', '。').split('。'):
            if s.strip():
                sentences.append(s.strip())
        
        if not sentences:
            return ""
        
        if len(sentences) == 1:
            return self._compress_sentence(sentences[0], keywords, entities)
        
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            score = 0.0
            
            position_score = 1.0 if i == 0 else 0.0
            
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
                keyword_score * 0.4 +
                entity_score * 0.2 +
                title_score * 0.2
            )
            
            sentence_scores.append((i, score, sentence))
        
        sorted_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
        
        best_sentence = sorted_sentences[0][2]
        
        headline = self._compress_sentence(best_sentence, keywords, entities)
        
        return headline
    
    def _compress_sentence(self, sentence, keywords, entities):
        """
        Compress sentence to headline
        
        Args:
            sentence: Input sentence
            keywords: List of keywords
            entities: List of entities
            
        Returns:
            Compressed headline
        """
        if len(sentence) <= 15:
            return sentence
        
        segments = []
        for segment in re.split(r'[,，;；]', sentence):
            if segment.strip():
                segments.append(segment.strip())
        
        if not segments:
            return sentence
        
        if len(segments) == 1:
            words = list(jieba.cut(sentence))
            
            keyword_positions = []
            for i, word in enumerate(words):
                if word in keywords or word in entities:
                    keyword_positions.append(i)
            
            if not keyword_positions:
                return sentence[:15]
            
            if len(keyword_positions) == 1:
                pos = keyword_positions[0]
                start = max(0, pos - 3)
                end = min(len(words), pos + 4)
                return ''.join(words[start:end])
            else:
                clusters = []
                current_cluster = [keyword_positions[0]]
                
                for i in range(1, len(keyword_positions)):
                    if keyword_positions[i] - keyword_positions[i-1] <= 5:
                        current_cluster.append(keyword_positions[i])
                    else:
                        clusters.append(current_cluster)
                        current_cluster = [keyword_positions[i]]
                
                if current_cluster:
                    clusters.append(current_cluster)
                
                largest_cluster = max(clusters, key=len)
                
                start = max(0, largest_cluster[0] - 2)
                end = min(len(words), largest_cluster[-1] + 3)
                
                return ''.join(words[start:end])
        else:
            segment_scores = []
            for segment in segments:
                score = 0
                for keyword in keywords:
                    if keyword in segment:
                        score += 1
                for entity in entities:
                    if entity in segment:
                        score += 1
                segment_scores.append((segment, score))
            
            sorted_segments = sorted(segment_scores, key=lambda x: x[1], reverse=True)
            
            return sorted_segments[0][0]
    
    def generate_summary(self, text, max_length=1):
        """
        Generate summary from text
        
        Args:
            text: Input text
            max_length: Maximum number of sentences in summary (ignored for headline generation)
            
        Returns:
            Summary text
        """
        return self.generate_headline(text)
