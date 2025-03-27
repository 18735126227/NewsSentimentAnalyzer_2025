import torch
import torch.nn as nn
from transformers import BertTokenizerFast, PreTrainedModel
from .modeling import BertSumExt

class BertSumExtractor(PreTrainedModel):
    """
    BertSum抽取器，用于抽取式摘要
    包装了BertSumExt模型，提供更方便的接口
    """
    
    def __init__(self, config=None, bert_model_name="bert-base-chinese", dropout=0.1, use_sentiment=False):
        if config is None:
            from transformers import BertConfig
            config = BertConfig.from_pretrained(bert_model_name)
        super().__init__(config)
        self.model = BertSumExt(bert_model_name=bert_model_name, dropout=dropout, use_sentiment=use_sentiment)
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)
    
    def forward(self, **kwargs):
        """
        前向传播
        """
        outputs = self.model(**kwargs)
        return outputs["sent_scores"].squeeze(-1)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        从预训练模型加载
        """
        model = cls(*model_args, **kwargs)
        
        try:
            state_dict = torch.load(pretrained_model_name_or_path, map_location="cpu")
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"加载预训练模型时出错: {e}")
            print("使用未初始化的模型")
        
        return model
    
    def save_pretrained(self, save_directory):
        """
        保存模型
        """
        torch.save(self.state_dict(), f"{save_directory}/pytorch_model.bin")
    
    def generate_summary(self, text, max_length=3):
        """
        生成摘要
        
        Args:
            text: 输入文本
            max_length: 最大摘要长度（句子数）
            
        Returns:
            摘要文本
        """
        sentences = [s.strip() for s in text.split('。') if s.strip()]
        
        if not sentences:
            return ""
        
        inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            scores = self(**inputs)
            
        max_length = min(max_length, len(sentences))
        top_indices = scores.argsort(descending=True)[:max_length].cpu().numpy()
        top_indices = sorted(top_indices)
        
        summary = '。'.join([sentences[i] for i in top_indices]) + '。'
        
        return summary
