# BertSum抽取式摘要模型

## 模型描述

这是一个基于BERT的中文抽取式摘要模型，使用多任务学习框架同时优化摘要生成和情感分类任务，以保证生成的摘要与原文具有一致的情感倾向。

## 训练参数

- 预训练模型: bert-base-chinese
- 数据集: LCSTS (中文短文本摘要数据集)
- 训练样本数: 900
- 验证样本数: 100
- 批次大小: 4
- 学习率: 2e-5
- 训练轮数: 2
- 情感一致性训练: 启用
- 情感损失权重: 1.0

## 性能指标

### 训练集 (Epoch 2)

- 损失: 0.5768
- 情感准确率: 0.7978
- 情感F1: 0.7709

### 验证集 (Epoch 2)

- 损失: 0.7649
- 情感准确率: 0.7100
- 情感F1: 0.6668

## 模型架构

BertSum模型基于bert-base-chinese预训练模型，通过多任务学习框架同时优化摘要生成和情感分类任务。模型包含以下主要组件：

1. **BERT编码器**：使用bert-base-chinese作为基础编码器，提取文本的语义表示。

2. **句子表示提取**：从BERT编码器的输出中提取每个句子的表示（使用[CLS]标记的输出）。

3. **句子重要性评分**：使用线性层和sigmoid激活函数计算每个句子的重要性得分。

4. **文档情感分类**：使用线性层和softmax激活函数对文档进行情感分类。

5. **多任务联合损失**：结合摘要损失和情感分类损失进行联合优化。

6. **情感感知的句子选择**：在选择句子组成摘要时，不仅考虑句子的重要性得分，还考虑句子的情感与文档整体情感的一致性。

## 使用方法

```python
from models.bertsum.extractor import BertSumExtractor

# 加载模型
model = BertSumExtractor.from_pretrained("models/checkpoints/bertsum_sentiment_20250327_153529/best_sentiment_checkpoint_epoch_2")

# 生成摘要
text = "这是一篇关于人工智能的新闻文章。人工智能技术正在迅速发展，改变着我们的生活方式。许多公司正在投资人工智能研究，希望在这一领域取得突破。专家认为，人工智能将在未来十年内彻底改变多个行业。"
summary = model.generate_summary(text, max_length=3)
print(summary)
```

## 训练日期

2025-03-27

## 模型文件

由于GitHub文件大小限制，模型二进制文件未包含在仓库中。完整模型文件包括：

- `pytorch_model.bin`: 模型权重文件
- `extractor.bin`: 抽取器模型文件
- `special_tokens_map.json`: 特殊标记映射
- `tokenizer.json`: 分词器配置
- `tokenizer_config.json`: 分词器配置
- `vocab.txt`: 词汇表
- `training_config.json`: 训练配置
- `metrics_epoch_1.json`: 第一轮训练指标
- `metrics_epoch_2.json`: 第二轮训练指标

## 注意事项

1. 模型在小型数据集（1000个样本）上训练，用于验证训练流程和模型架构的有效性。
2. 在实际应用中，建议使用更大的数据集进行训练，以获得更好的性能。
3. 模型的情感一致性特性使其生成的摘要能够保持与原文相同的情感倾向，适用于需要保持情感一致性的应用场景。
