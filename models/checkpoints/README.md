# 模型检查点

本目录包含训练好的模型检查点文件。

## BertSum抽取式摘要模型

### 训练参数

- 预训练模型: bert-base-chinese
- 数据集: LCSTS (中文短文本摘要数据集)
- 训练样本数: 900 (小型数据集用于快速验证)
- 验证样本数: 100
- 批次大小: 4
- 训练轮数: 2
- 学习率: 2e-5
- 情感一致性训练: 启用

### 模型特点

1. **多任务学习框架**
   - 主任务: 抽取式摘要生成
   - 辅助任务: 情感分类

2. **情感感知的句子选择**
   - 在选择句子时，不仅考虑内容重要性，还考虑句子的情感与文档整体情感的一致性

3. **基于bert-base-chinese的编码器**
   - 使用中文预训练模型，更好地适应中文语言特点

### 使用方法

```python
from models.bertsum.extractor import BertSumExtractor

# 加载模型
model = BertSumExtractor.from_pretrained("models/checkpoints/bertsum_model.pt")

# 生成摘要
summary = model.generate_summary(text, max_length=3)
```

### 评估指标

- ROUGE-1: 待评估
- ROUGE-2: 待评估
- ROUGE-L: 待评估
- 情感一致性: 待评估

## 模型文件说明

- `bertsum_model.pt`: BertSum抽取式摘要模型参数
- `config.json`: 模型配置文件
- `training_args.json`: 训练参数配置
- `training_log.txt`: 训练日志

## 评估方法

使用以下命令评估模型性能:

```bash
python scripts/evaluate_model.py \
  --data_path data/lcsts/lcsts_small.json \
  --model_path models/checkpoints/bertsum_model.pt \
  --output_dir results/evaluation \
  --num_samples 100
```
