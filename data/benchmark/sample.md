# SFT 微调技术详解

## 什么是 SFT？

SFT（Supervised Fine-Tuning，监督式微调）是指在预训练语言模型的基础上，使用高质量的标注数据对模型进行有监督训练的过程。

## SFT 的关键参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| learning_rate | 1e-5 ~ 2e-5 | 比预训练小 10 倍 |
| batch_size | 4 ~ 32 | 取决于显存 |
| epochs | 2 ~ 5 | 过拟合检测 |
| warmup_ratio | 0.03 ~ 0.1 | 学习率预热 |

## LoRA 简介

LoRA（Low-Rank Adaptation）是一种参数高效的微调方法，通过引入低秩矩阵来更新模型权重，大幅降低显存占用。

### LoRA 与 Full FT 对比

| 维度 | LoRA | Full Fine-tuning |
|------|------|------------------|
| 可训练参数量 | 0.1% ~ 1% | 100% |
| 显存占用 | ~12GB (7B模型) | ~60GB (7B模型) |
| 训练速度 | 快 3-5 倍 | 基准 |
| 效果 | 接近 Full FT | 最好 |

## 多轮对话数据格式

```json
{
  "conversations": [
    {"role": "system", "content": "你是一个AI助手"},
    {"role": "user", "content": "什么是SFT？"},
    {"role": "assistant", "content": "SFT是监督式微调..."}
  ]
}
```
