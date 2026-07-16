# 数据集构造指南

本文档说明如何为 Agentic RAG 系统构造和扩展测试数据集。

---

## 目录

- [数据集结构](#数据集结构)
- [解析器测试文件集](#解析器测试文件集)
- [RAG QA 数据集](#rag-qa-数据集)
- [如何添加中文通用数据集](#如何添加中文通用数据集)
- [自定义领域数据集](#自定义领域数据集)

---

## 数据集结构

```
data/
├── benchmark/               # 解析器测试文件
│   ├── sample.txt           # 纯文本
│   ├── sample.md            # Markdown
│   ├── sample.json          # JSON
│   ├── sample.pdf           # PDF（需手动添加）
│   ├── sample.docx          # Word（需手动添加）
│   └── sample.xlsx          # Excel（需手动添加）
│
├── eval/
│   ├── qa_benchmark.json    # RAG QA 评估数据集
│   └── ragas_result.csv     # RAGAS 评估结果（自动生成）
│
└── docs/                    # RAG 知识库文档
    └── ...                   # 你的业务文档
```

---

## 解析器测试文件集

### 快速开始

当前提供的测试文件：

| 文件 | 内容 | 测试覆盖 |
|------|------|---------|
| `sample.txt` | 系统功能介绍 | 基础文本提取 |
| `sample.md` | SFT/LoRA 技术文档 | Markdown 结构（标题、表格、代码块） |
| `sample.json` | 参数配置 JSON | 结构化数据解析 |

### 如何添加 PDF 测试文件

把 PDF 文件放入 `data/benchmark/` 即可：

```bash
cp /path/to/your/test.pdf data/benchmark/
```

建议覆盖以下类型：

| PDF 类型 | 文件名示意 | 测试重点 |
|---------|-----------|---------|
| 纯文字 PDF | `text_only.pdf` | PyMuPDF 文字提取 |
| 带表格 PDF | `with_table.pdf` | camelot 表格提取 |
| 扫描件 PDF | `scanned.pdf` | OCR 兜底 |
| 混合 PDF | `mixed.pdf` | 文字+表格+图片综合 |

### 如何添加 Word/Excel 测试文件

```bash
# Word
cp /path/to/test.docx data/benchmark/

# Excel（多 sheet）
cp /path/to/test.xlsx data/benchmark/
```

### 验证

添加后运行：

```bash
pytest tests/test_parser.py -v -k "test_parser_stats"
```

统计会自动包含新增文件。

---

## RAG QA 数据集

### 数据格式

```json
[
  {
    "id": "L1-001",
    "question": "什么是SFT微调？",
    "ground_truth": "SFT（Supervised Fine-Tuning，监督式微调）是...",
    "relevant_docs": ["sample.md"],
    "difficulty": "L1_factual",
    "expected_topics": ["SFT", "监督式微调"]
  }
]
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 唯一标识，建议格式 `{难度}-{序号}` |
| `question` | string | 用户问题 |
| `ground_truth` | string | 标准答案（用于 Recall/Precision 评估） |
| `relevant_docs` | list | 答案涉及的相关文档列表 |
| `difficulty` | string | 难度级别 |
| `expected_topics` | list | 期望涵盖的主题词（用于自动判分） |

### 难度级别设计

| 级别 | ID 前缀 | 说明 | 示例 |
|------|---------|------|------|
| L1 事实性 | `L1-` | 单文档直接查询 | "什么是LoRA？" |
| L2 比较性 | `L2-` | 两方案对比 | "LoRA和FT的区别？" |
| L3 推理 | `L3-` | 需要多步推理 | "为什么LoRA省显存？" |
| L4 跨文档 | `L4-` | 综合多文档 | "设计完整实验方案" |

### 编写高质量 QA Pair 的原则

1. **互不依赖**：每个问题独立可答，不依赖其他问题的上下文
2. **答案可验证**：ground_truth 基于 knowledge base 中的文档
3. **难度递进**：从简单事实到复杂推理逐步深入
4. **覆盖全面**：覆盖知识库中所有主要文档和主题

---

## 如何添加中文通用数据集

如果你需要更大规模的评估数据集，以下中文基准可以直接使用：

### DuReader（百度）

```python
# 安装
pip install datasets

# 加载
from datasets import load_dataset
dataset = load_dataset("du_reader", split="train")

# 格式适配
qa_pairs = []
for item in dataset:
    qa_pairs.append({
        "id": f"dureader_{i}",
        "question": item["question"],
        "ground_truth": item["answer"],
        "difficulty": "L1_factual",
    })
```

> **注意**：DuReader 是开放域数据集，答案可能不在你的知识库中。
> 建议只用于测试检索模块的 Recall，不适合评估 Faithfulness。

### CMedQA（医疗）

```python
from datasets import load_dataset
dataset = load_dataset("cmedqa", split="train")
```

### CLAP（学术论文）

```python
from datasets import load_dataset
dataset = load_dataset("clap", split="train")
```

### 适配自己的数据

无论来源如何，统一转换为标准格式：

```python
import json

def convert_to_benchmark(data, source: str):
    """将任意数据集转换为 benchmark 格式"""
    qa_pairs = []
    for item in data:
        qa_pairs.append({
            "id": f"{source}_{len(qa_pairs):04d}",
            "question": item["question"],
            "ground_truth": item.get("answer", item.get("ground_truth", "")),
            "relevant_docs": item.get("documents", []),
            "difficulty": item.get("difficulty", "L1_factual"),
        })
    return qa_pairs
```

---

## 自定义领域数据集

如果你的 RAG 系统用于特定领域（法律、医疗、金融等），建议构造领域专属数据集。

### 步骤

1. **收集领域文档** → 放入 `data/docs/` 作为知识库
2. **提取关键知识点** → 从文档中提取 20-30 个关键事实
3. **构造 QA** → 为每个知识点构造 1-2 个问题
4. **标注难度** → 按上述 4 级难度标注

### 模板

```python
# scripts/build_custom_dataset.py
import json

qa_pairs = [
    {
        "id": "CUSTOM-001",
        "question": "【你的领域问题】",
        "ground_truth": "【标准答案】",
        "relevant_docs": ["【涉及的文件名】"],
        "difficulty": "L1_factual",
    },
]

with open("data/eval/qa_benchmark.json", "w", encoding="utf-8") as f:
    json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
```

---

## 生成测试 PDF/Word 的工具

如果你需要快速生成测试文件，可以使用以下 Python 脚本：

### 生成测试 PDF

```bash
pip install reportlab
```

```python
# scripts/generate_test_pdf.py
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

c = canvas.Canvas("data/benchmark/test_generated.pdf", pagesize=A4)
c.drawString(100, 750, "这是一个测试 PDF")
c.drawString(100, 730, "用于评估 Agentic RAG 的 PDF 解析能力")
c.save()
```

### 生成测试 Word

```bash
pip install python-docx
```

```python
from docx import Document
doc = Document()
doc.add_heading("测试文档", level=1)
doc.add_paragraph("这是一段测试文本。")
doc.add_table(rows=3, cols=3)
doc.save("data/benchmark/test_generated.docx")
```

### 生成测试 Excel

```python
import openpyxl
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "测试 Sheet"
ws.append(["姓名", "年龄", "城市"])
ws.append(["张三", 28, "北京"])
ws.append(["李四", 32, "上海"])
wb.save("data/benchmark/test_generated.xlsx")
```
