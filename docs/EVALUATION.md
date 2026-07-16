# Agentic RAG 评估指南

本文档说明如何对该系统的各项指标进行量化评估。

---

## 目录

- [指标体系总览](#指标体系总览)
- [解析器评估](#解析器评估)
- [RAG 检索评估](#rag-检索评估)
- [端到端性能评估](#端到端性能评估)
- [CI 集成](#ci-集成)
- [评估报告解读](#评估报告解读)

---

## 指标体系总览

| 类别 | 指标 | 工具 | 自动/手动 |
|------|------|------|-----------|
| 解析器 | Parse Success Rate | pytest + test_parser.py | 自动 |
| 解析器 | Text Extraction Rate | pytest + test_parser.py | 自动 |
| 解析器 | Avg Parse Time | pytest + test_parser.py | 自动 |
| RAG | Faithfulness | RAGAS / DeepEval | 自动（需 LLM judge） |
| RAG | Answer Relevancy | RAGAS / DeepEval | 自动（需 LLM judge） |
| RAG | Context Recall | RAGAS / DeepEval | 自动（需 LLM judge） |
| RAG | Context Precision | RAGAS / DeepEval | 自动（需 LLM judge） |
| 性能 | 端到端 P50/P95/P99 | scripts/run_benchmark.py | 自动 |
| 性能 | 组件延迟拆解 | scripts/run_benchmark.py | 自动 |
| 压测 | 并发吞吐 | Locust | 手动启动 |
| 记忆 | 会话保持准确率 | pytest + test_memory.py | 自动 |

---

## 解析器评估

### 运行方式

```bash
# 快速测试
pytest tests/test_parser.py -v

# 含成功率统计
pytest tests/test_parser.py -v -k "test_parser_stats"
```

### 工作原理

1. 遍历 `data/benchmark/` 下的所有文件
2. 自动检测文件类型并调用对应解析器
3. 统计每个文件的解析耗时、文档数、字符数
4. 汇总成功率（阈值：≥80%）

### 测试文件集

| 文件 | 类型 | 测试重点 |
|------|------|---------|
| sample.txt | 纯文本 | 基础文本提取 |
| sample.md | Markdown | 标题/表格/代码块结构保留 |
| sample.json | JSON | 结构化数据解析 |

> **需要添加更多格式？**
> 在 `data/benchmark/` 下放入 PDF、Word、Excel、图片等文件即可自动纳入测试。

---

## RAG 检索评估

### 方式一：RAGAS（标准报告）

```bash
pip install ragas datasets

# 使用默认 OpenAI judge
python tests/eval_ragas.py

# 使用本地 judge (qwen-turbo)
python tests/eval_ragas_local.py
```

RAGAS 会自动计算 4 个核心指标并输出：

```
=========================================================
  RAGAS 评估结果
=========================================================
             faithfulness: 0.8432 (±0.1021)
            answer_relevancy: 0.9012 (±0.0654)
              context_recall: 0.7654 (±0.1521)
           context_precision: 0.8210 (±0.0987)
=========================================================
```

### 方式二：DeepEval（CI 集成）

```bash
pip install deepeval

# 作为 pytest 测试运行
pytest tests/eval_deepeval.py -v

# 生成 DeepEval dashboard
deepeval login  # 可选，用于云 dashboard
```

DeepEval 的优势：
- 天然 pytest 集成，直接作为 CI 断言
- 内置 Hallucination、Bias、Toxicity 等高级指标
- 支持自定义 judge LLM

### 评估数据集

QA pair 保存在 `data/eval/qa_benchmark.json`，共 18 条，覆盖 4 个难度级别：

| 级别 | 说明 | 数量 |
|------|------|------|
| L1 事实性 | 单文档直接查询 | 5 |
| L2 比较性 | 两方案对比 | 5 |
| L3 推理 | 需要多步推理 | 5 |
| L4 跨文档 | 综合多文档 | 3 |

每条数据包含：
- `question` — 用户问题
- `ground_truth` — 标准答案
- `relevant_docs` — 相关文档列表
- `expected_topics` — 期望涵盖的主题词

---

## 端到端性能评估

### 一键基准测试

```bash
# 全量测试（解析器 + RAG + 端到端性能）
python scripts/run_benchmark.py --all

# 仅端到端性能
python scripts/run_benchmark.py --perf-only

# 仅解析器
python scripts/run_benchmark.py --parser-only

# 仅 RAG 检索
python scripts/run_benchmark.py --rag-only
```

输出示例：

```
=========================================================
  Agentic RAG 基准测试
=========================================================

[1/3] 解析器基准...
  📄 解析器性能基准
  文件数: 3 | 成功: 3 | 失败: 0
  平均解析耗时: 0.012s

[2/3] RAG 检索基准...
  🔍 RAG 检索性能
  索引构建: 0.523s (10 文档)
  检索 P50: 0.042s

[3/3] 端到端性能基准...
  ⚡ 端到端性能基准
  P50: 3.21s
  P95: 5.87s
  P99: 8.12s
```

### Locust 负载测试

```bash
pip install locust

# Web UI 模式
locust -f scripts/locustfile.py --host=http://localhost:8000

# 无头模式（5 用户并发，运行 60 秒）
locust -f scripts/locustfile.py --host=http://localhost:8000 \
       --headless -u 5 -r 1 --run-time 60s
```

---

## CI 集成

### GitHub Actions 示例

在 `.github/workflows/eval.yml` 中配置：

```yaml
name: RAG Evaluation
on: [push, pull_request]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - name: Run parser tests
        run: pytest tests/test_parser.py -v
      - name: Run memory tests
        run: pytest tests/test_memory.py -v
      - name: Run RAG evaluation (DeepEval)
        run: pytest tests/eval_deepeval.py -v
        env:
          DASHSCOPE_API_KEY: ${{ secrets.DASHSCOPE_API_KEY }}
```

### 本地一键全跑

```bash
# 自行组合
pytest tests/test_parser.py -v && \
pytest tests/test_memory.py -v && \
python scripts/run_benchmark.py --all
```

---

## 评估报告解读

### Faithfulness（忠实度）

> **含义**：系统生成的回答是否忠实地基于检索到的上下文，没有编造内容。

- **0.9~1.0**：优秀。回答完全基于上下文
- **0.7~0.9**：良好。大部分内容基于上下文
- **< 0.7**：需要检查。可能存在幻觉

### Answer Relevancy（回答相关性）

> **含义**：生成的回答是否直接回应了用户的问题。

- **0.9~1.0**：优秀。回答切题
- **0.7~0.9**：良好。基本切题
- **< 0.7**：需要改进。可能答非所问

### Context Recall（上下文召回率）

> **含义**：所有相关的文档块是否都被检索到了。

- **0.8~1.0**：优秀。检索覆盖全面
- **0.6~0.8**：良好。大部分相关文档已检索
- **< 0.6**：需要优化检索策略

### Context Precision（上下文精确度）

> **含义**：检索到的文档块中有多少是真正相关的。

- **0.8~1.0**：优秀。检索结果噪声少
- **0.6~0.8**：良好。有少量噪声
- **< 0.6**：需要优化。检索结果不够精确

---

## 指标趋势跟踪

每次运行评估后，结果会自动保存：

| 工具 | 保存路径 |
|------|---------|
| RAGAS | `data/eval/ragas_result.csv` |
| 本地评估 | `data/eval/local_eval_report.json` |
| 性能基准 | `data/eval/benchmark_report.md` |

你可以将这些文件纳入版本控制，追踪指标变化趋势。
