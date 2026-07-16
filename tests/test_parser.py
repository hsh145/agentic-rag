"""
解析器测试 + 成功率统计

测试覆盖：
  ✓ 纯文本解析 (.txt, .md, .json, .csv)
  ✓ Markdown 结构保留
  ✓ 图片解析（OCR + VLM，如依赖可用）
  ✓ PDF 解析（如依赖可用）
  ✓ Office 解析（如依赖可用）
  ✓ 错误处理（文件不存在、格式不支持）
  ✓ 耗时统计
"""

import time
from pathlib import Path
from typing import Optional

import pytest

from conftest import BENCHMARK_DIR, check_package
from parser.detector import FileTypeDetector
from parser.text_parser import TextParser
from parser.pdf_parser import PDFParser


# ============================================================
# 基础测试：文本解析器
# ============================================================

class TestTextParser:
    """文本解析器测试"""

    @pytest.fixture
    def parser(self):
        return TextParser()

    def test_parse_txt(self, parser):
        """解析 .txt 文件"""
        file_path = str(BENCHMARK_DIR / "sample.txt")
        docs = parser.parse(file_path)
        assert len(docs) > 0, "txt 解析应返回至少一个文档"
        assert len(docs[0].page_content) > 50, "txt 内容不应为空"

    def test_parse_markdown(self, parser):
        """解析 .md 文件，验证 Markdown 结构保留"""
        file_path = str(BENCHMARK_DIR / "sample.md")
        docs = parser.parse(file_path)
        assert len(docs) > 0, "md 解析应返回至少一个文档"
        content = docs[0].page_content
        assert "# SFT" in content, "Markdown 标题应被保留"
        assert "| 参数" in content, "Markdown 表格应被保留"
        assert "```json" in content, "代码块应被保留"

    def test_parse_json(self, parser):
        """解析 .json 文件"""
        file_path = str(BENCHMARK_DIR / "sample.json")
        docs = parser.parse(file_path)
        assert len(docs) > 0, "json 解析应返回文档"
        assert "learning_rate" in docs[0].page_content, "JSON 内容应被提取"

    def test_parse_nonexistent_file(self, parser):
        """测试不存在的文件应抛出异常"""
        with pytest.raises(FileNotFoundError):
            parser.parse("/path/to/nonexistent/file.txt")

    def test_file_type_detection(self):
        """验证文件类型检测"""
        assert FileTypeDetector.detect("test.txt") == "text"
        assert FileTypeDetector.detect("test.md") == "markdown"
        assert FileTypeDetector.detect("test.pdf") == "pdf"
        assert FileTypeDetector.detect("test.docx") == "docx"
        assert FileTypeDetector.detect("test.xlsx") == "xlsx"
        assert FileTypeDetector.detect("test.png") == "image"
        assert FileTypeDetector.detect("test.unknown") == "unknown"


# ============================================================
# 条件和集成测试（需可选依赖）
# ============================================================

class TestPDFParser:
    """PDF 解析器测试（需要 PyMuPDF）"""

    @pytest.fixture
    def parser(self):
        return PDFParser()

    def test_parse_pdf_file_not_found(self, parser):
        """PDF 文件不存在应报错"""
        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent.pdf")

    @pytest.mark.skipif(not check_package("fitz"), reason="需要 PyMuPDF")
    def test_parse_sample_pdf(self, parser):
        """解析 PDF 文件"""
        # 尝试查找项目中的 PDF 文件
        pdf_files = list(Path(BENCHMARK_DIR).glob("*.pdf"))
        if not pdf_files:
            pytest.skip("未找到 PDF 测试文件")
        docs = parser.parse(str(pdf_files[0]))
        assert len(docs) > 0


# ============================================================
# 解析成功率统计
# ============================================================

def test_parser_stats():
    """遍历所有 benchmark 文件，统计解析成功率"""
    if not BENCHMARK_DIR.exists():
        pytest.skip("benchmark 目录不存在")

    # 使用 DocumentParserTool（按文件类型自动选择解析器）
    from agent.tools import DocumentParserTool
    parser_tool = DocumentParserTool()
    results = {
        "total": 0,
        "success": 0,
        "failed": 0,
        "errors": [],
        "by_type": {},
    }

    for file_path in sorted(BENCHMARK_DIR.iterdir()):
        if not file_path.is_file():
            continue

        ftype = FileTypeDetector.detect(str(file_path))
        if ftype == "unknown":
            continue

        results["total"] += 1
        results["by_type"].setdefault(ftype, {"total": 0, "success": 0, "failed": 0})

        try:
            t0 = time.time()
            docs, errors = parser_tool.parse_files([str(file_path)])
            elapsed = time.time() - t0

            if docs and len(docs[0].page_content) > 0:
                results["success"] += 1
                results["by_type"][ftype]["success"] += 1
            elif errors:
                results["failed"] += 1
                results["by_type"][ftype]["failed"] += 1
                results["errors"].append(f"{file_path.name}: {errors[0]}（{elapsed:.2f}s）")
            else:
                results["failed"] += 1
                results["by_type"][ftype]["failed"] += 1
                results["errors"].append(f"{file_path.name}: 内容为空（{elapsed:.2f}s）")
        except Exception as e:
            results["failed"] += 1
            results["by_type"][ftype]["failed"] += 1
            results["errors"].append(f"{file_path.name}: {e}")

    # 输出统计
    success_rate = results["success"] / results["total"] * 100 if results["total"] else 0
    print(f"\n{'='*50}")
    print(f"  解析器成功率统计")
    print(f"{'='*50}")
    print(f"  总计: {results['total']} 文件")
    print(f"  成功: {results['success']} ({success_rate:.1f}%)")
    print(f"  失败: {results['failed']}")
    print(f"\n  按类型:")
    for ftype, stats in sorted(results["by_type"].items()):
        rate = stats["success"] / stats["total"] * 100 if stats["total"] else 0
        print(f"    {ftype:>10}: {stats['success']}/{stats['total']} ({rate:.1f}%)")
    if results["errors"]:
        print(f"\n  错误详情:")
        for err in results["errors"]:
            print(f"    - {err}")
    print(f"{'='*50}\n")

    assert success_rate >= 80, f"解析成功率 {success_rate:.1f}% < 80%"
