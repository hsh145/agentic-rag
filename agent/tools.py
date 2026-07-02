"""
Agent 工具 - 文档解析 + RAG 检索 + 搜索
"""
import logging
from typing import List, Optional

from langchain_core.documents import Document

from parser import FileTypeDetector, PDFParser, OfficeParser, ImageParser, TextParser
from rag import HybridRetriever

logger = logging.getLogger("agent.tools")


class DocumentParserTool:
    """文档解析工具：自动检测类型并调用对应解析器"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.pdf_parser = PDFParser()
        self.office_parser = OfficeParser()
        self.image_parser = ImageParser(api_key=api_key)
        self.text_parser = TextParser()

    def parse_files(self, file_paths: List[str]) -> tuple:
        """解析文件列表，返回 (documents, errors)"""
        all_docs = []
        errors = []

        for fp in file_paths:
            ftype = FileTypeDetector.detect(fp)
            try:
                if ftype == "pdf":
                    docs = self.pdf_parser.parse(fp)
                elif ftype in ("docx", "xlsx", "xls"):
                    docs = self.office_parser.parse(fp)
                elif ftype == "image":
                    docs = self.image_parser.parse(fp)
                elif ftype in ("text", "markdown"):
                    docs = self.text_parser.parse(fp)
                else:
                    errors.append(f"不支持的文件类型: {fp}")
                    continue
                all_docs.extend(docs)
            except Exception as e:
                errors.append(f"解析失败 {fp}: {e}")

        return all_docs, errors


class RAGRetrievalTool:
    """RAG 检索工具"""

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def search(self, query: str, top_k: int = 3) -> List[Document]:
        return self.retriever.hybrid_search(query, top_k=top_k)


class EvidenceEvaluator:
    """证据评估工具 - 判断检索结果是否充分"""

    @staticmethod
    def evaluate(chunks: List[Document], query: str) -> tuple:
        """
        返回 (scores_dict, feedback_str, needs_more)
        """
        if not chunks:
            return {}, "未检索到任何相关内容", True

        # 简单统计：chunk 数量和内容长度作为线索
        total_chars = sum(len(c.page_content) for c in chunks)
        has_table = any(c.metadata.get("source_type") == "table" for c in chunks)
        has_image = any(c.metadata.get("source_type") == "image_description" for c in chunks)

        feedback_parts = []
        feedback_parts.append(f"检索到 {len(chunks)} 个相关文档块，共 {total_chars} 字")
        if has_table:
            feedback_parts.append("包含表格数据")
        if has_image:
            feedback_parts.append("包含图片描述")

        # 简单判断是否需要补搜
        needs_more = False
        if len(chunks) < 2 and total_chars < 300:
            needs_more = True
            feedback_parts.append("→ 信息不足，建议补搜")

        scores = {
            "chunk_count": len(chunks),
            "total_chars": total_chars,
            "has_table": int(has_table),
            "has_image": int(has_image),
        }
        return scores, "；".join(feedback_parts), needs_more
