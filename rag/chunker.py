"""
语义分块 - 支持多类型文档的自适应分块
无外部依赖，纯 Python 实现。
"""
import logging
import re
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger("rag.chunker")


class SemanticChunker:
    """多类型自适应分块器（纯 Python，无 torch/sentence_transformers 依赖）"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # --------------------------------------------------
    # Markdown 分块：按标题层级分割
    # --------------------------------------------------
    def _chunk_markdown(self, doc: Document) -> List[Document]:
        content = doc.page_content
        # 按 ## 标题分割
        sections = re.split(r'\n(?=#{1,3}\s)', content)
        result = []
        for i, section in enumerate(sections):
            if not section.strip():
                continue
            # 如果段落太长，递归用 text 分块
            if len(section) > self.chunk_size * 1.5:
                text_chunks = self._split_text(section)
                for j, tc in enumerate(text_chunks):
                    chunk = Document(
                        page_content=tc,
                        metadata={**doc.metadata, "chunk_id": f"md_{id(doc)}_{i}_{j}", "chunk_index": i},
                    )
                    result.append(chunk)
            else:
                chunk = Document(
                    page_content=section,
                    metadata={**doc.metadata, "chunk_id": f"md_{id(doc)}_{i}", "chunk_index": i},
                )
                result.append(chunk)
        return result

    # --------------------------------------------------
    # 文本分块：按分隔符分割 + 重叠
    # --------------------------------------------------
    def _split_text(self, text: str) -> List[str]:
        """将文本按分隔符分割成块，每块不超过 chunk_size"""
        separators = ["\n\n", "\n", "。", "；", "，", " ", ""]
        return self._split_with_separators(text, separators, self.chunk_size, self.chunk_overlap)

    def _split_with_separators(
        self, text: str, separators: List[str], chunk_size: int, overlap: int
    ) -> List[str]:
        """递归分割：先用第一个分隔符切，超长的块递归用下一个分隔符"""
        if not text or not text.strip():
            return []

        if len(text) <= chunk_size:
            return [text]

        sep = separators[0]
        rest_seps = separators[1:] if len(separators) > 1 else [""]

        # 用当前分隔符分割
        if sep:
            parts = text.split(sep)
        else:
            # 最后一级：按字符切
            parts = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]
            return parts

        # 合并短段落
        merged = []
        current = ""
        for part in parts:
            if not part.strip():
                continue
            if not current:
                current = part
            elif len(current) + len(sep) + len(part) <= chunk_size:
                current += sep + part
            else:
                merged.append(current)
                current = part
        if current:
            merged.append(current)

        # 超长的块递归处理
        result = []
        for m in merged:
            if len(m) <= chunk_size:
                result.append(m)
            else:
                result.extend(self._split_with_separators(m, rest_seps, chunk_size, overlap))
        return result

    def _chunk_text(self, doc: Document) -> List[Document]:
        chunks = self._split_text(doc.page_content)
        result = []
        for i, c in enumerate(chunks):
            chunk = Document(
                page_content=c,
                metadata={**doc.metadata, "chunk_id": f"txt_{id(doc)}_{i}", "chunk_index": i},
            )
            result.append(chunk)
        return result

    # --------------------------------------------------
    # 代码分块：按函数/类定义分割
    # --------------------------------------------------
    def _chunk_code(self, doc: Document) -> List[Document]:
        content = doc.page_content
        # 按 def/class 分割
        sections = re.split(r'\n(?=def |class |async def )', content)
        result = []
        for i, section in enumerate(sections):
            if not section.strip():
                continue
            if len(section) > self.chunk_size:
                sub_chunks = self._split_text(section)
                for j, sc in enumerate(sub_chunks):
                    chunk = Document(
                        page_content=sc,
                        metadata={**doc.metadata, "chunk_id": f"code_{id(doc)}_{i}_{j}", "chunk_index": i},
                    )
                    result.append(chunk)
            else:
                chunk = Document(
                    page_content=section,
                    metadata={**doc.metadata, "chunk_id": f"code_{id(doc)}_{i}", "chunk_index": i},
                )
                result.append(chunk)
        return result

    # --------------------------------------------------
    # 对外接口
    # --------------------------------------------------
    def chunk_document(self, doc: Document) -> List[Document]:
        """根据文档类型选择分块策略"""
        source_type = doc.metadata.get("source_type", "text")
        content = doc.page_content

        if not content or not content.strip():
            return []

        if source_type in ("markdown", "md"):
            return self._chunk_markdown(doc)
        elif source_type == "code":
            return self._chunk_code(doc)
        elif source_type in ("table", "image_description"):
            doc.metadata["chunk_id"] = f"{source_type[:3]}_{id(doc)}"
            return [doc]
        else:
            return self._chunk_text(doc)

    def chunk_all(self, documents: List[Document]) -> List[Document]:
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        logger.info(f"分块完成: {len(documents)} 文档 → {len(all_chunks)} 块")
        return all_chunks
