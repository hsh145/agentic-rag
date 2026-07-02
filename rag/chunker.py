"""
语义分块 - 支持多类型文档的自适应分块
所有 sentence_transformers/sklearn 依赖使用延迟导入
"""
import logging
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger("rag.chunker")


class SemanticChunker:
    """多类型自适应分块器"""

    def __init__(self):
        # 延迟导入，避免加载 sklearn/torch
        from langchain_text_splitters import (
            RecursiveCharacterTextSplitter,
            MarkdownHeaderTextSplitter,
        )
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "header1"),
                ("##", "header2"),
                ("###", "header3"),
            ],
            strip_headers=False,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=64,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        )
        self.code_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\ndef ", "\nclass ", "\n    def ", "\n\n", "\n", " "],
        )

    # ... 其余方法不变 ...
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
        elif source_type == "table":
            doc.metadata["chunk_id"] = f"table_{id(doc)}"
            return [doc]
        elif source_type == "image_description":
            doc.metadata["chunk_id"] = f"img_{id(doc)}"
            return [doc]
        else:
            return self._chunk_text(doc)

    def _chunk_markdown(self, doc: Document) -> List[Document]:
        try:
            chunks = self.md_splitter.split_text(doc.page_content)
        except Exception:
            chunks = self.text_splitter.create_documents([doc.page_content])

        result = []
        for i, chunk in enumerate(chunks):
            chunk.metadata.update(doc.metadata)
            chunk.metadata["chunk_id"] = f"md_{id(doc)}_{i}"
            chunk.metadata["chunk_index"] = i
            result.append(chunk)
        return result

    def _chunk_code(self, doc: Document) -> List[Document]:
        chunks = self.code_splitter.create_documents([doc.page_content])
        for i, chunk in enumerate(chunks):
            chunk.metadata.update(doc.metadata)
            chunk.metadata["chunk_id"] = f"code_{id(doc)}_{i}"
            chunk.metadata["chunk_index"] = i
        return chunks

    def _chunk_text(self, doc: Document) -> List[Document]:
        chunks = self.text_splitter.create_documents([doc.page_content])
        for i, chunk in enumerate(chunks):
            chunk.metadata.update(doc.metadata)
            chunk.metadata["chunk_id"] = f"txt_{id(doc)}_{i}"
            chunk.metadata["chunk_index"] = i
        return chunks

    def chunk_all(self, documents: List[Document]) -> List[Document]:
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        logger.info(f"分块完成: {len(documents)} 文档 → {len(all_chunks)} 块")
        return all_chunks
