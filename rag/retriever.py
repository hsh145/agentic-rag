"""
混合检索 - 向量 + BM25 + RRF 融合
"""
import logging
from typing import List, Dict, Any

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from .indexer import IndexManager

logger = logging.getLogger("rag.retriever")


class HybridRetriever:
    def __init__(self, index_mgr: IndexManager, chunks: List[Document], rrf_k: int = 60):
        self.index_mgr = index_mgr
        self.chunks = chunks
        self.rrf_k = rrf_k
        self.bm25_retriever = None
        if chunks:
            self.bm25_retriever = BM25Retriever.from_documents(chunks, k=5)
            logger.info(f"BM25 检索器就绪 ({len(chunks)} 文档)")

    def hybrid_search(self, query: str, top_k: int = 3) -> List[Document]:
        # 向量检索
        vector_docs = self.index_mgr.similarity_search(query, k=top_k * 2)

        # BM25 检索
        bm25_docs = []
        if self.bm25_retriever:
            try:
                bm25_docs = self.bm25_retriever.invoke(query)
            except Exception as e:
                logger.warning(f"BM25 检索失败: {e}")

        if not vector_docs and not bm25_docs:
            return []
        if not vector_docs:
            return bm25_docs[:top_k]
        if not bm25_docs:
            return vector_docs[:top_k]

        return self._rrf_rerank(vector_docs, bm25_docs)[:top_k]

    def metadata_filtered_search(
        self, query: str, filters: Dict[str, Any], top_k: int = 5
    ) -> List[Document]:
        docs = self.hybrid_search(query, top_k=top_k * 3)
        filtered = []
        for doc in docs:
            match = True
            for key, value in filters.items():
                if doc.metadata.get(key) != value:
                    match = False
                    break
            if match:
                filtered.append(doc)
                if len(filtered) >= top_k:
                    break
        return filtered

    def _rrf_rerank(
        self, vector_docs: List[Document], bm25_docs: List[Document]
    ) -> List[Document]:
        scores = {}
        objects = {}
        k = self.rrf_k

        for rank, doc in enumerate(vector_docs):
            did = hash(doc.page_content[:200])
            objects[did] = doc
            scores[did] = scores.get(did, 0) + 1.0 / (k + rank + 1)

        for rank, doc in enumerate(bm25_docs):
            did = hash(doc.page_content[:200])
            objects[did] = doc
            scores[did] = scores.get(did, 0) + 1.0 / (k + rank + 1)

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        result = []
        for did, score in sorted_docs:
            doc = objects[did]
            doc.metadata["rrf_score"] = score
            result.append(doc)

        return result
