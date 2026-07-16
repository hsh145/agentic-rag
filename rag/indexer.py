"""
FAISS 索引管理 - 先算向量再建索引，兼容 DashScope API
"""
import logging
import os
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import faiss
from langchain_core.documents import Document

logger = logging.getLogger("rag.indexer")


class IndexManager:
    def __init__(self, embeddings, index_save_path: str = "./data/index"):
        self.embeddings = embeddings
        self.index_save_path = Path(index_save_path)
        self.index: Optional[faiss.Index] = None
        self.documents: List[Document] = []

    def build_index(self, chunks: List[Document]) -> None:
        """先算向量，再建 FAISS 索引"""
        logger.info(f"计算 {len(chunks)} 个文档块的向量...")
        texts = [c.page_content for c in chunks]
        vectors = self.embeddings.embed_documents(texts)
        embedding_dim = len(vectors[0])

        # 建 FAISS 索引
        matrix = np.array(vectors).astype("float32")
        self.index = faiss.IndexFlatIP(embedding_dim)  # 内积 = 余弦相似度（embedder.py 已做 L2 归一化）
        self.index.add(matrix)
        self.documents = chunks
        logger.info(f"FAISS 索引构建完成，共 {len(chunks)} 个向量，维度 {embedding_dim}")

    def add_documents(self, new_chunks: List[Document]):
        if self.index is None:
            raise ValueError("请先构建索引")
        texts = [c.page_content for c in new_chunks]
        vectors = self.embeddings.embed_documents(texts)
        matrix = np.array(vectors).astype("float32")
        self.index.add(matrix)
        self.documents.extend(new_chunks)
        logger.info(f"已添加 {len(new_chunks)} 个新文档")

    def save_index(self):
        self.index_save_path.mkdir(parents=True, exist_ok=True)
        # 保存 FAISS 索引
        faiss.write_index(self.index, str(self.index_save_path / "index.faiss"))
        # 保存文档列表
        with open(self.index_save_path / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)
        logger.info(f"索引已保存: {self.index_save_path}")

    def load_index(self) -> bool:
        index_file = self.index_save_path / "index.faiss"
        docs_file = self.index_save_path / "documents.pkl"
        if not index_file.exists() or not docs_file.exists():
            return False
        try:
            self.index = faiss.read_index(str(index_file))
            with open(docs_file, "rb") as f:
                self.documents = pickle.load(f)
            logger.info(f"索引已加载: {self.index_save_path} ({self.index.ntotal} 个向量)")
            return True
        except Exception as e:
            logger.warning(f"加载索引失败: {e}")
            return False

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        if self.index is None or self.index.ntotal == 0:
            return []

        query_vec = self.embeddings.embed_query(query)
        query_np = np.array([query_vec]).astype("float32")

        scores, indices = self.index.search(query_np, min(k, self.index.ntotal))

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                doc.metadata["similarity_score"] = float(scores[0][i])
                results.append(doc)
        return results
