"""
嵌入模型管理 - 使用 DashScope 在线 embedding API（含显式 L2 归一化）
"""
import logging
import os
from typing import List

import numpy as np

logger = logging.getLogger("rag.embedder")


class DashScopeEmbeddings:
    """DashScope text-embedding-v2 封装

    所有返回的向量经过 L2 归一化，确保与 FAISS IndexFlatIP（内积=余弦相似度）兼容。
    """

    def __init__(self, api_key: str = "", model: str = "text-embedding-v2"):
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.base_url = os.environ.get("DASHSCOPE_BASE_URL", "")
        self.model = model

    # --------------------------------------------------
    # 显式 L2 归一化（保证内积 = 余弦相似度）
    # --------------------------------------------------
    @staticmethod
    def _l2_normalize(vector: List[float]) -> List[float]:
        arr = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()

    @staticmethod
    def _l2_normalize_batch(vectors: List[List[float]]) -> List[List[float]]:
        arr = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        arr = arr / norms
        return arr.tolist()

    def _call(self, input_texts):
        """统一调用 DashScope Embedding API（显式传 api_key/url，线程安全）"""
        import dashscope
        kwargs = dict(
            model=self.model,
            input=input_texts,
            api_key=self.api_key,
        )
        if self.base_url:
            kwargs["base_url"] = self.base_url
        resp = dashscope.TextEmbedding.call(**kwargs)
        return resp

    def embed_query(self, text: str) -> List[float]:
        resp = self._call(text)
        if resp.status_code == 200:
            vec = resp.output["embeddings"][0]["embedding"]
            return self._l2_normalize(vec)
        raise RuntimeError(f"Embedding API 失败: {resp}")

    def embed_documents(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """批量向量化，自动分批避免 API 超时

        Args:
            texts: 文本列表
            batch_size: 每批数量（DashScope 推荐 ≤10）

        Returns:
            向量列表
        """
        import logging
        logger = logging.getLogger("rag.embedder")
        logger.info(f"embed_documents: model={self.model}, texts_count={len(texts)}, batch_size={batch_size}")

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"  batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}: {len(batch)} texts")
            try:
                resp = self._call(batch)
            except Exception as e:
                logger.warning(f"  batch {i//batch_size + 1} failed: {e}")
                # 失败时退避重试一次
                import time
                time.sleep(5)
                resp = self._call(batch)

            if resp.status_code == 200:
                batch_vectors = [e["embedding"] for e in resp.output["embeddings"]]
                batch_vectors = self._l2_normalize_batch(batch_vectors)
                all_embeddings.extend(batch_vectors)
                logger.info(f"  batch {i//batch_size + 1} OK, got {len(batch_vectors)} vectors")
            else:
                raise RuntimeError(f"Embedding API 失败 (batch {i//batch_size + 1}): {resp}")

        logger.info(f"embed_documents 完成: {len(all_embeddings)} vectors")
        return all_embeddings


class EmbeddingManager:
    def __init__(self, model_name: str = "text-embedding-v2"):
        self.model_name = model_name
        self._embeddings = None

    def get_embeddings(self):
        if self._embeddings is None:
            key = os.environ.get("DASHSCOPE_API_KEY", "")
            logger.info(f"初始化 DashScope Embedding: {self.model_name}")
            self._embeddings = DashScopeEmbeddings(api_key=key, model=self.model_name)
        return self._embeddings
