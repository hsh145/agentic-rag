"""
嵌入模型管理 - 使用 DashScope 在线 embedding API
"""
import logging
import os
from typing import List

logger = logging.getLogger("rag.embedder")


class DashScopeEmbeddings:
    """DashScope text-embedding-v2 封装"""

    def __init__(self, api_key: str = "", model: str = "text-embedding-v2"):
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.model = model

    def _call(self, input_texts):
        """统一调用 DashScope Embedding API（显式传 api_key，线程安全）"""
        import dashscope
        resp = dashscope.TextEmbedding.call(
            model=self.model,
            input=input_texts,
            api_key=self.api_key,  # 显式传参，避免线程安全问题
        )
        return resp

    def embed_query(self, text: str) -> List[float]:
        resp = self._call(text)
        if resp.status_code == 200:
            return resp.output["embeddings"][0]["embedding"]
        raise RuntimeError(f"Embedding API 失败: {resp}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        import logging
        logger = logging.getLogger("rag.embedder")
        logger.info(f"embed_documents: model={self.model}, api_key_len={len(self.api_key)}, texts_count={len(texts)}")
        resp = self._call(texts)
        logger.info(f"embed_documents resp: status={resp.status_code}")
        if resp.status_code == 200:
            return [e["embedding"] for e in resp.output["embeddings"]]
        raise RuntimeError(f"Embedding API 失败: {resp}")


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
