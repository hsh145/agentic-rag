"""
Reranker — 交叉编码器精排（可选组件）

作用：
  在 FAISS + BM25 + RRF 粗排之后，用交叉编码器对 Top-N 候选做第二遍精排。
  相比向量余弦相似度，交叉编码器能捕捉 query-doc 间的深层语义匹配。

用法：
    reranker = Reranker(model_name="BAAI/bge-reranker-v2-m3", enabled=True)
    reranked_docs = reranker.rerank(query, docs)

设计原则：
  - 可选：enabled=False 时不加载模型，直接返回原列表
  - 容错：模型加载失败或推理异常，自动回退到原排序
  - 延迟加载：模型在第一次 rerank 调用时才初始化
"""

import logging
from typing import List, Optional

from langchain_core.documents import Document

logger = logging.getLogger("rag.reranker")


class Reranker:
    """交叉编码器精排器

    Args:
        model_name: 模型名称或本地路径（默认 bge-reranker-v2-m3）
        enabled: 是否启用（默认 False，跳过全部推理）
        top_k: 对前 top_k 个候选重排
        device: 推理设备（cpu / cuda）
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        enabled: bool = False,
        top_k: int = 10,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.enabled = enabled
        self.top_k = top_k
        self.device = device
        self._model = None  # 延迟加载

    # --------------------------------------------------
    # 内部：延迟加载模型
    # --------------------------------------------------
    def _load_model(self):
        if self._model is not None:
            return True

        if not self.enabled:
            return False

        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                trust_remote_code=True,
            )
            logger.info(
                f"Reranker 模型已加载: {self.model_name} (device={self.device})"
            )
            return True
        except ImportError:
            logger.warning(
                "sentence-transformers 未安装，reranker 已禁用。"
                "如需启用: pip install sentence-transformers"
            )
            self.enabled = False
            return False
        except Exception as e:
            logger.warning(f"Reranker 模型加载失败，已禁用: {e}")
            self.enabled = False
            return False

    # --------------------------------------------------
    # 对外接口
    # --------------------------------------------------
    def rerank(
        self,
        query: str,
        documents: List[Document],
    ) -> List[Document]:
        """对文档列表做交叉编码器精排

        Args:
            query: 用户查询
            documents: 待重排的文档列表

        Returns:
            重排后的文档列表（带 rerank_score 元数据）
        """
        if not documents or not self.enabled:
            return documents

        if not self._load_model():
            return documents

        # 只重排前 top_k 个候选（保持尾部不变）
        candidates = documents[: self.top_k]
        tail = documents[self.top_k :]

        try:
            # 构造 (query, doc) 对
            pairs = [(query, doc.page_content) for doc in candidates]
            scores = self._model.predict(pairs)

            # 将分数附加到文档并重排序
            for doc, score in zip(candidates, scores):
                doc.metadata["rerank_score"] = float(score)

            candidates.sort(
                key=lambda d: d.metadata.get("rerank_score", 0),
                reverse=True,
            )

            logger.debug(
                f"Reranker: {len(candidates)} 个文档重排完成, "
                f"top-1 score={candidates[0].metadata['rerank_score']:.4f}"
            )
        except Exception as e:
            logger.warning(f"Reranker 推理失败，跳过重排: {e}")
            return documents

        return candidates + tail
