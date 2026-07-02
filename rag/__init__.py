"""
RAG 包初始化 — 使用延迟导入避免 sklearn/torch 内存问题
"""
from .embedder import EmbeddingManager
from .indexer import IndexManager
from .retriever import HybridRetriever

# SemanticChunker 在使用时动态导入，避免 sentence_transformers 提前加载
def get_chunker():
    from .chunker import SemanticChunker
    return SemanticChunker

__all__ = ["EmbeddingManager", "IndexManager", "HybridRetriever", "get_chunker"]
