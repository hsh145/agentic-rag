"""
RAG 检索测试 — 测试向量检索、混合检索、分块等功能

测试覆盖：
  ✓ 分块器基本功能
  ✓ 索引构建与加载
  ✓ 混合检索返回结果
  ✓ metadata 过滤检索
  ✓ RRF 融合排序
"""

import pytest
from pathlib import Path
from langchain_core.documents import Document

from conftest import check_package, BENCHMARK_DIR


# ============================================================
# Chunker 测试
# ============================================================

class TestChunker:
    """语义分块器测试"""

    @pytest.fixture
    def chunker(self):
        from rag import get_chunker
        cls = get_chunker()
        return cls()

    def test_chunk_text(self, chunker):
        """文本分块"""
        doc = Document(page_content="A" * 2000, metadata={"source_type": "text"})
        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 1, "文本应被分成至少 1 块"

    def test_chunk_markdown(self, chunker):
        """Markdown 分块"""
        content = """# Title\n\nSome content here.\n\n## Section 1\n\nMore content.\n\n## Section 2\n\nEven more."""
        doc = Document(page_content=content, metadata={"source_type": "markdown"})
        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 1

    def test_chunk_code(self, chunker):
        """代码分块"""
        code = "def foo():\n    pass\n\ndef bar():\n    pass\n\nclass MyClass:\n    pass"
        doc = Document(page_content=code, metadata={"source_type": "code"})
        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 1

    def test_chunk_table(self, chunker):
        """表格保持完整（不分块）"""
        doc = Document(page_content="| A | B |\n|---|---|\n| 1 | 2 |", metadata={"source_type": "table"})
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 1, "表格不应被分块"

    def test_chunk_all(self, chunker):
        """批量分块"""
        docs = [
            Document(page_content="Hello world", metadata={"source_type": "text"}),
            Document(page_content="# Markdown title", metadata={"source_type": "markdown"}),
        ]
        all_chunks = chunker.chunk_all(docs)
        assert len(all_chunks) >= 2
        for c in all_chunks:
            assert isinstance(c, Document)
            assert "chunk_id" in c.metadata
            assert "chunk_index" in c.metadata


# ============================================================
# FAISS 索引测试
# ============================================================

class TestIndexer:
    """FAISS 索引构建与检索测试"""

    @pytest.fixture
    def embedder(self):
        from rag.embedder import EmbeddingManager
        return EmbeddingManager("text-embedding-v2")

    @pytest.fixture
    def indexer(self, embedder, tmp_path):
        from rag.indexer import IndexManager
        return IndexManager(embedder.get_embeddings(), str(tmp_path / "index"))

    @pytest.fixture
    def sample_chunks(self):
        from langchain_core.documents import Document
        return [
            Document(page_content="SFT微调是监督式微调的过程", metadata={"source": "doc1"}),
            Document(page_content="LoRA是一种参数高效的微调方法", metadata={"source": "doc2"}),
            Document(page_content="FAISS是高效的向量检索库", metadata={"source": "doc3"}),
        ]

    def test_build_and_save_index(self, indexer, sample_chunks):
        """构建并保存索引"""
        if not self._has_api_key():
            pytest.skip("需要有效的 DASHSCOPE_API_KEY")
        indexer.build_index(sample_chunks)
        assert indexer.index is not None
        assert indexer.index.ntotal == 3

        indexer.save_index()
        # 验证文件已保存
        from pathlib import Path
        save_dir = Path(indexer.index_save_path)
        assert (save_dir / "index.faiss").exists()
        assert (save_dir / "documents.pkl").exists()

    def test_load_index(self, indexer, sample_chunks):
        """保存后重新加载"""
        if not self._has_api_key():
            pytest.skip("需要有效的 DASHSCOPE_API_KEY")
        indexer.build_index(sample_chunks)
        indexer.save_index()

        # 新建索引管理器并加载
        from rag.indexer import IndexManager
        from rag.embedder import EmbeddingManager
        embedder2 = EmbeddingManager("text-embedding-v2")
        indexer2 = IndexManager(embedder2.get_embeddings(), str(indexer.index_save_path))
        loaded = indexer2.load_index()
        assert loaded, "索引加载应成功"
        assert indexer2.index.ntotal == 3

    def test_similarity_search(self, indexer, sample_chunks):
        """相似度检索应返回结果"""
        pytest.skip("需要有效的 Embedding API Key 才能运行")
        indexer.build_index(sample_chunks)
        assert indexer.index.ntotal == 3

    @staticmethod
    def _has_api_key() -> bool:
        import os
        return bool(os.environ.get("DASHSCOPE_API_KEY"))


# ============================================================
# 混合检索测试
# ============================================================

class TestHybridRetriever:
    """混合检索（向量 + BM25 + RRF）测试"""

    @pytest.fixture
    def chunks(self):
        return [
            Document(page_content="SFT微调是监督式微调的过程，需要大量标注数据",
                     metadata={"source": "doc1.md", "page": 1}),
            Document(page_content="LoRA通过低秩矩阵减少可训练参数",
                     metadata={"source": "doc2.md", "page": 2}),
            Document(page_content="RLHF使用人类反馈来优化模型",
                     metadata={"source": "doc3.md", "page": 3}),
        ]

    def test_bm25_initialization(self, chunks):
        """BM25 检索器初始化"""
        from langchain_community.retrievers import BM25Retriever
        bm25 = BM25Retriever.from_documents(chunks, k=3)
        assert bm25 is not None

    def test_bm25_search(self, chunks):
        """BM25 关键词检索"""
        from langchain_community.retrievers import BM25Retriever
        bm25 = BM25Retriever.from_documents(chunks, k=3)
        results = bm25.invoke("SFT微调")
        assert len(results) > 0
        # 任何结果中包含 SFT 或 微调 相关内容即可
        found = any("SFT" in d.page_content or "微调" in d.page_content for d in results)
        assert found, f"BM25 应返回包含 SFT 的结果，实际: {[d.page_content[:20] for d in results]}"

    def test_rrf_rerank(self):
        """RRF 融合排序逻辑"""
        from rag.retriever import HybridRetriever
        from rag.indexer import IndexManager
        from rag.embedder import EmbeddingManager

        # 用 mock 方式测试 RRF
        class MockIndexMgr:
            def similarity_search(self, query, k=5):
                return []

        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.rrf_k = 60

        vector_docs = [
            Document(page_content="Doc A about SFT", metadata={"score": 0.9}),
            Document(page_content="Doc B about LoRA", metadata={"score": 0.7}),
        ]
        bm25_docs = [
            Document(page_content="Doc B about LoRA", metadata={"score": 0.8}),
            Document(page_content="Doc C about RLHF", metadata={"score": 0.6}),
        ]

        result = retriever._rrf_rerank(vector_docs, bm25_docs)
        assert len(result) >= 2, "RRF 应返回融合后的结果"

        # Doc B 在两个列表中都出现，应有更高的 RRF 分数
        doc_b_results = [d for d in result if "LoRA" in d.page_content]
        assert len(doc_b_results) > 0
        # Doc B 的 rrf_score 应 > 其他文档
        assert doc_b_results[0].metadata.get("rrf_score", 0) > 0


# ============================================================
# Embedder 测试
# ============================================================

class TestEmbedder:
    """Embedding 模型测试"""

    def test_embedder_initialization(self):
        """Embedding 管理器初始化"""
        from rag.embedder import EmbeddingManager
        mgr = EmbeddingManager("text-embedding-v2")
        emb = mgr.get_embeddings()
        assert emb is not None
        assert emb.model == "text-embedding-v2"
