"""
Agentic RAG 系统配置
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class AgenticRAGConfig:
    # ========== 路径配置 ==========
    data_path: str = "./data/docs"
    index_save_path: str = "./data/index"
    image_save_path: str = "./data/images"

    # ========== 模型配置 ==========
    embedding_model: str = "text-embedding-v2"
    llm_model: str = "qwen-turbo"
    llm_provider: str = "dashscope"  # dashscope | openai | moonshot

    # ========== 检索配置 ==========
    top_k: int = 5
    bm25_k: int = 5
    rrf_k: int = 60

    # ========== Agent 配置 ==========
    max_iterations: int = 2
    enable_web_search: bool = False

    # ========== 生成配置 ==========
    temperature: float = 0.1
    max_tokens: int = 4096

    # ========== 文件解析配置 ==========
    supported_extensions: List[str] = field(default_factory=lambda: [
        ".txt", ".md", ".pdf", ".docx", ".xlsx", ".xls",
        ".png", ".jpg", ".jpeg", ".csv", ".json", ".log",
    ])


DEFAULT_CONFIG = AgenticRAGConfig()
