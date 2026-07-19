"""
Agentic RAG 系统配置 — 含 Memory 配置项
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class AgenticRAGConfig:
    # ========== 路径配置 ==========
    data_path: str = "./data/docs"
    index_save_path: str = "./data/index"
    image_save_path: str = "./data/images"
    memory_db_path: str = "./data/memory.db"

    # ========== 模型配置 ==========
    embedding_model: str = "text-embedding-v2"
    llm_model: str = "qwen-turbo"
    llm_provider: str = "dashscope"  # dashscope | openai | moonshot

    # ---- 本地模型配置（微调/蒸馏后启用） ----
    use_local_judge: bool = False           # 是否用本地模型替代 qwen-turbo 做判断
    local_judge_model_path: str = ""        # 例如 "./models/qwen-judge"
    local_judge_device: str = "cpu"         # cpu 或 cuda
    use_local_embedding: bool = False       # 是否用本地 embedding 替代 text-embedding-v2
    local_embedding_model_path: str = ""    # 例如 "./models/bge-small-zh"

    fact_extraction_model: str = "qwen-turbo"  # 事实提取用模型（轻量即可）

    # ========== 检索配置 ==========
    top_k: int = 5
    bm25_k: int = 5
    rrf_k: int = 60

    # ========== Reranker 配置（可选，默认关闭）==========
    enable_rerank: bool = False             # 是否启用交叉编码器重排
    rerank_model: str = "BAAI/bge-reranker-v2-m3"  # 模型名称（或本地路径）
    rerank_top_k: int = 10                  # 对前 N 个候选重排
    rerank_device: str = "cpu"              # cpu 或 cuda

    # ========== Agent 配置 ==========
    max_iterations: int = 2
    enable_web_search: bool = False

    # ========== MQE 多查询扩展（可选，默认关闭）==========
    enable_mqe: bool = False             # 是否启用多查询扩展
    mqe_expansions: int = 3              # 每个子查询扩展为几个变体

    # ========== 生成配置 ==========
    temperature: float = 0.1
    max_tokens: int = 4096

    # ========== 记忆配置 ==========
    memory_max_turns: int = 20          # 会话记忆保留的最大轮次
    memory_recall_topk: int = 3         # 长期记忆召回数量
    enable_long_term_memory: bool = True  # 是否启用长期记忆

    # ========== 文件解析配置 ==========
    supported_extensions: List[str] = field(default_factory=lambda: [
        ".txt", ".md", ".pdf", ".docx", ".xlsx", ".xls",
        ".png", ".jpg", ".jpeg", ".csv", ".json", ".log",
    ])

    # ========== 评估配置 ==========
    eval_judge_model: str = "qwen-max"   # 评估时作为 judge 的 LLM
    faithfulness_threshold: float = 0.7  # faithfulness 合格线
    recall_threshold: float = 0.6        # recall 合格线


DEFAULT_CONFIG = AgenticRAGConfig()
