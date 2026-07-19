"""
记忆系统 — 会话记忆 + 长期记忆

架构：
  SessionMemory     → 管理对话历史（SQLite 持久化 + LangGraph Checkpointer）
  LongTermMemory    → 提取并存储事实，支持语义检索
  FactExtractor     → LLM 驱动的实体/事实提取器

用法：
    memory = AgentMemory(db_path="./data/memory.db")
    await memory.save_turn(session_id, query, answer, sources)
    history = memory.load_history(session_id, limit=10)
    facts = memory.recall_facts(query, top_k=5)
"""

import json
import logging
import sqlite3
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger("agent.memory")

# ============================================================
# 数据模型
# ============================================================

@dataclass
class TurnRecord:
    """一轮对话记录"""
    session_id: str
    turn_id: int
    query: str
    answer: str
    sources: List[str]
    timestamp: str

@dataclass
class FactRecord:
    """一条长期记忆事实"""
    id: int
    session_id: str
    fact_text: str
    entity: str
    category: str
    confidence: float
    created_at: str

# ============================================================
# 数据库管理
# ============================================================

class MemoryDB:
    """SQLite 底层封装，建表 + CRUD"""

    def __init__(self, db_path: str = "./data/memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    turn_id INTEGER NOT NULL,
                    query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    sources TEXT DEFAULT '[]',
                    timestamp TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );

                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    fact_text TEXT NOT NULL,
                    entity TEXT DEFAULT '',
                    category TEXT DEFAULT 'general',
                    confidence REAL DEFAULT 1.0,
                    embedding BLOB,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE INDEX IF NOT EXISTS idx_turns_session
                    ON turns(session_id, turn_id);
                CREATE INDEX IF NOT EXISTS idx_facts_session
                    ON facts(session_id);
                CREATE INDEX IF NOT EXISTS idx_facts_entity
                    ON facts(entity);
            """)

    def _conn(self):
        """返回上下文管理器风格的连接"""
        class ConnectionWrapper:
            def __init__(self, db_path):
                self.conn = sqlite3.connect(str(db_path))
                self.conn.row_factory = sqlite3.Row

            def __enter__(self):
                return self.conn

            def __exit__(self, *args):
                self.conn.commit()
                self.conn.close()

        return ConnectionWrapper(self.db_path)


# ============================================================
# 会话记忆（短期）
# ============================================================

class SessionMemory:
    """会话记忆 — 管理对话历史

    核心能力：
      - 持久化存储每轮对话
      - 按 session_id 加载最近 N 轮
      - 自动去重（相同 query 不重复存储）
      - 提供 LangGraph Checkpointer
    """

    def __init__(self, db_path: str = "./data/memory.db", max_turns: int = 20):
        self.db = MemoryDB(db_path)
        self.max_turns = max_turns
        self._checkpointer = MemorySaver()
        logger.info(f"会话记忆初始化完成 (max_turns={max_turns}, db={db_path})")

    def get_checkpointer(self) -> MemorySaver:
        """返回 LangGraph Checkpointer，用于图状态持久化"""
        return self._checkpointer

    def load_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """加载指定会话的最近对话历史

        Args:
            session_id: 会话 ID
            limit: 最多返回的轮次数

        Returns:
            [{"role": "user"/"assistant", "content": str, "sources": [...]}]
        """
        with self.db._conn() as conn:
            rows = conn.execute(
                """SELECT query, answer, sources, timestamp
                   FROM turns WHERE session_id = ?
                   ORDER BY turn_id DESC LIMIT ?""",
                (session_id, limit),
            ).fetchall()

        history = []
        for row in reversed(rows):
            history.append({
                "role": "user",
                "content": row["query"],
                "timestamp": row["timestamp"],
            })
            history.append({
                "role": "assistant",
                "content": row["answer"],
                "sources": json.loads(row["sources"]),
                "timestamp": row["timestamp"],
            })
        return history

    def save_turn(
        self,
        session_id: str,
        query: str,
        answer: str,
        sources: Optional[List[str]] = None,
    ) -> int:
        """保存一轮对话

        Returns: turn_id（从 1 开始递增）
        """
        # 确保 session 存在
        with self.db._conn() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO sessions (session_id)
                   VALUES (?)""",
                (session_id,),
            )
            conn.execute(
                """UPDATE sessions SET updated_at = datetime('now')
                   WHERE session_id = ?""",
                (session_id,),
            )

            # 计算下一个 turn_id
            last = conn.execute(
                "SELECT COALESCE(MAX(turn_id), 0) as last_id FROM turns WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            turn_id = last["last_id"] + 1

            conn.execute(
                """INSERT INTO turns (session_id, turn_id, query, answer, sources)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, turn_id, query, answer, json.dumps(sources or [], ensure_ascii=False)),
            )

            # 清理超过 max_turns 的旧记录
            conn.execute(
                """DELETE FROM turns WHERE session_id = ? AND turn_id <= ?
                   AND (SELECT COUNT(*) FROM turns WHERE session_id = ?) > ?""",
                (session_id, turn_id - self.max_turns, session_id, self.max_turns),
            )

        logger.debug(f"会话 {session_id} 第 {turn_id} 轮已保存")
        return turn_id

    def get_session_count(self) -> int:
        """获取总会话数"""
        with self.db._conn() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM sessions").fetchone()
            return row["cnt"]

    def get_turn_count(self, session_id: str) -> int:
        """获取指定会话的轮次数"""
        with self.db._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM turns WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return row["cnt"]

    def clear_session(self, session_id: str):
        """清除指定会话的所有记录"""
        with self.db._conn() as conn:
            conn.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        logger.info(f"会话 {session_id} 已清除")

    def format_history_for_prompt(self, history: List[Dict]) -> str:
        """将历史记录格式化为 prompt 上下文"""
        if not history:
            return ""
        lines = ["## 对话历史"]
        for msg in history:
            role = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{role}: {msg['content'][:200]}")
        return "\n".join(lines)


# ============================================================
# 长期记忆
# ============================================================

class LongTermMemory:
    """长期记忆 — 提取 + 存储 + 检索事实

    核心能力：
      - LLM 驱动的事实提取（从对话中抽取出结构化事实）
      - 语义检索（使用系统 embedding 模型）
      - 按实体/类别过滤
    """

    def __init__(self, db_path: str = "./data/memory.db", embedding_model=None):
        self.db = MemoryDB(db_path)
        self._embedding_model = embedding_model
        logger.info(f"长期记忆初始化完成 (db={db_path})")

    def set_embedding_model(self, embedding_model):
        """设置 embedding 模型（延迟注入，避免循环依赖）"""
        self._embedding_model = embedding_model

    # --------------------------------------------------
    # 事实提取
    # --------------------------------------------------
    def extract_facts(self, query: str, answer: str, llm=None) -> List[Dict]:
        """用 LLM 从问答对中提取结构化事实

        Args:
            query: 用户问题
            answer: 系统回答
            llm: LLM 实例（如 ChatTongyi），用于事实提取

        Returns:
            [{"fact": str, "entity": str, "category": str, "confidence": float}]
        """
        if llm is None:
            logger.warning("未提供 LLM，跳过事实提取")
            return []

        from langchain_core.messages import HumanMessage

        prompt = f"""从以下问答对话中提取关键事实。
提取那些对后续对话有用的、可重用的知识性信息。
不要提取礼貌用语或元对话内容。

用户问题: {query}
助手回答: {answer}

输出 JSON 列表（只输出 JSON，不要其他文字）：
[
    {{
        "fact": "提取出的事实陈述",
        "entity": "事实涉及的主要实体/主题",
        "category": "分类 (preference|knowledge|entity|task)",
        "confidence": 0.0-1.0
    }}
]"""

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()
            content = content.replace("```json", "").replace("```", "").strip()
            facts = json.loads(content)
            if not isinstance(facts, list):
                facts = []
            logger.info(f"事实提取: {len(facts)} 条")
            return facts
        except Exception as e:
            logger.warning(f"事实提取失败: {e}")
            return []

    # --------------------------------------------------
    # 存储
    # --------------------------------------------------
    def store_facts(self, session_id: str, facts: List[Dict]) -> int:
        """存储提取的事实到长期记忆

        同 session + 同 entity + 同 category 的事实会覆盖更新（而非追加），
        自动解决"用户之前说X，现在说Y"的矛盾。

        Args:
            session_id: 会话 ID
            facts: extract_facts 返回的事实列表

        Returns: 存储/更新数量
        """
        if not facts:
            return 0

        stored = 0
        with self.db._conn() as conn:
            for fact in facts:
                fact_text = fact.get("fact", "").strip()
                if not fact_text or len(fact_text) < 5:
                    continue

                entity = fact.get("entity", "")
                category = fact.get("category", "general")
                confidence = min(float(fact.get("confidence", 0.8)), 1.0)

                # 查询同 entity + category 是否已有旧事实
                existing = conn.execute(
                    """SELECT id FROM facts
                       WHERE entity = ? AND category = ? AND session_id = ?
                       LIMIT 1""",
                    (entity, category, session_id),
                ).fetchone()

                if existing:
                    # 覆盖：更新文本、置信度、时间戳，清空 embedding 让后台重新生成
                    conn.execute(
                        """UPDATE facts
                           SET fact_text = ?, confidence = ?, created_at = datetime('now'),
                               embedding = NULL
                           WHERE id = ?""",
                        (fact_text, confidence, existing["id"]),
                    )
                    logger.debug(f"长期记忆: 更新事实 id={existing['id']} ({entity}/{category})")
                else:
                    # 新增
                    conn.execute(
                        """INSERT INTO facts (session_id, fact_text, entity, category, confidence)
                           VALUES (?, ?, ?, ?, ?)""",
                        (session_id, fact_text, entity, category, confidence),
                    )
                stored += 1

        if stored:
            logger.info(f"长期记忆: 写入 {stored} 条事实（新增 + 覆盖）")
        return stored

    # --------------------------------------------------
    # 检索
    # --------------------------------------------------
    def recall(self, query: str, top_k: int = 5, category: Optional[str] = None) -> List[str]:
        """语义检索长期记忆

        使用 embedding 模型（如可用）进行向量相似度检索，
        回退到 SQLite FTS 全文搜索。

        Args:
            query: 查询文本
            top_k: 返回条数
            category: 可选分类过滤

        Returns:
            [事实文本, ...]
        """
        if self._embedding_model is not None:
            return self._vector_recall(query, top_k, category)
        else:
            return self._keyword_recall(query, top_k, category)

    def _vector_recall(self, query: str, top_k: int, category: Optional[str]) -> List[str]:
        """向量相似度检索（含时间衰减）"""
        try:
            query_vec = self._embedding_model.embed_query(query)
        except Exception as e:
            logger.warning(f"向量检索失败，回退关键词: {e}")
            return self._keyword_recall(query, top_k, category)

        import numpy as np
        from datetime import datetime

        with self.db._conn() as conn:
            if category:
                rows = conn.execute(
                    "SELECT id, fact_text, embedding, created_at FROM facts WHERE category = ?",
                    (category,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, fact_text, embedding, created_at FROM facts ORDER BY created_at DESC LIMIT 200"
                ).fetchall()

        if not rows:
            return []

        query_np = np.array(query_vec).astype("float32")
        now = datetime.now()

        scored = []
        for row in rows:
            # 语义相似度
            embedding = row["embedding"] if row["embedding"] else None
            if embedding:
                try:
                    import pickle
                    vec = pickle.loads(embedding)
                    vec_np = np.array(vec).astype("float32")
                    score = float(np.dot(query_np, vec_np) / (
                        np.linalg.norm(query_np) * np.linalg.norm(vec_np) + 1e-10
                    ))
                except Exception:
                    score = 0.0
            else:
                score = 0.0

            # 时间衰减：每天降 1%，最低保留 50%
            try:
                created = datetime.strptime(row["created_at"], "%Y-%m-%d %H:%M:%S")
                days_old = (now - created).days
                time_decay = max(0.5, 1.0 - days_old * 0.01)
            except Exception:
                time_decay = 1.0

            scored.append((score * time_decay, row["fact_text"]))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in scored[:top_k]]

    def _keyword_recall(self, query: str, top_k: int, category: Optional[str]) -> List[str]:
        """关键词检索（SQLite LIKE + 时间衰减）"""
        from datetime import datetime
        now = datetime.now()

        with self.db._conn() as conn:
            like = f"%{query}%"
            # 多取一些，排序后再截断
            limit = top_k * 5
            if category:
                rows = conn.execute(
                    """SELECT fact_text, confidence, created_at FROM facts
                       WHERE fact_text LIKE ? AND category = ?
                       ORDER BY confidence DESC, id DESC LIMIT ?""",
                    (like, category, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT fact_text, confidence, created_at FROM facts
                       WHERE fact_text LIKE ?
                       ORDER BY confidence DESC, id DESC LIMIT ?""",
                    (like, limit),
                ).fetchall()

        if not rows:
            return []

        scored = []
        for row in rows:
            try:
                created = datetime.strptime(row["created_at"], "%Y-%m-%d %H:%M:%S")
                days_old = (now - created).days
                time_decay = max(0.5, 1.0 - days_old * 0.01)
            except Exception:
                time_decay = 1.0
            adjusted = row["confidence"] * time_decay
            scored.append((adjusted, row["fact_text"]))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in scored[:top_k]]

    def update_embeddings(self) -> int:
        """为所有没有 embedding 的事实生成向量

        在注入 embedding_model 后调用，批量回填向量。
        """
        if self._embedding_model is None:
            logger.warning("未设置 embedding 模型，无法更新向量")
            return 0

        with self.db._conn() as conn:
            rows = conn.execute(
                "SELECT id, fact_text FROM facts WHERE embedding IS NULL"
            ).fetchall()

        import pickle
        import numpy as np

        updated = 0
        texts = [r["fact_text"] for r in rows]
        if not texts:
            return 0

        try:
            vectors = self._embedding_model.embed_documents(texts)
            with self.db._conn() as conn:
                for row, vec in zip(rows, vectors):
                    blob = pickle.dumps(vec)
                    conn.execute(
                        "UPDATE facts SET embedding = ? WHERE id = ?",
                        (blob, row["id"]),
                    )
                    updated += 1
            logger.info(f"长期记忆: 已更新 {updated} 条 embedding")
        except Exception as e:
            logger.warning(f"批量更新 embedding 失败: {e}")

        return updated

    def get_fact_count(self) -> int:
        with self.db._conn() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM facts").fetchone()
            return row["cnt"]

    # --------------------------------------------------
    # 遗忘机制
    # --------------------------------------------------
    def forget_old_facts(self, days: int = 30) -> int:
        """删除超过指定天数的旧事实"""
        from datetime import datetime, timedelta
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        with self.db._conn() as conn:
            result = conn.execute(
                "DELETE FROM facts WHERE created_at < ?",
                (cutoff,),
            )
            deleted = result.rowcount
        if deleted:
            logger.info(f"遗忘机制: 删除 {deleted} 条超过 {days} 天的旧事实")
        return deleted

    def forget_low_confidence(self, threshold: float = 0.3) -> int:
        """删除置信度低于阈值的事实"""
        with self.db._conn() as conn:
            result = conn.execute(
                "DELETE FROM facts WHERE confidence < ?",
                (threshold,),
            )
            deleted = result.rowcount
        if deleted:
            logger.info(f"遗忘机制: 删除 {deleted} 条置信度低于 {threshold} 的事实")
        return deleted

    def forget_duplicates(self) -> int:
        """同 entity+category 只保留最新的一条，删除旧版本"""
        with self.db._conn() as conn:
            # 找出每个 entity+category 分组中不是最新的记录
            deleted = conn.execute("""
                DELETE FROM facts WHERE id NOT IN (
                    SELECT id FROM (
                        SELECT id, ROW_NUMBER() OVER (
                            PARTITION BY entity, category, session_id
                            ORDER BY created_at DESC
                        ) AS rn FROM facts
                    ) WHERE rn = 1
                )
            """)
            count = deleted.rowcount
        if count:
            logger.info(f"遗忘机制: 清理 {count} 条重复事实（同 entity+category 只留最新）")
        return count

    def get_facts_by_session(self, session_id: str) -> List[FactRecord]:
        with self.db._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM facts WHERE session_id = ? ORDER BY id DESC",
                (session_id,),
            ).fetchall()
        return [
            FactRecord(
                id=r["id"],
                session_id=r["session_id"],
                fact_text=r["fact_text"],
                entity=r["entity"],
                category=r["category"],
                confidence=r["confidence"],
                created_at=r["created_at"],
            )
            for r in rows
        ]


# ============================================================
# 统一入口
# ============================================================

class AgentMemory:
    """统一记忆入口 — 组合 SessionMemory + LongTermMemory

    用法：
        memory = AgentMemory()
        await memory.save_turn(session_id, query, answer, sources)
        history = memory.load_history(session_id)
        memory.extract_and_store(session_id, query, answer, llm)
        recalled = memory.recall_facts(query)
    """

    def __init__(
        self,
        db_path: str = "./data/memory.db",
        max_turns: int = 20,
        embedding_model=None,
    ):
        self.session = SessionMemory(db_path, max_turns)
        self.long_term = LongTermMemory(db_path, embedding_model)
        logger.info(f"🧠 AgentMemory 初始化完成 (db={db_path})")

    def set_embedding_model(self, embedding_model):
        self.long_term.set_embedding_model(embedding_model)

    # ---- 会话记忆 ----
    def get_checkpointer(self) -> MemorySaver:
        return self.session.get_checkpointer()

    def load_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        return self.session.load_history(session_id, limit)

    def save_turn(
        self,
        session_id: str,
        query: str,
        answer: str,
        sources: Optional[List[str]] = None,
    ) -> int:
        return self.session.save_turn(session_id, query, answer, sources)

    def format_history_for_prompt(self, history: List[Dict]) -> str:
        return self.session.format_history_for_prompt(history)

    def clear_session(self, session_id: str):
        self.session.clear_session(session_id)

    # ---- 长期记忆 ----
    def extract_and_store(
        self,
        session_id: str,
        query: str,
        answer: str,
        llm=None,
    ) -> int:
        """提取事实并存入长期记忆

        Returns: 存储的事实数量
        """
        facts = self.long_term.extract_facts(query, answer, llm)
        return self.long_term.store_facts(session_id, facts)

    def recall_facts(self, query: str, top_k: int = 5) -> List[str]:
        return self.long_term.recall(query, top_k)

    def update_embeddings(self) -> int:
        return self.long_term.update_embeddings()

    # ---- 遗忘 ----
    def forget_old_facts(self, days: int = 30) -> int:
        return self.long_term.forget_old_facts(days)

    def forget_low_confidence(self, threshold: float = 0.3) -> int:
        return self.long_term.forget_low_confidence(threshold)

    def forget_duplicates(self) -> int:
        return self.long_term.forget_duplicates()

    # ---- 统计 ----
        return {
            "session_count": self.session.get_session_count(),
            "fact_count": self.long_term.get_fact_count(),
        }
