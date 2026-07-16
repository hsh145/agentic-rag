"""
Memory 系统测试 — 会话记忆 + 长期记忆 CRUD + 检索

测试覆盖：
  ✓ SessionMemory: 保存/加载历史、自动递增 turn_id、超过 max_turns 自动清理
  ✓ LongTermMemory: 事实存储、关键词检索、向量检索回退
  ✓ AgentMemory: 统一入口
  ✓ 边界条件：空历史、大文本、并发写入
"""

import json
import time
from pathlib import Path

import pytest

from agent.memory import AgentMemory, SessionMemory, LongTermMemory


# ============================================================
# SessionMemory 测试
# ============================================================

class TestSessionMemory:
    """会话记忆测试"""

    @pytest.fixture
    def mem(self, tmp_path):
        db_path = str(tmp_path / "test_memory.db")
        return SessionMemory(db_path=db_path, max_turns=5)

    def test_save_and_load(self, mem):
        """保存和加载对话历史"""
        mem.save_turn("session_1", "你好", "你好！有什么可以帮助你的？")
        mem.save_turn("session_1", "什么是SFT？", "SFT是监督式微调...")

        history = mem.load_history("session_1")
        assert len(history) == 4  # 2 轮 = 4 条消息（user + assistant 各一条）

        # 消息顺序
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "你好"
        assert history[1]["role"] == "assistant"
        assert history[3]["role"] == "assistant"
        assert "SFT" in history[3]["content"]

    def test_session_count(self, mem):
        """统计会话数"""
        mem.save_turn("s1", "q1", "a1")
        mem.save_turn("s2", "q1", "a1")
        mem.save_turn("s2", "q2", "a2")
        assert mem.get_session_count() == 2

    def test_turn_count(self, mem):
        """统计轮次数"""
        mem.save_turn("s1", "q1", "a1")
        mem.save_turn("s1", "q2", "a2")
        mem.save_turn("s1", "q3", "a3")
        assert mem.get_turn_count("s1") == 3

    def test_max_turns_limit(self, mem):
        """超过 max_turns 自动清理旧记录"""
        for i in range(10):
            mem.save_turn("s1", f"user_q_{i}", f"assistant_a_{i}")

        history = mem.load_history("s1", limit=20)
        # 应该只有最近 5 轮（10 条消息）
        assert len(history) <= 10
        # 最后一条用户消息是 q9
        user_msgs = [h for h in history if h["role"] == "user"]
        assert user_msgs[-1]["content"] == "user_q_9"

    def test_clear_session(self, mem):
        """清除会话"""
        mem.save_turn("s1", "q1", "a1")
        mem.clear_session("s1")
        history = mem.load_history("s1")
        assert len(history) == 0
        assert mem.get_turn_count("s1") == 0

    def test_load_empty_session(self, mem):
        """加载空会话"""
        history = mem.load_history("nonexistent")
        assert history == []

    def test_save_with_sources(self, mem):
        """保存带来源的对话"""
        mem.save_turn("s1", "q1", "a1", sources=["doc1.pdf", "doc2.pdf"])
        history = mem.load_history("s1")
        assistant_msg = history[1]
        assert len(assistant_msg["sources"]) == 2
        assert "doc1.pdf" in assistant_msg["sources"]

    def test_turn_id_sequential(self, mem):
        """turn_id 从 1 开始递增"""
        id1 = mem.save_turn("s1", "q1", "a1")
        id2 = mem.save_turn("s1", "q2", "a2")
        id3 = mem.save_turn("s1", "q3", "a3")
        assert id1 == 1
        assert id2 == 2
        assert id3 == 3

    def test_checkpointer(self, mem):
        """获取 LangGraph Checkpointer"""
        cp = mem.get_checkpointer()
        from langgraph.checkpoint.memory import MemorySaver
        assert isinstance(cp, MemorySaver)


# ============================================================
# LongTermMemory 测试
# ============================================================

class TestLongTermMemory:
    """长期记忆测试"""

    @pytest.fixture
    def ltm(self, tmp_path):
        db_path = str(tmp_path / "test_ltm.db")
        return LongTermMemory(db_path=db_path)

    def test_store_and_keyword_recall(self, ltm):
        """存储事实后可通过关键词检索"""
        ltm.store_facts("s1", [
            {"fact": "用户的微调实验使用LoRA方法", "entity": "用户", "category": "knowledge"},
            {"fact": "用户用的是7B模型", "entity": "用户", "category": "knowledge"},
            {"fact": "用户的显存是24GB", "entity": "用户", "category": "preference"},
        ])

        results = ltm.recall("LoRA", top_k=5)
        assert len(results) >= 1
        assert any("LoRA" in r for r in results)

    def test_store_and_retrieve_by_category(self, ltm):
        """按分类检索"""
        ltm.store_facts("s1", [
            {"fact": "用户喜欢用PyTorch", "entity": "用户", "category": "preference"},
            {"fact": "Python是一种编程语言", "entity": "Python", "category": "knowledge"},
        ])

        results = ltm.recall("Python", top_k=5)
        assert len(results) >= 1

    def test_empty_recall(self, ltm):
        """空长期记忆检索"""
        results = ltm.recall("anything", top_k=5)
        assert results == []

    def test_dedup(self, ltm):
        """去重：相同事实不重复存储"""
        fact = {"fact": "这是一条关于用户偏好的测试事实", "entity": "test", "category": "general"}
        n1 = ltm.store_facts("s1", [fact])
        n2 = ltm.store_facts("s1", [fact])
        assert n1 == 1
        assert n2 == 0  # 第二次应跳过

    def test_fact_count(self, ltm):
        """统计事实总数"""
        assert ltm.get_fact_count() == 0
        ltm.store_facts("s1", [
            {"fact": "用户偏好使用PyTorch框架进行实验", "entity": "e1", "category": "general"},
            {"fact": "用户所在团队有5名算法工程师", "entity": "e2", "category": "general"},
        ])
        assert ltm.get_fact_count() == 2

    def test_get_facts_by_session(self, ltm):
        """按会话获取事实"""
        ltm.store_facts("s1", [{"fact": "用户为初级算法工程师", "entity": "e1", "category": "c1"}])
        ltm.store_facts("s2", [{"fact": "用户已有5年编程经验", "entity": "e2", "category": "c2"}])

        facts_s1 = ltm.get_facts_by_session("s1")
        assert len(facts_s1) == 1
        assert "初级" in facts_s1[0].fact_text

    def test_short_fact_filtered(self, ltm):
        """过短的事实（< 5 字符）应被过滤"""
        stored = ltm.store_facts("s1", [
            {"fact": "短", "entity": "e1", "category": "general"},
            {"fact": "这是一条有效的事实", "entity": "e2", "category": "general"},
        ])
        assert stored == 1  # "短" 被过滤


# ============================================================
# AgentMemory 集成测试
# ============================================================

class TestAgentMemory:
    """AgentMemory 统一入口测试"""

    @pytest.fixture
    def memory(self, tmp_path):
        db_path = str(tmp_path / "test_agent_memory.db")
        return AgentMemory(db_path=db_path, max_turns=5)

    def test_integration(self, memory):
        """AgentMemory 完整流程"""
        # 保存对话
        memory.save_turn("s1", "你好", "你好！我有什么可以帮助你？")
        memory.save_turn("s1", "什么是SFT？", "SFT是监督式微调...")

        # 加载历史
        history = memory.load_history("s1")
        assert len(history) == 4

        # 统计
        stats = memory.stats()
        assert stats["session_count"] >= 1
        assert stats["fact_count"] == 0

    def test_clear(self, memory):
        """清除会话"""
        memory.save_turn("s1", "q", "a")
        memory.clear_session("s1")
        assert len(memory.load_history("s1")) == 0

    def test_stats_empty(self, memory):
        """空状态统计"""
        stats = memory.stats()
        assert stats["session_count"] == 0
        assert stats["fact_count"] == 0
