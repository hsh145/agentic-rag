"""
API 端点测试 — FastAPI 健康检查、提问、记忆

用法：
    pytest tests/test_api.py -v
    # 或启动服务后：
    pytest tests/test_api.py -v --live

注意：
    默认使用 mock 模式（不依赖后端服务）。
    使用 --live 参数测试真实运行中的后端。
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import pytest

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================
# Mock 模式：使用 TestClient
# ============================================================

try:
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


# ============================================================
# Mock 模式测试（不需要启动后端）
# ============================================================

@pytest.mark.skipif(not HAS_FASTAPI, reason="需要 fastapi testclient")
class TestAPIHealth:
    """API 健康检查测试（mock 模式）"""

    @pytest.fixture(scope="class")
    def client(self):
        from main import app
        return TestClient(app)

    def test_root_endpoint(self, client):
        """GET / 应返回服务信息"""
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "Agentic RAG"
        assert "version" in data
        assert "endpoints" in data

    def test_health_endpoint(self, client):
        """GET /api/health 应返回健康状态"""
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "memory" in data
        assert "api_key_configured" in data

    def test_memory_stats_endpoint(self, client):
        """GET /api/memory/stats 应返回记忆统计"""
        resp = client.get("/api/memory/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "data" in data


@pytest.mark.skipif(not HAS_FASTAPI, reason="需要 fastapi testclient")
class TestAPIAskValidation:
    """API /api/ask 输入校验测试"""

    @pytest.fixture(scope="class")
    def client(self):
        from main import app
        return TestClient(app)

    def test_missing_query(self, client):
        """缺少 query 应返回 422"""
        resp = client.post("/api/ask", json={})
        assert resp.status_code == 422

    def test_empty_query(self, client):
        """query 为空字符串应返回 422 或处理"""
        resp = client.post("/api/ask", json={"query": ""})
        assert resp.status_code in (200, 422)  # 200 表示系统处理了空查询

    def test_invalid_json(self, client):
        """非法的 JSON 请求体"""
        resp = client.post("/api/ask", data="not json", headers={"Content-Type": "application/json"})
        assert resp.status_code == 422

    def test_max_iterations_validation(self, client):
        """max_iterations 超出范围应返回 422"""
        resp = client.post("/api/ask", json={
            "query": "test",
            "max_iterations": 99,
        })
        assert resp.status_code == 422

    def test_valid_request_structure(self, client):
        """合法请求应返回结构化响应"""
        resp = client.post("/api/ask", json={
            "query": "什么是SFT？",
            "max_iterations": 2,
        })
        # 虽然可能没有 API Key 导致内部错误，但响应结构应正确
        assert resp.status_code == 200
        data = resp.json()
        assert "success" in data
        assert "answer" in data
        assert "session_id" in data


@pytest.mark.skipif(not HAS_FASTAPI, reason="需要 fastapi testclient")
class TestAPIMemory:
    """API 记忆相关测试"""

    @pytest.fixture(scope="class")
    def client(self):
        from main import app
        return TestClient(app)

    def test_clear_memory(self, client):
        """DELETE /api/memory/{session_id} 应清除记忆"""
        resp = client.delete("/api/memory/test_session")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True


# ============================================================
# Live 模式测试（需要后端运行）
# ============================================================

@pytest.mark.skipif("--live" not in sys.argv, reason="使用 --live 运行真实后端测试")
class TestAPILive:
    """真实后端测试（依赖运行中的服务）"""

    @pytest.fixture(scope="class")
    def base_url(self):
        return "http://localhost:8000"

    @pytest.fixture(scope="class")
    def client(self, base_url):
        import requests
        return requests.Session()

    def test_live_health(self, base_url, client):
        resp = client.get(f"{base_url}/api/health", timeout=5)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_live_ask(self, base_url, client):
        resp = client.post(
            f"{base_url}/api/ask",
            json={"query": "什么是SFT微调？", "max_iterations": 1},
            timeout=60,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert "memory_stats" in data

    def test_live_session_continuity(self, base_url, client):
        """同一 session_id 应记住上下文"""
        session_id = "test_session_continuity"

        resp1 = client.post(
            f"{base_url}/api/ask",
            json={"query": "我的名字是张三", "session_id": session_id},
            timeout=60,
        )
        assert resp1.status_code == 200

        resp2 = client.post(
            f"{base_url}/api/ask",
            json={"query": "我叫什么名字？", "session_id": session_id},
            timeout=60,
        )
        assert resp2.status_code == 200
        # 第二次应能记住名字
        assert "张三" in resp2.json().get("answer", "")

    def test_live_memory_stats(self, base_url, client):
        resp = client.get(f"{base_url}/api/memory/stats", timeout=5)
        assert resp.status_code == 200

    def test_live_clear_memory(self, base_url, client):
        resp = client.delete(f"{base_url}/api/memory/test_session_continuity", timeout=5)
        assert resp.status_code == 200
