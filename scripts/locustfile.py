"""
Locust 负载测试 — 模拟多用户并发访问

启动：
    # 安装 locust
    pip install locust

    # 启动测试（Web UI 模式）
    locust -f scripts/locustfile.py --host=http://localhost:8000

    # 无头模式（命令行输出）
    locust -f scripts/locustfile.py --host=http://localhost:8000 \
           --headless -u 5 -r 1 --run-time 60s

参数：
    -u 5        # 模拟 5 个并发用户
    -r 1        # 每秒启动 1 个用户
    --run-time 60s  # 运行 60 秒

参考：
    https://docs.locust.io/en/stable/writing-a-locustfile.html
"""

import random
from locust import HttpUser, task, between


class RAGUser(HttpUser):
    """模拟 RAG 系统用户"""

    # 用户等待时间（1~5 秒）
    wait_time = between(1, 5)

    def on_start(self):
        """用户启动时执行"""
        self.session_id = f"locust_{self.id}_{random.randint(1000, 9999)}"

    # --------------------------------------------------
    # 查询集
    # --------------------------------------------------
    SHORT_QUERIES = [
        "什么是SFT微调？",
        "LoRA的全称是什么？",
        "FAISS是什么？",
        "什么是RLHF？",
        "Batch size推荐范围是多少？",
    ]

    LONG_QUERIES = [
        "LoRA和Full Fine-tuning的主要区别是什么？请详细说明。",
        "对比SFT和RLHF两种训练方法的优缺点",
        "为什么LoRA比Full Fine-tuning省显存？原理是什么？",
        "在模型微调中SFT和RLHF的顺序可以调换吗？为什么？",
    ]

    FILE_QUERIES = [
        "总结这份文档的核心内容",
        "文档中提到了哪些关键技术？",
    ]

    # --------------------------------------------------
    # 任务定义
    # --------------------------------------------------
    @task(5)
    def simple_query(self):
        """简单的单轮查询（高频）"""
        query = random.choice(self.SHORT_QUERIES)
        payload = {
            "query": query,
            "max_iterations": 1,
            "session_id": self.session_id,
        }
        with self.client.post(
            "/api/ask",
            json=payload,
            name="simple_query",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if data.get("success"):
                    resp.success()
                else:
                    resp.failure(f"API 返回错误: {data.get('error', 'unknown')}")
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @task(3)
    def complex_query(self):
        """复杂多轮查询（中频）"""
        query = random.choice(self.LONG_QUERIES)
        payload = {
            "query": query,
            "max_iterations": 3,
            "session_id": self.session_id,
        }
        with self.client.post(
            "/api/ask",
            json=payload,
            name="complex_query",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if data.get("success"):
                    resp.success()
                else:
                    resp.failure(f"API 返回错误: {data.get('error', 'unknown')}")
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @task(1)
    def health_check(self):
        """健康检查（低频）"""
        with self.client.get(
            "/api/health",
            name="health_check",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @task(1)
    def memory_stats(self):
        """记忆统计（低频）"""
        with self.client.get(
            "/api/memory/stats",
            name="memory_stats",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @task(2)
    def session_query(self):
        """带 session 的连续查询（模拟真实用户多轮对话）"""
        # 第一轮：简单问题
        q1 = random.choice(self.SHORT_QUERIES)
        with self.client.post(
            "/api/ask",
            json={"query": q1, "session_id": self.session_id, "max_iterations": 1},
            name="session_turn_1",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200 and resp.json().get("success"):
                resp.success()
            else:
                resp.failure(f"第一轮失败")
                return

        # 第二轮：追问（利用 session 记忆）
        followups = [
            "刚才说的能再详细解释一下吗？",
            "那实际应用中应该怎么选？",
            "有没有具体的代码示例？",
        ]
        q2 = random.choice(followups)
        with self.client.post(
            "/api/ask",
            json={"query": q2, "session_id": self.session_id, "max_iterations": 2},
            name="session_turn_2",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200 and resp.json().get("success"):
                resp.success()
            else:
                resp.failure(f"第二轮失败")
