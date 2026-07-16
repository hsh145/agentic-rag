"""
pytest 共享夹具 — 测试配置、mock、benchmark 路径
用法：
    pytest tests/ -v
"""
import os
import sys
import json
from pathlib import Path

import pytest

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# 路径常量
# ============================================================
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
DOCS_DIR = PROJECT_ROOT / "data" / "docs"


# ============================================================
# 夹具
# ============================================================

@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def benchmark_dir() -> Path:
    return BENCHMARK_DIR


@pytest.fixture(scope="session")
def eval_dir() -> Path:
    return EVAL_DIR


@pytest.fixture(scope="session")
def benchmark_files() -> list:
    """返回所有 benchmark 文件的路径列表"""
    if not BENCHMARK_DIR.exists():
        return []
    return sorted(BENCHMARK_DIR.iterdir())


@pytest.fixture(scope="session")
def qa_benchmark() -> list:
    """加载 QA benchmark 数据集"""
    qa_file = EVAL_DIR / "qa_benchmark.json"
    if not qa_file.exists():
        return []
    with open(qa_file, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def sample_query() -> str:
    return "什么是SFT微调？"


@pytest.fixture
def mock_env(monkeypatch):
    """注入测试用环境变量（不依赖真实 API Key）"""
    # 如果已有 API Key 就用真的，否则用 mock
    key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not key:
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test_mock_key")
    return key


# ============================================================
# Helper 函数
# ============================================================

def count_lines(text: str) -> int:
    """统计文本行数"""
    return len(text.strip().split("\n")) if text.strip() else 0


def check_package(package_name: str) -> bool:
    """检查 Python 包是否已安装"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False
