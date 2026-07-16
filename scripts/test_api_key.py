"""
API Key 连通性测试
用法：python scripts/test_api_key.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从 .env 读取
from dotenv import load_dotenv
load_dotenv()

key = os.environ.get("DASHSCOPE_API_KEY", "")
base_url = os.environ.get("DASHSCOPE_BASE_URL", "")

print(f"Key prefix: {key[:10]}...")
print(f"Base URL: {base_url or '(none, using default)'}")

# 1. Embedding
import dashscope
kwargs = dict(model="text-embedding-v2", input="测试消息", api_key=key)
if base_url:
    kwargs["base_url"] = base_url
resp = dashscope.TextEmbedding.call(**kwargs)
if resp.status_code == 200:
    emb = resp.output["embeddings"][0]["embedding"]
    print(f"[PASS] Embedding -> 维度 {len(emb)}")
else:
    print(f"[FAIL] Embedding -> {resp.status_code} {resp.code}: {resp.message}")
    sys.exit(1)

# 2. LLM
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage
kwargs = dict(model="qwen-turbo", temperature=0.1, dashscope_api_key=key)
if base_url:
    kwargs["dashscope_api_base"] = base_url
llm = ChatTongyi(**kwargs)
resp = llm.invoke([HumanMessage(content="say hi back in 3 words")])
if resp.content:
    print(f"[PASS] LLM -> {resp.content.strip()}")
else:
    print(f"[FAIL] LLM -> empty response")
    sys.exit(1)
