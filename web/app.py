"""
Agentic RAG — Streamlit 前端
"""
import time
import requests
import streamlit as st
from pathlib import Path

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="Agentic RAG 智能检索",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 样式
# ============================================================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .main-header p {
        color: #6b7280;
        font-size: 0.95rem;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .status-ok {
        background: #d1fae5;
        color: #065f46;
    }
    .status-err {
        background: #fee2e2;
        color: #991b1b;
    }
    .source-item {
        background: #f3f4f6;
        border-radius: 8px;
        padding: 8px 12px;
        margin: 4px 0;
        font-size: 0.85rem;
        border-left: 3px solid #667eea;
    }
    .stApp {
        background-color: #fafafa;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    .chat-msg {
        border-radius: 12px;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        text-align: center;
    }
    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 0.75rem;
        padding-top: 2rem;
    }
    /* 深色模式适配 */
    @media (prefers-color-scheme: dark) {
        .source-item {
            background: #1f2937;
            color: #e5e7eb;
        }
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Session State 初始化
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "backend_url" not in st.session_state:
    st.session_state.backend_url = "http://localhost:8000"
if "health_ok" not in st.session_state:
    st.session_state.health_ok = None
if "session_id" not in st.session_state:
    st.session_state.session_id = ""  # 空 = 后端自动生成，后端返回后保存
if "memory_stats" not in st.session_state:
    st.session_state.memory_stats = {}
if "current_session" not in st.session_state:
    st.session_state.current_session = "新建"


# ============================================================
# 辅助函数
# ============================================================
def check_health(url: str) -> dict | None:
    try:
        resp = requests.get(f"{url}/api/health", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def send_query(url: str, query: str, file_paths: list[str], max_iter: int, session_id: str = ""):
    """调用后端 API 并逐步更新状态"""
    payload = {
        "query": query,
        "file_paths": file_paths,
        "max_iterations": max_iter,
        "session_id": session_id,
    }
    try:
        with requests.post(
            f"{url}/api/ask",
            json=payload,
            timeout=300,  # 长超时，因为 LLM 生成可能较慢
            stream=False,
        ) as resp:
            resp.raise_for_status()
            return resp.json()
    except requests.exceptions.Timeout:
        return {"success": False, "error": "请求超时，请检查后端服务或减少检索迭代次数"}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": f"无法连接到后端 {url}，请确保服务已启动"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def format_sources(sources: list[str]) -> str:
    """将来源列表格式化为易读文本"""
    if not sources:
        return "暂无来源信息"
    return "\n".join(f"📄 {s}" for s in sources)


# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ 配置")

    # 后端地址
    backend_url = st.text_input(
        "后端地址",
        value=st.session_state.backend_url,
        placeholder="http://localhost:8000",
        help="FastAPI 后端服务的地址",
    )
    st.session_state.backend_url = backend_url.rstrip("/")

    # 健康检查
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("#### 服务状态")
    with col2:
        if st.button("刷新", use_container_width=True):
            st.session_state.health_ok = None

    health_data = check_health(st.session_state.backend_url)
    if health_data:
        st.session_state.health_ok = True
        st.markdown(
            f'<span class="status-badge status-ok">● 运行中</span>',
            unsafe_allow_html=True,
        )
        if not health_data.get("api_key_configured"):
            st.warning("⚠️ API Key 未配置", icon="⚠️")
    else:
        st.session_state.health_ok = False
        st.markdown(
            f'<span class="status-badge status-err">● 无法连接</span>',
            unsafe_allow_html=True,
        )
        st.error("后端服务未启动或被关闭", icon="🔴")

    st.divider()

    # 查询参数
    st.markdown("#### 🔍 检索参数")
    max_iterations = st.slider(
        "最大检索迭代次数",
        min_value=1,
        max_value=5,
        value=2,
        help="迭代次数越多，检索越深入，但耗时也更长",
    )

    st.divider()

    # 文件路径输入
    st.markdown("#### 📁 文件路径")
    st.caption("输入需要检索的文件路径（可选），每行一个")
    file_paths_text = st.text_area(
        "文件路径列表",
        placeholder="""例如：
C:/docs/论文.pdf
./data/docs/报告.docx""",
        height=100,
        label_visibility="collapsed",
    )
    file_paths = [p.strip() for p in file_paths_text.split("\n") if p.strip()]

    if file_paths:
        st.info(f"已添加 {len(file_paths)} 个文件", icon="📎")

    st.divider()

    # 会话信息
    st.markdown("#### 💬 会话")
    if st.session_state.get("session_id"):
        sid = st.session_state.session_id
        st.caption(f"ID: `{sid[:20]}...`")
        mem = st.session_state.get("memory_stats", {})
        if mem:
            st.caption(f"历史: {mem.get('history_len', 0)} 轮 | 记忆: {mem.get('facts_recalled', 0)} 条")
        if st.button("🔄 新建会话", use_container_width=True, type="secondary"):
            st.session_state.session_id = ""
            st.session_state.messages = []
            st.session_state.memory_stats = {}
            st.rerun()
    else:
        st.caption("新会话（首次提问后自动创建）")

    st.divider()

    # 导航
    st.markdown("#### 🧭 导航")
    st.page_link("app.py", label="💬 对话", icon="💬")
    if st.button("🔍 溯源问答", use_container_width=True, help="查看逐跳检索轨迹"):
        st.switch_page("pages/trace.py")
    if st.button("⚙️ 工作台", use_container_width=True, help="知识构建 / 反馈 / Wiki"):
        st.switch_page("pages/workbench.py")

    st.divider()

    # 关于
    st.markdown("#### ℹ️ 关于")
    st.caption(
        "**Agentic RAG** — 多格式智能检索系统\n\n"
        "支持 PDF、Word、Excel、图片、\n"
        "文本及代码文件的自动解析与\n"
        "AI 驱动的知识检索。"
    )


# ============================================================
# 主界面
# ============================================================
st.markdown(
    '<div class="main-header">'
    "<h1>🧠 Agentic RAG</h1>"
    "<p>多格式智能检索系统 — 上传文档，智能问答</p>"
    "</div>",
    unsafe_allow_html=True,
)

# --- 显示对话历史 ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=msg.get("avatar")):
        st.markdown(msg["content"])
        # 如果是助手回复且有来源，显示来源
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📚 来源参考（{len(msg['sources'])} 项）"):
                for s in msg["sources"]:
                    st.markdown(f'<div class="source-item">📄 {s}</div>', unsafe_allow_html=True)
        # 显示元数据
        if msg["role"] == "assistant" and msg.get("metadata"):
            cols = st.columns(3)
            meta = msg["metadata"]
            with cols[0]:
                st.metric("迭代次数", meta.get("iterations", "-"))
            with cols[1]:
                st.metric("检索文档块", meta.get("chunk_count", "-"))
            with cols[2]:
                st.metric("来源数", len(msg.get("sources", [])))

# --- 底部输入 ---
if prompt := st.chat_input("请输入你的问题..."):
    # 检查后端是否可用
    if not st.session_state.health_ok:
        st.error("❌ 后端服务不可用，请先在侧边栏确认服务状态", icon="🔴")
        st.stop()

    # 添加用户消息
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "avatar": "🧑‍💻",
    })
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # 调用后端 API
    with st.chat_message("assistant", avatar="🤖"):
        msg_placeholder = st.empty()
        status_placeholder = st.empty()

        with status_placeholder.status("🤔 正在思考...", expanded=True) as status:
            st.write("📡 连接后端服务...")
            st.write(f"📝 问题: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
            if file_paths:
                st.write(f"📁 关联文件: {len(file_paths)} 个")

            result = send_query(
                url=st.session_state.backend_url,
                query=prompt,
                file_paths=file_paths,
                max_iter=max_iterations,
                session_id=st.session_state.session_id,
            )

        status_placeholder.empty()

        if result.get("success"):
            answer = result["answer"]
            sources = result.get("sources", [])
            metadata = {
                "iterations": result.get("iterations", 0),
                "chunk_count": result.get("chunk_count", 0),
                "session_id": result.get("session_id", ""),
                "memory_stats": result.get("memory_stats", {}),
                "elapsed_ms": result.get("elapsed_ms", 0),
            }

            # 保存 session_id 以实现多轮对话记忆
            if result.get("session_id"):
                st.session_state.session_id = result["session_id"]
                st.session_state.memory_stats = result.get("memory_stats", {})
                # 在侧边栏显示 session_id
                st.session_state.current_session = result["session_id"][:20] + "..."

            # 显示答案
            msg_placeholder.markdown(answer)

            # 显示来源
            if sources:
                with st.expander(f"📚 来源参考（{len(sources)} 项）"):
                    for s in sources:
                        st.markdown(f'<div class="source-item">📄 {s}</div>', unsafe_allow_html=True)

            # 显示元数据
            cols = st.columns(4)
            with cols[0]:
                st.metric("🔄 迭代次数", metadata["iterations"])
            with cols[1]:
                st.metric("📦 检索文档块", metadata["chunk_count"])
            with cols[2]:
                st.metric("📚 来源数", len(sources))
            with cols[3]:
                elapsed = metadata.get("elapsed_ms", 0)
                st.metric("⏱ 耗时", f"{elapsed/1000:.1f}s")

            # 保存到历史
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "metadata": metadata,
                "avatar": "🤖",
            })
        else:
            error_msg = result.get("error", "未知错误")
            msg_placeholder.error(f"❌ {error_msg}")

            # 即使失败也保存 session_id，让重试可以恢复已完成的检索
            if result.get("session_id"):
                st.session_state.session_id = result["session_id"]
                st.session_state.current_session = result["session_id"][:20] + "..."

            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ 出错了: {error_msg}",
                "sources": [],
                "metadata": {},
                "avatar": "🤖",
            })

# --- 空状态引导 ---
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align: center; padding: 4rem 1rem; color: #9ca3af;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🧠</div>
        <h3 style="color: #6b7280; font-weight: 500;">开始提问吧！</h3>
        <p style="max-width: 480px; margin: 0.5rem auto;">
            在下方输入问题，系统会自动检索已索引的文档内容，
            并基于 AI 生成智能回答。
        </p>
        <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 2rem; flex-wrap: wrap;">
            <div style="background: white; border-radius: 12px; padding: 1.2rem 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.06); min-width: 140px;">
                <div style="font-size: 1.8rem;">📄</div>
                <div style="font-weight: 500; color: #374151;">PDF / Word</div>
            </div>
            <div style="background: white; border-radius: 12px; padding: 1.2rem 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.06); min-width: 140px;">
                <div style="font-size: 1.8rem;">📊</div>
                <div style="font-weight: 500; color: #374151;">Excel</div>
            </div>
            <div style="background: white; border-radius: 12px; padding: 1.2rem 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.06); min-width: 140px;">
                <div style="font-size: 1.8rem;">🖼️</div>
                <div style="font-weight: 500; color: #374151;">图片 OCR</div>
            </div>
            <div style="background: white; border-radius: 12px; padding: 1.2rem 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.06); min-width: 140px;">
                <div style="font-size: 1.8rem;">💻</div>
                <div style="font-weight: 500; color: #374151;">代码 / 文本</div>
            </div>
        </div>
        <p style="margin-top: 2rem; font-size: 0.85rem;">
            💡 在侧边栏配置后端地址和文件路径后即可开始
        </p>
    </div>
    """, unsafe_allow_html=True)

# --- 底部留白 ---
st.markdown(
    '<div class="footer">Agentic RAG · 基于 LangGraph + FastAPI + Streamlit</div>',
    unsafe_allow_html=True,
)
