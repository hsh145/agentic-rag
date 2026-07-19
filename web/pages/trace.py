"""
Agentic RAG — 溯源问答 & 分块可视化观测页
"""
import time
import requests
import streamlit as st
import pandas as pd
from pathlib import Path
from collections import Counter

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="溯源问答 · Agentic RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 样式
# ============================================================
st.markdown("""
<style>
    .trace-header h1 {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #0ea5e9 0%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .trace-header p { color: #6b7280; font-size: 0.95rem; }
    /* 隐藏 Streamlit 自带 deploy 按钮和顶部白条 */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    .stAppToolbar {display: none;}
    .stAppDeployButton {display: none;}
    .block-container {padding-top: 1rem !important;}
    .timeline-node {
        border-left: 3px solid #0ea5e9;
        padding: 0.6rem 1rem;
        margin: 0.5rem 0;
        background: #f8fafc;
        border-radius: 0 8px 8px 0;
    }
    .timeline-node.hop { border-left-color: #f59e0b; }
    .timeline-node.intent { border-left-color: #8b5cf6; }
    .timeline-node.search { border-left-color: #0ea5e9; }
    .timeline-node.eval { border-left-color: #10b981; }
    .timeline-node.reflect { border-left-color: #f97316; }
    .timeline-node.gen { border-left-color: #ec4899; }
    .chunk-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        margin: 0.3rem 0;
        font-size: 0.85rem;
    }
    .chunk-card .score-bar {
        height: 4px;
        border-radius: 2px;
        background: #e5e7eb;
        margin-top: 4px;
    }
    .chunk-card .score-fill {
        height: 100%;
        border-radius: 2px;
        background: linear-gradient(90deg, #0ea5e9, #06b6d4);
    }
    .gap-badge {
        display: inline-block;
        background: #fef3c7;
        color: #92400e;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        margin: 2px;
    }
    .source-tag {
        display: inline-block;
        background: #e0f2fe;
        color: #0369a1;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        margin: 2px;
    }
    .status-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .status-dot.green { background: #10b981; }
    .status-dot.yellow { background: #f59e0b; }
    .status-dot.red { background: #ef4444; }
    .stApp { background-color: #fafafa; }
    .block-container { padding-top: 1rem; }
    @media (prefers-color-scheme: dark) {
        .timeline-node { background: #1f2937; }
        .chunk-card { background: #1f2937; border-color: #374151; }
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Session State
# ============================================================
if "backend_url" not in st.session_state:
    st.session_state.backend_url = "http://localhost:8000"
if "trace_result" not in st.session_state:
    st.session_state.trace_result = None
if "trace_query" not in st.session_state:
    st.session_state.trace_query = ""
if "trace_loading" not in st.session_state:
    st.session_state.trace_loading = False
if "trace_file_paths" not in st.session_state:
    st.session_state.trace_file_paths = []


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


def send_trace_query(url: str, query: str, file_paths: list[str], max_iter: int):
    """调用溯源 API"""
    payload = {
        "query": query,
        "file_paths": file_paths,
        "max_iterations": max_iter,
        "session_id": "",
    }
    try:
        with requests.post(
            f"{url}/api/ask/trace",
            json=payload,
            timeout=300,
            stream=False,
        ) as resp:
            resp.raise_for_status()
            return resp.json()
    except requests.exceptions.Timeout:
        return {"success": False, "error": "请求超时"}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": f"无法连接到后端 {url}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def render_timeline(trace: list):
    """渲染 agent 决策时间线"""
    st.markdown("#### 🕐 Agent 决策时间线")

    for i, entry in enumerate(trace):
        t = entry.get("type", "")
        label = entry.get("stage_label", t)
        css_class = {
            "parse_intent": "intent",
            "plan_retrieval": "intent",
            "execute_search": "search",
            "evaluate_evidence": "eval",
            "reflect_search": "reflect",
            "generate_answer": "gen",
        }.get(t, "hop")

        hop = entry.get("hop", 0)

        if t == "parse_intent":
            with st.container():
                st.markdown(
                    f'<div class="timeline-node {css_class}">'
                    f'<strong>{label}</strong> '
                    f'<span style="color:#6b7280;font-size:0.85rem;">#{hop}</span><br/>'
                    f'<span style="font-size:0.9rem;">{entry.get("analysis", "")}</span><br/>'
                    f'<span style="font-size:0.8rem;color:#6b7280;">'
                    f'类型: {entry.get("query_type", "")} | '
                    f'需解析文件: {entry.get("need_file_parse", False)}'
                    f'</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        elif t == "plan_retrieval":
            sq = entry.get("sub_queries", [])
            with st.container():
                sub_qs = " | ".join(f"`{q}`" for q in sq)
                st.markdown(
                    f'<div class="timeline-node {css_class}">'
                    f'<strong>{label}</strong> '
                    f'<span style="color:#6b7280;font-size:0.85rem;">#{hop}</span><br/>'
                    f'<span style="font-size:0.85rem;">拆分子查询: {sub_qs}</span><br/>'
                    f'<span style="font-size:0.8rem;color:#6b7280;">{entry.get("reasoning", "")}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        elif t == "execute_search":
            top = entry.get("top_chunks", [])
            score_str = ", ".join(f"{c['score']:.3f}" for c in top[:5]) if top else "无"
            with st.container():
                st.markdown(
                    f'<div class="timeline-node {css_class}">'
                    f'<strong>{label}</strong> '
                    f'<span style="color:#6b7280;font-size:0.85rem;">#{hop}</span><br/>'
                    f'<span style="font-size:0.9rem;">'
                    f'查询: {" | ".join(entry.get("queries", []))}</span><br/>'
                    f'<span style="font-size:0.8rem;color:#6b7280;">'
                    f'检索到 {entry.get("chunk_count", 0)} 个块 | '
                    f'Top-5 分数: {score_str}'
                    f'</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        elif t == "evaluate_evidence":
            can_answer = entry.get("can_answer", False)
            conf = entry.get("confidence", 0)
            gaps = entry.get("missing_gaps", [])
            dot_class = "green" if can_answer else ("yellow" if conf > 0.3 else "red")
            verdict = "✅ 信息充分" if can_answer else "❌ 信息不足"
            with st.container():
                gaps_html = ""
                if gaps:
                    gaps_html = "<br/>缺口: " + " ".join(
                        f'<span class="gap-badge">{g}</span>' for g in gaps
                    )
                st.markdown(
                    f'<div class="timeline-node {css_class}">'
                    f'<strong>{label}</strong> '
                    f'<span style="color:#6b7280;font-size:0.85rem;">#{hop}</span><br/>'
                    f'<span class="status-dot {dot_class}"></span>'
                    f'{verdict} (置信度: {conf:.0%})<br/>'
                    f'<span style="font-size:0.8rem;color:#6b7280;">'
                    f'块数: {entry.get("chunk_count", 0)} | '
                    f'总字符: {entry.get("total_chars", 0)}'
                    f'</span>'
                    f'{gaps_html}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        elif t == "reflect_search":
            gq = entry.get("generated_queries", [])
            gaps = entry.get("missing_gaps", [])
            with st.container():
                gaps_html = ""
                if gaps:
                    gaps_html = "<br/>缺口: " + " ".join(
                        f'<span class="gap-badge">{g}</span>' for g in gaps
                    )
                q_html = "<br/>补搜查询: " + " | ".join(f"`{q}`" for q in gq) if gq else "无新查询"
                st.markdown(
                    f'<div class="timeline-node {css_class}">'
                    f'<strong>{label}</strong> '
                    f'<span style="color:#6b7280;font-size:0.85rem;">#{hop}</span>'
                    f'{gaps_html}'
                    f'{q_html}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        elif t == "generate_answer":
            sources = entry.get("sources", [])
            src_html = " ".join(
                f'<span class="source-tag">{s}</span>' for s in sources[:10]
            ) if sources else "无来源"
            with st.container():
                st.markdown(
                    f'<div class="timeline-node {css_class}">'
                    f'<strong>{label}</strong> '
                    f'<span style="color:#6b7280;font-size:0.85rem;">#{hop}</span><br/>'
                    f'<span style="font-size:0.85rem;">'
                    f'使用了 {entry.get("chunks_used", 0)} 个块</span><br/>'
                    f'{src_html}'
                    f'</div>',
                    unsafe_allow_html=True,
                )


def render_chunk_analysis(chunks: list):
    """渲染分块得分分布和来源分布"""
    if not chunks:
        st.info("无检索结果")
        return

    import altair as alt

    # 分块得分分布（左）
    scores = [c.get("score", 0) for c in chunks]
    if scores:
        st.markdown("#### 📊 分块得分分布")
        score_df = {"块序号": [str(i+1) for i in range(len(scores))], "RRF 分数": scores}
        score_chart = alt.Chart(pd.DataFrame(score_df)).mark_bar().encode(
            x=alt.X("块序号:N", axis=alt.Axis(labelAngle=0, labelLimit=60), sort=None),
            y=alt.Y("RRF 分数:Q"),
        ).properties(height=250, width=400)
        st.altair_chart(score_chart, use_container_width=True)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            hi = sum(1 for s in scores if s > 0.15)
            st.metric("高置信度", hi)
        with col_b:
            mid = sum(1 for s in scores if 0.05 <= s <= 0.15)
            st.metric("中等置信度", mid)
        with col_c:
            lo = sum(1 for s in scores if s < 0.05)
            st.metric("低置信度", lo)

    st.divider()

    # 来源文件分布（右）
    sources = [Path(c.get("source", "unknown")).name for c in chunks]
    if sources:
        st.markdown("#### 📁 来源文件分布")
        counter = Counter(sources)
        src_df = {"文件名": list(counter.keys()), "块数": list(counter.values())}
        src_chart = alt.Chart(pd.DataFrame(src_df)).mark_bar().encode(
            x=alt.X("文件名:N", axis=alt.Axis(labelAngle=0, labelLimit=120), sort=None),
            y=alt.Y("块数:Q"),
        ).properties(height=250, width=400)
        st.altair_chart(src_chart, use_container_width=True)

        st.markdown("**来源详情:**")
        for src, cnt in counter.most_common():
            pct = cnt / len(chunks) * 100
            st.markdown(
                f'<span class="source-tag">{src}</span> '
                f'{cnt} 块 ({pct:.0f}%)',
                unsafe_allow_html=True,
            )


def render_chunk_detail(chunks: list):
    """展开显示每个 chunk 的详情"""
    if not chunks:
        return

    with st.expander(f"📄 全部检索结果详情 ({len(chunks)} 个块)", expanded=True):
        for i, c in enumerate(chunks):
            score = c.get("score", 0)
            source = Path(c.get("source", "unknown")).name
            snippet = c.get("content_snippet", "")[:400]
            pct = min(score * 100 / 0.3, 100)  # max at 0.3 → 100%

            st.markdown(
                f'<div class="chunk-card">'
                f'<strong>#{i + 1}</strong> '
                f'<span class="source-tag">{source}</span> '
                f'<span style="color:#6b7280;font-size:0.8rem;">分数: {score:.4f}</span>'
                f'<div class="score-bar">'
                f'<div class="score-fill" style="width:{pct:.0f}%"></div>'
                f'</div>'
                f'<div style="margin-top:4px;font-size:0.85rem;color:#374151;">{snippet}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def render_hop_details(trace: list, chunks: list):
    """逐跳展开详细数据"""
    st.markdown("#### 🔬 逐跳详细数据")
    for i, entry in enumerate(trace):
        t = entry.get("type", "")
        label = entry.get("stage_label", t)
        hop = entry.get("hop", 0)

        with st.expander(f"#{hop} {label}", expanded=False):
            st.json(entry)


# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ 配置")
    backend_url = st.text_input(
        "后端地址",
        value=st.session_state.backend_url,
        placeholder="http://localhost:8000",
    )
    st.session_state.backend_url = backend_url.rstrip("/")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("#### 服务状态")
    with col2:
        if st.button("刷新", use_container_width=True):
            st.rerun()

    health_data = check_health(st.session_state.backend_url)
    if health_data:
        st.markdown(
            f'<span class="status-badge" style="background:#d1fae5;color:#065f46;'
            f'padding:4px 12px;border-radius:20px;font-size:0.8rem;">● 运行中</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<span class="status-badge" style="background:#fee2e2;color:#991b1b;'
            f'padding:4px 12px;border-radius:20px;font-size:0.8rem;">● 无法连接</span>',
            unsafe_allow_html=True,
        )
        st.error("请确认后端已启动")

    st.divider()

    st.markdown("#### 🔍 检索参数")
    max_iterations = st.slider(
        "最大迭代次数", min_value=1, max_value=5, value=2,
        help="迭代次数越多，追溯越详细",
    )

    st.divider()
    st.markdown("#### 📁 文件路径（可选）")
    file_paths_text = st.text_area(
        "文件路径", placeholder="每行一个路径", height=80,
        label_visibility="collapsed",
    )
    file_paths = [p.strip() for p in file_paths_text.split("\n") if p.strip()]
    if file_paths:
        st.info(f"{len(file_paths)} 个文件")

    st.divider()
    st.markdown("#### ℹ️ 溯源问答")
    st.caption(
        "此页面显示 Agentic RAG 的完整检索与决策过程。\n\n"
        "**时间线** → 每步决策节点\n"
        "**分块分析** → 检索结果的得分与来源分布\n"
        "**逐跳细节** → 每条 trace 的原始数据"
    )

    if st.button("← 返回主对话", use_container_width=True):
        try:
            st.switch_page("app.py")
        except Exception:
            st.markdown('[打开主对话 →](/)', unsafe_allow_html=True)


# ============================================================
# 主界面
# ============================================================
st.markdown(
    '<div class="trace-header">'
    "<h1>🔍 溯源问答 & 分块可视化</h1>"
    "<p>观测 Agent 的每一步检索、评估与决策过程，追踪答案的来源</p>"
    "</div>",
    unsafe_allow_html=True,
)

st.divider()

# 输入区
query = st.text_input(
    "请输入问题",
    placeholder="例如：这篇文章主要讲了什么？",
    label_visibility="visible",
    disabled=st.session_state.trace_loading,
)

col_q, col_s = st.columns([1, 5])
with col_q:
    trace_btn = st.button(
        "🔍 溯源检索",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.trace_loading or not query,
    )
with col_s:
    if st.session_state.trace_loading:
        st.info("⏳ 正在溯源检索中（LLM 生成可能需要 30-60 秒）...")

# 执行溯源
if trace_btn and query:
    st.session_state.trace_loading = True
    st.session_state.trace_query = query
    st.session_state.trace_file_paths = file_paths

    with st.spinner("Agent 正在多步检索与推理..."):
        result = send_trace_query(
            url=st.session_state.backend_url,
            query=query,
            file_paths=file_paths,
            max_iter=max_iterations,
        )
        st.session_state.trace_result = result

    st.session_state.trace_loading = False
    st.rerun()

# 显示结果
trace_result = st.session_state.trace_result
trace_query = st.session_state.trace_query

if trace_result is not None:
    if not trace_result.get("success"):
        st.error(f"❌ 溯源失败: {trace_result.get('error', '未知错误')}")
    else:
        answer = trace_result.get("answer", "")
        trace = trace_result.get("agentic_trace", [])
        chunks = trace_result.get("retrieved_chunks", [])
        sources = trace_result.get("sources", [])
        iterations = trace_result.get("iterations", 0)
        elapsed = trace_result.get("elapsed_ms", 0)
        ev_scores = trace_result.get("evidence_scores", {})
        ev_feedback = trace_result.get("evidence_feedback", "")
        missing_gaps = trace_result.get("missing_gaps", [])
        retrieval_plan = trace_result.get("retrieval_plan", [])
        supp_queries = trace_result.get("supplementary_queries", [])

        # === 顶部元信息 ===
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("迭代轮次", iterations)
        with col_m2:
            st.metric("检索块数", len(chunks))
        with col_m3:
            st.metric("来源文件", len(sources))
        with col_m4:
            st.metric("耗时", f"{elapsed / 1000:.1f}s")

        # === 问题和子查询 ===
        with st.expander("📋 检索计划与子查询", expanded=False):
            st.markdown(f"**原始问题:** {trace_query}")
            if retrieval_plan:
                st.markdown("**子查询:**")
                for q in retrieval_plan:
                    st.markdown(f"- `{q}`")
            if supp_queries:
                st.markdown("**补搜查询:**")
                for q in supp_queries:
                    st.markdown(f"- `{q}` (补搜)")

        # === 答案 ===
        st.markdown("#### 💡 回答")
        answer_container = st.container()
        with answer_container:
            st.markdown(answer)

        if sources:
            with st.expander(f"📚 来源参考（{len(sources)} 项）"):
                for s in sources:
                    st.markdown(f'<div class="source-item" style="padding:4px 8px;margin:2px 0;background:#f3f4f6;border-radius:4px;border-left:3px solid #0ea5e9;">📄 {s}</div>', unsafe_allow_html=True)

        st.divider()

        # === 证据评估反馈 ===
        if ev_feedback:
            with st.expander("📋 证据评估摘要", expanded=False):
                st.markdown(f"**评估:** {ev_feedback}")
                st.markdown(f"**分数:** {ev_scores}")
                if missing_gaps:
                    st.markdown("**信息缺口:**")
                    for g in missing_gaps:
                        st.markdown(f'- <span class="gap-badge">{g}</span>', unsafe_allow_html=True)

        st.divider()

        # === 时间线 (必须展开) ===
        if trace:
            render_timeline(trace)
            st.divider()

        # === 分块分析 ===
        render_chunk_analysis(chunks)
        st.divider()

        # === 全部 chunk 详情 ===
        render_chunk_detail(chunks)
        st.divider()

        # === 逐跳原始数据 ===
        render_hop_details(trace, chunks)
        st.divider()

        # === 导出 ===
        with st.expander("💾 导出原始数据", expanded=False):
            import json
            export_data = {
                "query": trace_query,
                "answer": answer,
                "agentic_trace": trace,
                "retrieved_chunks": chunks,
                "sources": sources,
                "evidence_scores": ev_scores,
                "evidence_feedback": ev_feedback,
                "missing_gaps": missing_gaps,
                "retrieval_plan": retrieval_plan,
                "supplementary_queries": supp_queries,
                "iterations": iterations,
                "elapsed_ms": elapsed,
            }
            st.download_button(
                "📥 下载 JSON",
                data=json.dumps(export_data, ensure_ascii=False, indent=2),
                file_name=f"trace_{int(time.time())}.json",
                mime="application/json",
                use_container_width=True,
            )

# === 空状态 ===
else:
    st.info(
        "在上方输入问题并点击「溯源检索」按钮，"
        "查看 Agentic RAG 的完整检索轨迹与分块可视化分析。"
    )
    st.markdown("""
    <div style="text-align:center;padding:3rem;color:#9ca3af;">
        <div style="font-size:3rem;margin-bottom:1rem;">🔍</div>
        <p style="font-size:1rem;">
            观测内容包括：Agent 决策时间线 · 分块得分分布 · 来源文件归属 · 逐跳检索与评估轨迹
        </p>
    </div>
    """, unsafe_allow_html=True)

# 底部
st.markdown(
    '<div style="text-align:center;color:#9ca3af;font-size:0.75rem;padding-top:2rem;">'
    '溯源问答 · Agentic RAG | 数据来自 POST /api/ask/trace</div>',
    unsafe_allow_html=True,
)
