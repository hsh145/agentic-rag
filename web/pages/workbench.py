"""
Agent 工作台 — 产品闭环全流程

模块：
  🔧 Knowledge Builder   发现→解析→分块→索引 可视化
  💬 QA Workbench        检索问答 + trace + 引用 + abstain
  📊 Feedback            收集好评/差评/错误类型
  📝 Deliverables        导出报告/摘要/引用清单
  🧠 Knowledge Wiki      浏览长期记忆中的知识事实
"""
import time
import json
import requests
import streamlit as st
from pathlib import Path
from datetime import datetime

st.set_page_config(
    page_title="Agent 工作台 · Agentic RAG",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 样式
# ============================================================
st.markdown("""
<style>
    /* 隐藏 Streamlit 自带 deploy 按钮和顶部白条 */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    .stAppToolbar {display: none;}
    .stAppDeployButton {display: none;}
    .block-container {padding-top: 1rem !important;}
    .wb-header h1 {
        font-size: 1.8rem; font-weight: 700;
        background: linear-gradient(135deg, #0ea5e9, #8b5cf6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .pipeline-stage {
        border: 1px solid #e5e7eb; border-radius: 10px;
        padding: 1rem; margin: 0.5rem 0;
        background: white;
        transition: all 0.2s;
    }
    .pipeline-stage.active { border-color: #0ea5e9; box-shadow: 0 0 0 2px rgba(14,165,233,0.2); }
    .pipeline-stage.done { border-color: #10b981; background: #f0fdf4; }
    .pipeline-stage.error { border-color: #ef4444; background: #fef2f2; }
    .state-badge {
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        font-size: 0.75rem; font-weight: 600;
    }
    .state-badge.loading { background: #dbeafe; color: #1d4ed8; }
    .state-badge.done { background: #d1fae5; color: #065f46; }
    .state-badge.error { background: #fee2e2; color: #991b1b; }
    .state-badge.pending { background: #f3f4f6; color: #6b7280; }
    .feedback-btn { font-size: 1.5rem; cursor: pointer; padding: 0.3rem 0.6rem; border-radius: 8px; }
    .feedback-btn:hover { background: #f3f4f6; }
    .fact-card {
        background: white; border: 1px solid #e5e7eb; border-radius: 8px;
        padding: 0.6rem 0.8rem; margin: 0.4rem 0; border-left: 3px solid #8b5cf6;
    }
    .deliverable-card {
        background: white; border: 1px solid #e5e7eb; border-radius: 8px;
        padding: 1rem; margin: 0.5rem 0;
    }
    @media (prefers-color-scheme: dark) {
        .pipeline-stage { background: #1f2937; border-color: #374151; }
        .pipeline-stage.done { background: #064e3b; }
        .pipeline-stage.error { background: #7f1d1d; }
        .fact-card { background: #1f2937; border-color: #374151; }
        .deliverable-card { background: #1f2937; }
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="wb-header"><h1>⚙️ Agent 工作台</h1></div>', unsafe_allow_html=True)
st.caption("知识构建 → 检索问答 → 反馈收集 → 交付物生成 → 知识沉淀 — 完整产品闭环")

BACKEND_URL = st.session_state.get("backend_url", "http://localhost:8000")

# ============================================================
# Session State
# ============================================================
for key in ["wb_qa_result", "wb_qa_loading", "wb_last_answer", "wb_last_question",
            "wb_last_sources", "wb_last_chunks", "wb_last_trace"]:
    if key not in st.session_state:
        st.session_state[key] = None if "result" in key or "sources" in key or "chunks" in key or "trace" in key else False

# ============================================================
# 辅助函数
# ============================================================

def api_post(path: str, payload: dict, timeout=300):
    try:
        r = requests.post(f"{BACKEND_URL}{path}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        return {"success": False, "error": "请求超时"}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": f"无法连接 {BACKEND_URL}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def api_get(path: str, params=None, timeout=10):
    try:
        r = requests.get(f"{BACKEND_URL}{path}", params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def health_check() -> dict | None:
    data = api_get("/api/health", timeout=5)
    return data if data.get("status") == "ok" else None

# ============================================================
# Tab 1: Knowledge Builder
# ============================================================
def render_knowledge_builder():
    st.markdown("### 🔧 知识构建管道")
    st.caption("展示从原始文档到可检索索引的完整构建流程")

    health = health_check()
    index_ok = health and health.get("checks", {}).get("index", {}).get("loaded", False)
    vector_count = health.get("checks", {}).get("index", {}).get("total_vectors", 0) if health else 0

    # 管道状态
    stages = [
        {"name": "📄 文档解析", "key": "parse", "status": "done" if health else "pending",
         "desc": "PDF / Word / Excel / 图片 → 结构化文本"},
        {"name": "✂️ 语义分块", "key": "chunk", "status": "done" if health else "pending",
         "desc": "按语义边界切割为独立段落"},
        {"name": "🧮 向量化", "key": "embed", "status": "done" if vector_count > 0 else "pending",
         "desc": f"DashScope text-embedding-v2 → {vector_count} 个向量"},
        {"name": "📇 索引构建", "key": "index", "status": "done" if index_ok else "pending",
         "desc": "FAISS (IndexFlatIP) + BM25 混合索引"},
    ]

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("**管道状态**")
        for s in stages:
            cls = s["status"]
            icon = {"done": "✅", "active": "🔄", "pending": "⏳", "error": "❌"}.get(cls, "⏳")
            st.markdown(
                f'<div class="pipeline-stage {cls}">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<span><strong>{icon} {s["name"]}</strong></span>'
                f'<span class="state-badge {cls}">{cls}</span>'
                f'</div>'
                f'<div style="font-size:0.85rem;color:#6b7280;margin-top:4px;">{s["desc"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        if not health:
            st.warning("⏳ 后端未连接，状态未知", icon="🔌")
        elif not index_ok:
            st.info("💡 索引为空 — 可上传文档构建", icon="📂")

    with col2:
        st.markdown("**上传文档**")
        uploaded = st.file_uploader(
            "选择文件", type=["pdf", "docx", "xlsx", "txt", "md", "png", "jpg"],
            label_visibility="collapsed",
        )
        if uploaded:
            st.success(f"已选择: {uploaded.name} ({uploaded.size/1024:.1f} KB)", icon="📎")
            if st.button("🚀 开始构建", use_container_width=True, type="primary"):
                st.info("构建管道执行中（模拟）...")
                progress = st.progress(0)
                for pct, label in [(20, "解析中..."), (45, "分块中..."), (70, "向量化中..."), (95, "索引中...")]:
                    time.sleep(0.5)
                    progress.progress(pct, text=label)
                progress.progress(100, text="✅ 完成")
                st.success("文档已加入索引，可前往 QA 工作台提问")
                st.rerun()

    # 系统状态
    st.divider()
    col_a, col_b, col_c = st.columns(3)
    if health:
        mem = health.get("memory", {})
        idx = health.get("checks", {}).get("index", {})
        col_a.metric("📦 向量总数", idx.get("total_vectors", "?"))
        col_b.metric("💬 会话数", mem.get("session_count", "?"))
        col_c.metric("🧠 长期记忆", mem.get("fact_count", "?"))
    else:
        col_a.error("服务离线")
        col_b.empty()
        col_c.empty()


# ============================================================
# Tab 2: QA Workbench
# ============================================================
def render_qa_workbench():
    st.markdown("### 💬 检索问答工作台")
    st.caption("基于证据回答，支持 trace 溯源、引用展示、无证据拒答")

    health = health_check()
    if not health:
        st.error("🔌 后端服务未连接", icon="🔴")
        if st.button("重试连接"):
            st.rerun()
        return

    # 输入区
    col_q, col_s = st.columns([4, 1])
    with col_q:
        question = st.text_input(
            "问题", placeholder="输入你的问题，例如：什么是 SFT 微调？",
            label_visibility="collapsed",
            disabled=st.session_state.wb_qa_loading,
        )
    with col_s:
        submit = st.button(
            "🔍 检索", type="primary", use_container_width=True,
            disabled=st.session_state.wb_qa_loading or not question,
        )

    # 执行检索
    if submit and question:
        st.session_state.wb_qa_loading = True
        st.session_state.wb_qa_result = None

        result = api_post("/api/ask/trace", {
            "query": question, "file_paths": [], "max_iterations": 2,
            "session_id": f"wb_{int(time.time())}",
        })

        st.session_state.wb_qa_result = result
        st.session_state.wb_qa_loading = False
        st.rerun()

    # Loading 状态
    if st.session_state.wb_qa_loading:
        st.markdown("""
        <div style="text-align:center;padding:3rem;">
            <div style="font-size:2rem;margin-bottom:1rem;">🔄</div>
            <p>Agent 正在执行多步检索与推理...</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # 结果区
    result = st.session_state.wb_qa_result
    if result is None:
        st.info("💡 输入问题并点击「检索」查看结果", icon="🔍")
        return

    if not result.get("success"):
        st.error(f"❌ {result.get('error', '请求失败')}")
        if st.button("🔄 重试"):
            st.session_state.wb_qa_result = None
            st.rerun()
        return

    # === Success 状态 ===
    answer = result.get("answer", "")
    trace = result.get("agentic_trace", [])
    chunks = result.get("retrieved_chunks", [])
    sources = result.get("sources", [])
    iterations = result.get("iterations", 0)
    elapsed = result.get("elapsed_ms", 0)
    evidence_feedback = result.get("evidence_feedback", "")
    missing_gaps = result.get("missing_gaps", [])

    # 保存到 session（给 deliverables 用）
    st.session_state.wb_last_answer = answer
    st.session_state.wb_last_question = question
    st.session_state.wb_last_sources = sources

    # 判断 abstain 状态
    is_abstained = not answer or "没有相关信息" in answer or "无法回答" in answer or "信息不足" in answer
    no_evidence = len(chunks) == 0 or (chunks and all(c.get("score", 0) == 0 for c in chunks))

    st.divider()

    # 元信息
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        status = "⏸️ 拒答" if is_abstained else "✅ 已回答"
        st.metric("状态", status)
    with col_m2:
        st.metric("迭代轮次", iterations)
    with col_m3:
        st.metric("检索块数", len(chunks))
    with col_m4:
        st.metric("耗时", f"{elapsed/1000:.1f}s")

    if no_evidence and is_abstained:
        st.warning("🛑 **未检索到有效证据，系统已自动拒答** — 这是正确行为", icon="🛡️")

    # 答案区
    st.markdown("#### 📝 回答")
    if is_abstained:
        st.info(answer if answer else "系统未生成回答（无相关证据）")
    else:
        st.success(answer)

    # 引用
    if sources and not is_abstained:
        with st.expander(f"📚 引用来源 ({len(sources)} 项)", expanded=True):
            for s in sources:
                st.markdown(f'<div style="padding:4px 8px;margin:2px 0;background:#f3f4f6;border-radius:4px;border-left:3px solid #0ea5e9;">📄 {s}</div>', unsafe_allow_html=True)

    if evidence_feedback:
        with st.expander("📋 证据评估"):
            st.markdown(f"**评估:** {evidence_feedback}")
            if missing_gaps:
                st.markdown("**信息缺口:**")
                for g in missing_gaps:
                    st.markdown(f"- ⚠️ {g}")

    # Trace 时间线
    if trace:
        with st.expander("🕐 Agent 决策时间线", expanded=False):
            for entry in trace:
                t = entry.get("type", "")
                label = entry.get("stage_label", t)
                hop = entry.get("hop", 0)
                cls = {"parse_intent": "intent", "plan_retrieval": "intent",
                       "execute_search": "search", "evaluate_evidence": "eval",
                       "reflect_search": "reflect", "generate_answer": "gen"}.get(t, "")
                css = f"border-left:3px solid #0ea5e9;padding:0.4rem 0.8rem;margin:0.3rem 0;background:#f8fafc;border-radius:0 6px 6px 0;font-size:0.85rem;"

                if t == "evaluate_evidence":
                    can = entry.get("can_answer", False)
                    icon = "✅" if can else "❌"
                    st.markdown(
                        f'<div style="{css}">'
                        f'<strong>#{hop} {label}</strong> {icon} '
                        f'置信度: {entry.get("confidence", 0):.0%}'
                        f'</div>', unsafe_allow_html=True)
                elif t == "reflect_search":
                    qs = entry.get("generated_queries", [])
                    st.markdown(
                        f'<div style="{css}">'
                        f'<strong>#{hop} {label}</strong> '
                        f'{" | ".join(f"<code>{q}</code>" for q in qs) if qs else "无需补搜"}'
                        f'</div>', unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<div style="{css}">'
                        f'<strong>#{hop} {label}</strong></div>', unsafe_allow_html=True)

    # 检索结果
    if chunks:
        with st.expander(f"📄 检索块详情 ({len(chunks)})", expanded=False):
            for i, c in enumerate(chunks):
                score = c.get("score", 0)
                source = Path(c.get("source", "unknown")).name
                snippet = c.get("content_snippet", "")[:200]
                st.markdown(
                    f'<div style="background:white;border:1px solid #e5e7eb;border-radius:6px;'
                    f'padding:0.5rem 0.7rem;margin:0.3rem 0;font-size:0.85rem;">'
                    f'<strong>#{i+1}</strong> <code>{source}</code> 分数: {score:.4f}<br>{snippet}</div>',
                    unsafe_allow_html=True,
                )

    # 反馈快捷入口
    st.divider()
    st.markdown("#### 📊 评价此回答")
    col_r1, col_r2, col_r3 = st.columns([1, 1, 4])
    with col_r1:
        if st.button("👍 有用", use_container_width=True, key="fb_good"):
            api_post("/api/feedback", {
                "session_id": "workbench", "question": question,
                "answer_snippet": answer[:200], "rating": 1, "source": "qa_workbench",
            })
            st.toast("感谢好评！")
    with col_r2:
        if st.button("👎 无用", use_container_width=True, key="fb_bad"):
            api_post("/api/feedback", {
                "session_id": "workbench", "question": question,
                "answer_snippet": answer[:200], "rating": -1, "source": "qa_workbench",
            })
            st.toast("已记录，我们会改进")
    with col_r3:
        if st.button("🔄 重试（重新检索）", use_container_width=True):
            st.session_state.wb_qa_result = None
            st.rerun()


# ============================================================
# Tab 3: Feedback
# ============================================================
def render_feedback():
    st.markdown("### 📊 反馈中心")
    st.caption("收集用户评价，驱动系统迭代优化")

    col_form, col_history = st.columns([1, 1])

    with col_form:
        st.markdown("**提交反馈**")
        with st.form("feedback_form"):
            fb_question = st.text_input("问题", placeholder="你当时问的是...")
            fb_rating = st.select_slider("评价", options=["差评", "一般", "好评"], value="一般")
            fb_error = st.selectbox(
                "错误类型（可选）",
                ["", "幻觉（编造内容）", "信息缺失", "答案错误", "其他"],
            )
            fb_notes = st.text_area("备注（可选）", placeholder="具体哪里不对？", height=80)
            submitted = st.form_submit_button("📤 提交", use_container_width=True, type="primary")

            if submitted:
                rating_map = {"差评": -1, "一般": 0, "好评": 1}
                err_map = {"幻觉（编造内容）": "hallucination", "信息缺失": "missing_info",
                           "答案错误": "wrong", "其他": "other", "": ""}
                api_post("/api/feedback", {
                    "session_id": "feedback_page",
                    "question": fb_question,
                    "rating": rating_map[fb_rating],
                    "error_type": err_map[fb_error],
                    "notes": fb_notes,
                    "source": "feedback_page",
                })
                st.success("✅ 反馈已提交，感谢！")

    with col_history:
        st.markdown("**历史反馈**")
        data = api_get("/api/feedback", params={"limit": 20})
        if data.get("success") and data.get("data"):
            for fb in data["data"]:
                rating_icon = {1: "👍", 0: "➖", -1: "👎"}.get(fb.get("rating", 0), "➖")
                err = fb.get("error_type", "")
                with st.container():
                    st.markdown(
                        f'<div style="background:white;border:1px solid #e5e7eb;border-radius:8px;'
                        f'padding:0.5rem 0.8rem;margin:0.3rem 0;">'
                        f'<div style="display:flex;justify-content:space-between;">'
                        f'<span>{rating_icon} <strong>{fb.get("question", "")[:40]}</strong></span>'
                        f'<span style="font-size:0.75rem;color:#6b7280;">{fb.get("timestamp","")}</span>'
                        f'</div>'
                        f'{"<span class=state-badge.error>"+err+"</span>" if err else ""}'
                        f'<div style="font-size:0.85rem;color:#6b7280;">{fb.get("notes","")}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.info("💡 尚无反馈数据 — 去 QA 工作台提问后打分", icon="📭")


# ============================================================
# Tab 4: Deliverables
# ============================================================
def render_deliverables():
    st.markdown("### 📝 交付物生成")
    st.caption("将检索问答成果导出为结构化文档")

    # 检查是否有最近的 QA 结果
    has_recent = bool(st.session_state.get("wb_last_answer"))

    if not has_recent:
        st.info("💡 暂无问答记录 — 先去 QA 工作台完成一次检索问答", icon="📭")
        st.page_link("pages/workbench.py", label="→ 前往 QA 工作台", icon="💬")
        return

    question = st.session_state.wb_last_question
    answer = st.session_state.wb_last_answer
    sources = st.session_state.wb_last_sources or []

    st.markdown("**当前问答**")
    st.markdown(f'<div class="deliverable-card"><strong>Q:</strong> {question}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="deliverable-card"><strong>A:</strong> {answer[:300]}{"..." if len(answer) > 300 else ""}</div>', unsafe_allow_html=True)

    st.divider()
    fmt = st.radio("选择格式", ["📄 Markdown 报告", "📊 JSON 结构化", "📋 引用清单"], horizontal=True)

    if fmt == "📄 Markdown 报告":
        content = f"""# 检索问答报告

**问题**: {question}

**回答**: {answer}

## 引用来源
"""
        for s in sources:
            content += f"- {s}\n"
        content += f"\n*生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
        file_name = f"report_{int(time.time())}.md"
        mime = "text/markdown"

    elif fmt == "📊 JSON 结构化":
        content = json.dumps({
            "question": question, "answer": answer, "sources": sources,
            "generated_at": datetime.now().isoformat(),
        }, ensure_ascii=False, indent=2)
        file_name = f"report_{int(time.time())}.json"
        mime = "application/json"

    else:  # 引用清单
        content = "# 引用来源清单\n\n"
        for i, s in enumerate(sources, 1):
            content += f"{i}. {s}\n"
        content += f"\n来源: Agentic RAG 检索系统\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        file_name = f"sources_{int(time.time())}.md"
        mime = "text/markdown"

    st.download_button("📥 下载", data=content, file_name=file_name, mime=mime, use_container_width=True, type="primary")
    st.caption("交付物可复用为知识沉淀或报告附件")


# ============================================================
# Tab 5: Knowledge Wiki
# ============================================================
def render_knowledge_wiki():
    st.markdown("### 🧠 知识 Wiki")
    st.caption("浏览长期记忆中沉淀的结构化事实")

    health = health_check()
    if not health:
        st.error("🔌 后端服务未连接")
        return

    fact_count = health.get("memory", {}).get("fact_count", 0)

    if fact_count == 0:
        st.info("💡 暂无沉淀知识 — 多次问答后系统会自动提取事实", icon="📚")
        st.markdown("""
        <div style="text-align:center;padding:2rem;color:#9ca3af;">
            <div style="font-size:3rem;">🧠</div>
            <p>长期记忆会从每轮问答中提取<br>
            <strong>实体→事实→分类</strong> 的结构化知识</p>
            <p style="font-size:0.85rem;">支持的分类: preference | knowledge | entity | task</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # 事实统计
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("🧠 事实总数", fact_count)
    col_b.metric("💬 会话数", health.get("memory", {}).get("session_count", 0))
    col_c.metric("📦 向量索引", health.get("checks", {}).get("index", {}).get("total_vectors", 0))

    st.divider()
    st.markdown("**最近提取的事实（模拟展示）**")

    # 从 memory API 获取 — 目前 memory 没有获取全部 facts 的公开 API
    # 用模拟数据展示概念
    mock_facts = [
        {"fact": "用户对 SFT 微调技术感兴趣", "entity": "用户", "category": "preference", "confidence": 0.92},
        {"fact": "SFT 是监督式微调的缩写", "entity": "SFT", "category": "knowledge", "confidence": 0.98},
        {"fact": "LoRA 通过低秩矩阵减少参数量", "entity": "LoRA", "category": "knowledge", "confidence": 0.95},
        {"fact": "FAISS 是 Meta 开源的向量检索库", "entity": "FAISS", "category": "knowledge", "confidence": 0.97},
        {"fact": "Agentic RAG 使用 LangGraph 工作流", "entity": "Agentic RAG", "category": "knowledge", "confidence": 0.94},
    ]

    for f in mock_facts:
        cat_badge = {"preference": "💡", "knowledge": "📖", "entity": "🏷️", "task": "✅"}.get(f["category"], "📄")
        st.markdown(
            f'<div class="fact-card">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<span>{cat_badge} <strong>{f["fact"]}</strong></span>'
            f'<span style="font-size:0.75rem;">{f["entity"]} · {f["confidence"]:.0%}</span>'
            f'</div>'
            f'<div style="font-size:0.8rem;color:#6b7280;margin-top:4px;">'
            f'<span class="state-badge done">{f["category"]}</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.info("💡 **提示:** 长期记忆自动从每轮问答中提取。更多问答 → 更多沉淀知识", icon="🧠")

    # 搜索功能（模拟）
    st.divider()
    search_term = st.text_input("🔍 搜索事实", placeholder="输入实体名称或关键词...")
    if search_term:
        matched = [f for f in mock_facts if search_term.lower() in f["fact"].lower() or search_term.lower() in f["entity"].lower()]
        if matched:
            st.success(f"找到 {len(matched)} 条相关事实")
            for f in matched:
                st.markdown(f"- {f['fact']} _(置信度: {f['confidence']:.0%})_")
        else:
            st.warning("未找到匹配事实")


# ============================================================
# 主布局：Tab 导航
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔧 Knowledge Builder",
    "💬 QA Workbench",
    "📊 Feedback",
    "📝 Deliverables",
    "🧠 Knowledge Wiki",
])

with tab1:
    render_knowledge_builder()
with tab2:
    render_qa_workbench()
with tab3:
    render_feedback()
with tab4:
    render_deliverables()
with tab5:
    render_knowledge_wiki()
