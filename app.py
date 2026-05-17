import re
import time
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from agent.react_agent import ReactAgent
from utils.pdf_export import markdown_to_pdf

st.set_page_config(
    page_title="海洋牧场智能调度管家",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS Injection ─────────────────────────────────────────────
st.markdown("""
<style>
/* === Global Background === */
.stApp {
    background: linear-gradient(135deg, #E8F0F6 0%, #DEE8F0 50%, #D5E2EC 100%);
}
footer { visibility: hidden; }

/* === Scrollbar === */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #D5E0EA; }
::-webkit-scrollbar-thumb { background: #0D6E8F; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #0A8FAB; }

/* === Header Banner === */
.uvais-header {
    background: linear-gradient(135deg, #C4DCE8 0%, #B0D0DE 50%, #C4DCE8 100%);
    border-radius: 10px;
    padding: 0.5rem 1.5rem;
    margin-bottom: 0.8rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 2px 16px rgba(13, 110, 143, 0.18);
}
.uvais-header::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: linear-gradient(45deg, transparent 40%, rgba(255,255,255,0.5) 50%, transparent 60%);
    animation: shimmer 8s infinite;
}
@keyframes shimmer {
    0% { transform: translateX(-100%) translateY(-100%); }
    100% { transform: translateX(100%) translateY(100%); }
}
.uvais-title {
    color: #1A2332;
    font-size: 1.1rem;
    font-weight: 700;
    margin: 0 0 0.1rem 0;
    text-shadow: 0 1px 2px rgba(255,255,255,0.5);
}
.uvais-subtitle {
    color: #4A5A6A;
    font-size: 0.72rem;
    margin: 0;
    font-weight: 400;
}
.uvais-header-decoration {
    position: absolute; bottom: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, transparent, #0D6E8F, transparent);
}

/* === Sidebar === */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #DEE8F2 0%, #E2ECF4 50%, #E8F0F6 100%);
    border-right: 1px solid rgba(13, 110, 143, 0.15);
}
[data-testid="stSidebar"] .block-container { padding: 1.8rem 1.2rem; }

/* Sidebar radio cards */
[data-testid="stSidebar"] .stRadio > div { gap: 0.5rem; }
[data-testid="stSidebar"] .stRadio [role="radiogroup"] > label {
    padding: 0.75rem 1rem;
    border-radius: 10px;
    border: 2px solid rgba(13, 110, 143, 0.2);
    margin-bottom: 0.25rem;
    transition: all 0.25s ease;
    color: #1A2332;
}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] > label:hover {
    border-color: #0D6E8F;
    background: rgba(13, 110, 143, 0.1);
}

/* Sidebar status card */
.status-card {
    background: rgba(255, 255, 255, 0.7);
    border: 1px solid rgba(13, 110, 143, 0.15);
    border-radius: 12px;
    padding: 1rem 1.1rem;
    margin: 0.5rem 0 0.75rem 0;
}
.status-card-title {
    color: #0D6E8F;
    font-weight: 600;
    font-size: 0.88rem;
    margin-bottom: 0.6rem;
}
.status-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.84rem;
    color: #2D3A4A;
    padding: 0.3rem 0;
}
.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
    background: #00E676;
    box-shadow: 0 0 8px rgba(0, 230, 118, 0.55);
    animation: pulse-dot 2s infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.45; }
}

/* === Divider === */
.stDivider { background: linear-gradient(90deg, transparent, rgba(13,110,143,0.3), transparent); height: 1px; }

/* === Quick Action Buttons === */
.stButton > button {
    background: rgba(255, 255, 255, 0.7) !important;
    border: 1px solid rgba(13, 110, 143, 0.2) !important;
    border-radius: 22px !important;
    color: #0D6E8F !important;
    font-size: 0.85rem !important;
    transition: all 0.25s ease !important;
    padding: 0.4rem 1rem !important;
}
.stButton > button:hover {
    background: rgba(13, 110, 143, 0.28) !important;
    border-color: #0A8FAB !important;
    color: #0D6E8F !important;
    transform: translateY(-2px);
    box-shadow: 0 2px 12px rgba(13, 110, 143, 0.15);
}

/* === Welcome Screen === */
.welcome-container {
    text-align: center;
    padding: 2rem 2rem;
    animation: fadeSlideIn 0.6s ease-out;
}
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(16px); }
    to { opacity: 1; transform: translateY(0); }
}
.welcome-icon { font-size: 3.5rem; margin-bottom: 0.75rem; }
.welcome-title {
    color: #1A2332;
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.welcome-desc {
    color: #5A6677;
    font-size: 0.92rem;
    margin-bottom: 1.8rem;
}
.quick-actions {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    justify-content: center;
}
.quick-action-chip {
    background: rgba(13, 110, 143, 0.06);
    border: 1px solid rgba(13, 110, 143, 0.2);
    border-radius: 22px;
    padding: 0.55rem 1.2rem;
    color: #0D6E8F;
    font-size: 0.88rem;
    cursor: default;
    transition: all 0.25s ease;
    user-select: none;
}
.quick-action-chip:hover {
    background: rgba(13, 110, 143, 0.28);
    border-color: #0A8FAB;
    color: #0D6E8F;
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(13, 110, 143, 0.2);
}

/* === Chat Messages === */
[data-testid="stChatMessage"] {
    animation: messagePopIn 0.3s ease-out;
}
@keyframes messagePopIn {
    from { opacity: 0; transform: scale(0.97); }
    to { opacity: 1; transform: scale(1); }
}
/* Assistant message container */
[data-testid="stChatMessage"]:not(:has([data-testid="stChatMessageAvatarUser"])) {
    background: rgba(255, 255, 255, 0.75);
    border-left: 3px solid #0D6E8F;
    border-radius: 0 12px 12px 0;
    padding: 0.5rem 1rem;
    margin: 0.4rem 0;
}
/* User message container */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: rgba(13, 110, 143, 0.06);
    border-right: 3px solid #0A8FAB;
    border-radius: 12px 0 0 12px;
    padding: 0.5rem 1rem;
    margin: 0.4rem 0;
}
/* Force the inner content div to use our font */
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li {
    font-size: 0.95rem;
    line-height: 1.7;
}
[data-testid="stChatMessage"] table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
    margin: 0.6rem 0;
}
[data-testid="stChatMessage"] th {
    background: rgba(13, 110, 143, 0.25);
    color: #0D6E8F;
    padding: 0.4rem 0.8rem;
    text-align: left;
    border-bottom: 2px solid rgba(13, 110, 143, 0.4);
}
[data-testid="stChatMessage"] td {
    padding: 0.35rem 0.8rem;
    border-bottom: 1px solid rgba(13, 110, 143, 0.15);
    color: #1A2332;
}


/* === Spinner === */
.stSpinner > div { border-color: #0D6E8F !important; }

/* === Code blocks inside chat === */
[data-testid="stChatMessage"] code {
    background: rgba(13, 110, 143, 0.18);
    color: #0D6E8F;
    padding: 0.15rem 0.4rem;
    border-radius: 4px;
    font-size: 0.88rem;
}
[data-testid="stChatMessage"] pre {
    background: #F0F4F8;
    border: 1px solid rgba(13, 110, 143, 0.15);
    border-radius: 8px;
    padding: 1rem;
    overflow-x: auto;
}

/* === Markdown inside chat === */
[data-testid="stChatMessage"] h1 { font-size: 1.3rem; color: #1A2332; }
[data-testid="stChatMessage"] h2 { font-size: 1.15rem; color: #1A2332; }
[data-testid="stChatMessage"] h3 { font-size: 1.05rem; color: #0A8FAB; }
[data-testid="stChatMessage"] strong { color: #0D1B36; }
[data-testid="stChatMessage"] hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(13,110,143,0.3), transparent);
}

/* === Info box in sidebar === */
[data-testid="stSidebar"] .stAlert {
    background: rgba(255, 255, 255, 0.7);
    border: 1px solid rgba(13, 110, 143, 0.15);
    border-radius: 10px;
    color: #1A2332;
}

/* === Streamlit Native Bottom Container ===
   st.chat_input lives in [data-testid="stBottomBlockContainer"] which
   is properly fixed to the viewport bottom by Streamlit's own layout. */
[data-testid="stBottomBlockContainer"] {
    background: rgba(210, 228, 240, 0.95) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border-top: 1px solid rgba(13, 110, 143, 0.18) !important;
    box-shadow: 0 -2px 16px rgba(13, 110, 143, 0.10) !important;
    padding: 0.5rem 2rem 0.65rem 2rem !important;
}
/* Chat input textarea */
[data-testid="stBottomBlockContainer"] textarea {
    background: rgba(255, 255, 255, 0.85) !important;
    border: 1px solid rgba(13, 110, 143, 0.2) !important;
    border-radius: 8px !important;
    color: #1A2332 !important;
    padding: 0.5rem 0.85rem !important;
    font-size: 0.925rem !important;
}
[data-testid="stBottomBlockContainer"] textarea:focus {
    border-color: #0D6E8F !important;
    box-shadow: 0 0 0 2px rgba(13, 110, 143, 0.2) !important;
}
[data-testid="stBottomBlockContainer"] textarea::placeholder {
    color: #9AA5B4 !important;
}
/* Bottom padding so chat content doesn't hide behind fixed bar */
[data-testid="stAppViewContainer"] > .block-container {
    padding-bottom: 5rem !important;
}

/* === Download button === */
.download-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 1.2rem;
    margin-top: 0.3rem;
}
.wo-badge {
    display: inline-block;
    background: rgba(0, 180, 100, 0.12);
    border: 1px solid rgba(0, 180, 100, 0.3);
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    color: #00804A;
    font-size: 0.78rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ── Session State Init ───────────────────────────────────────────────
if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "message" not in st.session_state:
    st.session_state["message"] = []

# Preview mode for screenshots (?preview=manager or ?preview=worker)
_preview_role = st.query_params.get("preview", "")
if _preview_role in ("manager", "worker") and not st.session_state["message"]:
    if _preview_role == "manager":
        st.session_state["message"] = [
            {"role": "user", "content": "查询B区实时海况"},
            {"role": "assistant", "content": "**【海况数据 — B区】**\n\n| 指标 | 数值 | 状态 |\n|------|------|------|\n| 浪高 | 0.8m | ✅ 安全 |\n| 底层流速 | 0.3m/s | ✅ 安全 |\n| 风速 | 12km/h | ✅ 正常 |\n| 水温 | 22°C | ✅ 正常 |\n\n当前B区海况良好，符合人工下水作业安全标准。如需派单请告知具体任务。"},
            {"role": "user", "content": "派发B区网箱巡检工单"},
            {"role": "assistant", "content": "**【安全联锁检查】**\n- 浪高 0.8m ≤ 2.5m ✅\n- 底层流速 0.3m/s ≤ 1.0m/s ✅\n\n环境安全，允许派单。\n\n---\n\n## 📋 工单 — B区网箱巡检\n\n| 项目 | 内容 |\n|------|------|\n| 工单编号 | WO-2026-0515-001 |\n| 作业区域 | B区深水网箱 |\n| 作业类型 | 常规巡检 |\n| 派遣人员 | 2人（1名ROV操作员 + 1名安全员） |\n| 预计时长 | 4小时 |\n| 安全等级 | 三级（常规） |\n\n**SOP参考：** 依据《水下基础设施巡检与维修标准》执行网箱结构完整性检查、锚链张力检测及网衣破损排查。"},
        ]
    else:
        st.session_state["message"] = [
            {"role": "user", "content": "获取当前位置与测距"},
            {"role": "assistant", "content": "**【定位信息】距目标维修点 12.5 米，当前深度 18.3 米。**\n\n| 导航参数 | 数值 |\n|----------|------|\n| 经度 | 121.5372°E |\n| 纬度 | 28.4197°N |\n| 深度 | 18.3m |\n| 距目标 | 12.5m（东北方向） |\n| 水温 | 21°C |\n| 能见度 | 6.2m |\n\n当前位置稳定，可开展作业。"},
            {"role": "user", "content": "当前海况是否安全"},
            {"role": "assistant", "content": "**【海况安全检查】**\n\n| 指标 | 当前值 | 安全阈值 | 状态 |\n|------|--------|----------|------|\n| 浪高 | 0.6m | ≤2.5m | ✅ |\n| 底层流速 | 0.2m/s | ≤1.0m/s | ✅ |\n| 风速 | 10km/h | ≤30km/h | ✅ |\n\n当前海况安全，可继续作业。如遇海况突变请立即执行紧急撤离程序。"},
        ]

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem;">
        <span style="font-size:1.6rem;">🌊</span>
        <span style="font-size:1.1rem;font-weight:700;color:#1A2332;">UVAIS</span>
    </div>
    """, unsafe_allow_html=True)
    st.subheader("👤 角色切换")

    _default_idx = 1 if _preview_role == "worker" else 0
    role = st.radio(
        "选择当前身份",
        options=["manager", "worker"],
        index=_default_idx,
        format_func=lambda x: "🏗️ 总控人员 (Manager)" if x == "manager" else "🔧 水下作业人员 (Worker)",
        horizontal=False,
    )

    st.divider()

    # System status card
    st.markdown("""
    <div class="status-card">
        <div class="status-card-title">📊 系统状态</div>
        <div class="status-indicator"><span class="status-dot"></span> Agent 就绪</div>
        <div class="status-indicator"><span class="status-dot"></span> Milvus 在线</div>
    </div>
    """, unsafe_allow_html=True)

    if role == "manager":
        st.info("**总控模式**：环境评估 · 安全联锁 · 派单审批 · 工单生成")
        st.divider()
        st.subheader("📤 知识库管理")
        with st.container(border=True):
            uploaded_files = st.file_uploader(
                "上传 .txt / .pdf 文档到知识库",
                type=["txt", "pdf"],
                accept_multiple_files=True,
                key="kb_uploader",
            )
            if uploaded_files:
                if st.button("📥 入库到知识库", use_container_width=True, key="btn_ingest"):
                    import os
                    from rag.milvus_service import MilvusService
                    data_dir = "data"
                    os.makedirs(data_dir, exist_ok=True)
                    saved_count = 0
                    for uf in uploaded_files:
                        dest = os.path.join(data_dir, uf.name)
                        with open(dest, "wb") as f:
                            f.write(uf.getbuffer())
                        saved_count += 1
                    with st.spinner(f"已保存 {saved_count} 个文件，正在向量化入库..."):
                        try:
                            MilvusService().build_knowledge_base()
                            st.success(f"入库完成：{saved_count} 个文件已加入知识库")
                        except Exception as e:
                            st.error(f"入库失败：{e}")
    else:
        st.info("**作业模式**：自动定位 · 海况预警 · SOP 要点 · 紧急撤离")

# ── Main Content ──────────────────────────────────────────────────────

# Header banner
st.markdown("""
<div class="uvais-header">
    <h1 class="uvais-title">🌊 海洋牧场智能调度管家</h1>
    <div class="uvais-header-decoration"></div>
</div>
""", unsafe_allow_html=True)

# Welcome screen (shown when no messages)
if not st.session_state["message"]:
    if role == "manager":
        st.markdown("""
        <div class="welcome-container">
            <div class="welcome-icon">🏗️</div>
            <h2 class="welcome-title">总控调度中心</h2>
            <p class="welcome-desc">输入任务需求或点击下方快捷指令开始调度</p>
        </div>
        """, unsafe_allow_html=True)
        cols = st.columns(4)
        with cols[0]:
            if st.button("🌊 查询B区实时海况", key="qm0", use_container_width=True):
                st.session_state["chip_query"] = "查询B区实时海况"
        with cols[1]:
            if st.button("📋 派发网箱巡检工单", key="qm1", use_container_width=True):
                st.session_state["chip_query"] = "派发网箱巡检工单"
        with cols[2]:
            if st.button("🔍 查看SEAD-YOLO检测结果", key="qm2", use_container_width=True):
                st.session_state["chip_query"] = "查看SEAD-YOLO检测结果"
        with cols[3]:
            if st.button("📄 生成周度安全报告", key="qm3", use_container_width=True):
                st.session_state["chip_query"] = "生成周度安全报告"
    else:
        st.markdown("""
        <div class="welcome-container">
            <div class="welcome-icon">🔧</div>
            <h2 class="welcome-title">水下作业终端</h2>
            <p class="welcome-desc">输入现场情况或点击下方快捷指令开始作业</p>
        </div>
        """, unsafe_allow_html=True)
        cols = st.columns(4)
        with cols[0]:
            if st.button("📍 获取当前定位与测距", key="qw0", use_container_width=True):
                st.session_state["chip_query"] = "获取当前定位与测距"
        with cols[1]:
            if st.button("🌊 查询当前海况安全", key="qw1", use_container_width=True):
                st.session_state["chip_query"] = "查询当前海况安全"
        with cols[2]:
            if st.button("📖 查阅SOP操作规程", key="qw2", use_container_width=True):
                st.session_state["chip_query"] = "查阅SOP操作规程"
        with cols[3]:
            if st.button("📸 查看ROV视觉画面", key="qw3", use_container_width=True):
                st.session_state["chip_query"] = "查看ROV视觉画面"

# Chat history
for i, message in enumerate(st.session_state["message"]):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Work order download button
        if message["role"] == "assistant" and ("工单编号" in message["content"] or "自动化处置工单" in message["content"]):
            wo_match = re.search(r'WO-[\d-]+', message["content"])
            wo_id = wo_match.group(0) if wo_match else "workorder"
            try:
                pdf_data = bytes(markdown_to_pdf(message["content"]))
            except Exception:
                pdf_data = message["content"].encode("utf-8")
            st.download_button(
                label=f"📥 下载工单 ({wo_id})",
                data=pdf_data,
                file_name=f"{wo_id}.pdf",
                mime="application/pdf",
                key=f"dl_{i}",
            )

# ── Fixed Bottom Bar ──
# st.chat_input is natively fixed to the viewport bottom by Streamlit.
prompt = st.chat_input(placeholder="输入您的指令...")

# Handle chip quick-action prompts (welcome screen)
chip_prompt = st.session_state.pop("chip_query", None) if "chip_query" in st.session_state else None
if chip_prompt:
    prompt = chip_prompt

if prompt:
    st.chat_message("user").markdown(prompt)

    chat_history = st.session_state["message"].copy()
    st.session_state["message"].append({"role": "user", "content": prompt})

    response_messages = []
    with st.spinner("智能中枢思考中..."):
        res_stream = st.session_state["agent"].execute_stream(
            prompt, history=chat_history, role=role
        )

        def capture(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                for char in chunk:
                    time.sleep(0.01)
                    yield char

        st.chat_message("assistant").write_stream(capture(res_stream, response_messages))
        st.session_state["message"].append(
            {"role": "assistant", "content": response_messages[-1]}
        )
    st.rerun()
