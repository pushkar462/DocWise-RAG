"""
Multi-Doc Q&A Chatbot — Streamlit Frontend
LangChain + FAISS + OpenAI GPT-4o
"""
import os
import json
import tempfile
import streamlit as st
from datetime import datetime

import config
from ingestion import ingest_uploaded_files, load_existing_index
from query_engine import ask_question


# ══════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Multi-Doc Q&A Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════
# LIGHT THEME CSS
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* ── Global light theme ────────────────────────────── */
    .stApp {
        background-color: #f8f9fa;
    }

    /* ── Sidebar ───────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e9ecef;
    }

    /* ── Header ────────────────────────────────────────── */
    .main-header {
        background: linear-gradient(135deg, #1a7f64 0%, #2ba882 100%);
        padding: 1.8rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.75rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .main-header p {
        margin: 0.3rem 0 0;
        opacity: 0.9;
        font-size: 0.95rem;
    }

    /* ── Chat message bubbles ──────────────────────────── */
    .user-msg {
        background-color: #1a7f64;
        color: white;
        padding: 1rem 1.25rem;
        border-radius: 16px 16px 4px 16px;
        margin: 0.6rem 0;
        max-width: 80%;
        margin-left: auto;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .bot-msg {
        background-color: #ffffff;
        color: #1a1a2e;
        padding: 1rem 1.25rem;
        border-radius: 16px 16px 16px 4px;
        margin: 0.6rem 0;
        max-width: 85%;
        border: 1px solid #e9ecef;
        font-size: 0.95rem;
        line-height: 1.6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    /* ── Source citation cards ──────────────────────────── */
    .source-card {
        background-color: #f0faf6;
        border-left: 3px solid #1a7f64;
        padding: 0.6rem 1rem;
        margin: 0.3rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.82rem;
        color: #374151;
    }
    .source-card strong {
        color: #1a7f64;
    }

    /* ── Uploaded file chips ───────────────────────────── */
    .file-chip {
        display: inline-block;
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 0.3rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.2rem;
        font-weight: 500;
    }

    /* ── Stats row ─────────────────────────────────────── */
    .stat-box {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #e9ecef;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .stat-box .stat-num {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1a7f64;
    }
    .stat-box .stat-label {
        font-size: 0.78rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* ── Buttons ───────────────────────────────────────── */
    .stButton > button {
        background-color: #1a7f64;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background-color: #15674f;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(26,127,100,0.3);
    }

    /* ── Hide Streamlit defaults ───────────────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Input field ───────────────────────────────────── */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 1.5px solid #d1d5db;
        padding: 0.75rem 1rem;
        font-size: 0.95rem;
    }
    .stTextInput > div > div > input:focus {
        border-color: #1a7f64;
        box-shadow: 0 0 0 2px rgba(26,127,100,0.15);
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_existing_index()
if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0


# ══════════════════════════════════════════════════════════
# SIDEBAR — FILE UPLOAD + SETTINGS
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📁 Document Manager")
    st.markdown("---")

    # ── File uploader ──────────────────────────────────
    uploaded_files = st.file_uploader(
        "Upload your documents",
        type=["pdf", "docx", "txt", "md", "csv"],
        accept_multiple_files=True,
        help="Supported: PDF, DOCX, TXT, MD, CSV",
    )

    if uploaded_files:
        if st.button("🚀 Process Documents", use_container_width=True):
            with st.spinner("Indexing documents..."):
                # Save uploaded files to temp dir
                temp_paths = []
                for uf in uploaded_files:
                    temp_dir = tempfile.mkdtemp()
                    temp_path = os.path.join(temp_dir, uf.name)
                    with open(temp_path, "wb") as f:
                        f.write(uf.getbuffer())
                    temp_paths.append(temp_path)
                    if uf.name not in st.session_state.uploaded_files_list:
                        st.session_state.uploaded_files_list.append(uf.name)

                try:
                    vs, num_chunks = ingest_uploaded_files(temp_paths)
                    st.session_state.vectorstore = vs
                    st.session_state.total_chunks += num_chunks
                    st.success(f"✅ Indexed {len(uploaded_files)} file(s) → {num_chunks} chunks")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    # ── Indexed files list ─────────────────────────────
    if st.session_state.uploaded_files_list:
        st.markdown("### 📄 Indexed Documents")
        for fname in st.session_state.uploaded_files_list:
            ext = os.path.splitext(fname)[1].upper()
            st.markdown(f'<span class="file-chip">{ext} {fname}</span>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Settings ───────────────────────────────────────
    st.markdown("### ⚙️ Settings")
    config.TOP_K = st.slider("Top-K retrieval", 1, 15, config.TOP_K)
    config.LLM_TEMPERATURE = st.slider("LLM Temperature", 0.0, 1.0, config.LLM_TEMPERATURE, 0.1)

    st.markdown("---")

    # ── Export chat ────────────────────────────────────
    st.markdown("### 💾 Export")
    if st.session_state.chat_history:
        # Build export data
        export_data = {
            "exported_at": datetime.now().isoformat(),
            "total_messages": len(st.session_state.chat_history),
            "documents_indexed": st.session_state.uploaded_files_list,
            "conversation": [],
        }
        for msg in st.session_state.chat_history:
            entry = {"role": msg["role"], "content": msg["content"]}
            if msg["role"] == "assistant" and msg.get("sources"):
                entry["sources"] = msg["sources"]
            export_data["conversation"].append(entry)

        json_str = json.dumps(export_data, indent=2, ensure_ascii=False)

        # Markdown/Text export
        md_lines = [
            f"# Multi-Doc Q&A — Chat Export",
            f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**Documents:** {', '.join(st.session_state.uploaded_files_list)}",
            "",
            "---",
            "",
        ]
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                md_lines.append(f"### 🧑 You\n{msg['content']}\n")
            else:
                md_lines.append(f"### 🤖 Assistant\n{msg['content']}\n")
                if msg.get("sources"):
                    md_lines.append("**Sources:**")
                    for s in msg["sources"]:
                        md_lines.append(f"- {s['source']} (Page {s['page']})")
                    md_lines.append("")
        md_str = "\n".join(md_lines)

        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button(
                "📥 JSON",
                data=json_str,
                file_name="chat_export.json",
                mime="application/json",
                use_container_width=True,
            )
        with col_b:
            st.download_button(
                "📥 Markdown",
                data=md_str,
                file_name="chat_export.md",
                mime="text/markdown",
                use_container_width=True,
            )
    else:
        st.caption("Start a conversation to enable export.")

    st.markdown("---")

    # ── Clear chat ─────────────────────────────────────
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ══════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════

# ── Header ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📚 Multi-Doc Q&A Chatbot</h1>
    <p>Upload documents and ask questions — powered by LangChain, FAISS & Groq</p>
</div>
""", unsafe_allow_html=True)

# ── Stats row ──────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-num">{len(st.session_state.uploaded_files_list)}</div>
        <div class="stat-label">Documents</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-num">{st.session_state.total_chunks}</div>
        <div class="stat-label">Chunks</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-num">{len(st.session_state.chat_history) // 2}</div>
        <div class="stat-label">Questions</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-num">{config.TOP_K}</div>
        <div class="stat-label">Top-K</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Chat history display ──────────────────────────────
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">{msg["content"]}</div>', unsafe_allow_html=True)
        # Show sources
        if msg.get("sources"):
            for src in msg["sources"]:
                st.markdown(
                    f'<div class="source-card">'
                    f'<strong>📎 {src["source"]}</strong> — Page {src["page"]}<br>'
                    f'<em>{src["snippet"][:120]}...</em>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

# ── Empty state ────────────────────────────────────────
if not st.session_state.chat_history and not st.session_state.uploaded_files_list:
    st.markdown("""
    <div style="text-align:center; padding:3rem 1rem; color:#9ca3af;">
        <div style="font-size:3rem; margin-bottom:0.5rem;">📄</div>
        <div style="font-size:1.1rem; font-weight:600; color:#6b7280;">No documents uploaded yet</div>
        <div style="font-size:0.9rem; margin-top:0.3rem;">
            Upload PDFs, DOCX, TXT, or CSV files using the sidebar to get started.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Question input ─────────────────────────────────────
question = st.chat_input("Ask a question about your documents...")

if question:
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": question})
    st.markdown(f'<div class="user-msg">{question}</div>', unsafe_allow_html=True)

    # Get answer
    with st.spinner("🔍 Searching documents and generating answer..."):
        result = ask_question(question, st.session_state.vectorstore)

    # Add bot message
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })

    # Display answer
    st.markdown(f'<div class="bot-msg">{result["answer"]}</div>', unsafe_allow_html=True)
    if result["sources"]:
        for src in result["sources"]:
            st.markdown(
                f'<div class="source-card">'
                f'<strong>📎 {src["source"]}</strong> — Page {src["page"]}<br>'
                f'<em>{src["snippet"][:120]}...</em>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.rerun()
