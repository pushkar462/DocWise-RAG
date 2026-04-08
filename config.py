import os
from dotenv import load_dotenv

load_dotenv()

# ── Groq (check Streamlit secrets first, then .env) ────
try:
    import streamlit as st
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
except:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── Groq (free, fast) ──────────────────────────────────

# ── Embedding (local, free — runs on your machine) ─────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── LLM (Groq free tier) ───────────────────────────────
LLM_MODEL = "llama-3.1-8b-instant"
LLM_TEMPERATURE = 0.2

# ── Chunking ────────────────────────────────────────────
CHUNK_SIZE = 1000      # was 512 — bigger chunks keep headings with their content
CHUNK_OVERLAP = 200    # was 50 — more overlap prevents losing context at boundaries

# ── Retrieval ───────────────────────────────────────────
TOP_K = 8

# ── Paths ───────────────────────────────────────────────
RAW_DOCS_DIR = os.path.join(os.path.dirname(__file__), "data", "raw_docs")
FAISS_INDEX_DIR = os.path.join(os.path.dirname(__file__), "vectorstore", "faiss_index")