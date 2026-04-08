# 📚 Multi-Doc Q&A Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload documents and ask questions about them.

**Stack:** LangChain · FAISS · OpenAI GPT-4o · Streamlit

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
cd multi-doc-qa-chatbot
pip install -r requirements.txt
```

### 2. Add your OpenAI API key

Edit `.env` and replace the placeholder:

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

### 3. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## 📁 Folder Structure

```
multi-doc-qa-chatbot/
├── app.py                  # Streamlit frontend
├── ingestion.py            # Indexing pipeline
├── query_engine.py         # Query pipeline
├── config.py               # Central configuration
├── requirements.txt
├── .env                    # API key (not committed)
├── .streamlit/
│   └── config.toml         # Light theme config
├── data/
│   └── raw_docs/           # Place source documents here
├── vectorstore/
│   └── faiss_index/        # Auto-generated FAISS index
├── utils/
│   ├── __init__.py
│   ├── doc_loader.py       # Document loading helpers
│   ├── chunker.py          # Text splitting logic
│   └── prompt_templates.py # LLM prompt templates
└── assets/
    └── logo.png            # Optional branding
```

---

## 🔧 How It Works

### Phase 1 — Indexing (runs once per upload)
`Raw docs → Doc Loader → Chunker (512 tok, 50 overlap) → OpenAI Embedder → FAISS`

### Phase 2 — Query (every question)
`User question → Embedder → FAISS similarity search → Prompt Builder → GPT-4o → Answer with citations`

---

## 📤 Export

Chat history can be exported as **JSON** or **Markdown** from the sidebar.

---

## 📝 Supported File Types

| Format   | Extension |
|----------|-----------|
| PDF      | `.pdf`    |
| Word     | `.docx`   |
| Text     | `.txt`    |
| Markdown | `.md`     |
| CSV      | `.csv`    |
