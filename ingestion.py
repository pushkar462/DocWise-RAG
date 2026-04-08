import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import config
from utils.doc_loader import load_documents_from_dir, load_single_document
from utils.chunker import chunk_documents


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def ingest_from_directory(directory=None):
    directory = directory or config.RAW_DOCS_DIR
    documents = load_documents_from_dir(directory)
    if not documents:
        raise ValueError("No supported documents found.")
    chunks = chunk_documents(documents)
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs(config.FAISS_INDEX_DIR, exist_ok=True)
    vectorstore.save_local(config.FAISS_INDEX_DIR)
    return vectorstore


def ingest_uploaded_files(file_paths):
    embeddings = get_embeddings()
    all_chunks = []
    for path in file_paths:
        docs = load_single_document(path)
        chunks = chunk_documents(docs)
        all_chunks.extend(chunks)
    if not all_chunks:
        raise ValueError("No content extracted from uploaded files.")
    if os.path.exists(os.path.join(config.FAISS_INDEX_DIR, "index.faiss")):
        existing_vs = FAISS.load_local(
            config.FAISS_INDEX_DIR, embeddings,
            allow_dangerous_deserialization=True,
        )
        new_vs = FAISS.from_documents(all_chunks, embeddings)
        existing_vs.merge_from(new_vs)
        vectorstore = existing_vs
    else:
        vectorstore = FAISS.from_documents(all_chunks, embeddings)
    os.makedirs(config.FAISS_INDEX_DIR, exist_ok=True)
    vectorstore.save_local(config.FAISS_INDEX_DIR)
    return vectorstore, len(all_chunks)


def load_existing_index():
    embeddings = get_embeddings()
    index_path = os.path.join(config.FAISS_INDEX_DIR, "index.faiss")
    if not os.path.exists(index_path):
        return None
    return FAISS.load_local(
        config.FAISS_INDEX_DIR, embeddings,
        allow_dangerous_deserialization=True,
    )