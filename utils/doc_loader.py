"""
Document loading helpers.
Supports: PDF, DOCX, TXT, MD, CSV
"""

import os
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
)


# Map file extensions to their LangChain loaders
LOADER_MAP = {
    ".pdf":  PyMuPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt":  TextLoader,
    ".md":   UnstructuredMarkdownLoader,
    ".csv":  CSVLoader,
}


def load_single_document(file_path: str):
    """Load a single document based on its file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    loader_cls = LOADER_MAP.get(ext)
    if loader_cls is None:
        raise ValueError(f"Unsupported file format: {ext}")
    loader = loader_cls(file_path)
    docs = loader.load()
    # Inject source filename into metadata
    for doc in docs:
        doc.metadata["source"] = os.path.basename(file_path)
    return docs


def load_documents_from_dir(directory: str):
    """Load all supported documents from a directory."""
    all_docs = []
    for filename in sorted(os.listdir(directory)):
        ext = os.path.splitext(filename)[1].lower()
        if ext in LOADER_MAP:
            file_path = os.path.join(directory, filename)
            all_docs.extend(load_single_document(file_path))
    return all_docs
