"""
Text splitting / chunking logic.
Uses LangChain's RecursiveCharacterTextSplitter with tiktoken token counting.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
import config


def get_text_splitter():
    """Return a configured text splitter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,          # character-based; swap to tiktoken for token-based
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_documents(documents):
    """Split a list of LangChain Documents into smaller chunks."""
    splitter = get_text_splitter()
    chunks = splitter.split_documents(documents)

    # Add chunk_index to metadata
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx

    return chunks
