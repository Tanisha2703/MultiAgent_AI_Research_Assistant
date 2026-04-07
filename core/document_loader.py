"""Document ingestion & chunking pipeline.

Supports: PDF (.pdf), plain text (.txt), Markdown (.md)
Uses RecursiveCharacterTextSplitter to preserve semantic coherence.
"""
import logging
import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings

logger = logging.getLogger(__name__)


def load_and_chunk(file_path: str, filename: str) -> List[Document]:
    """Load a file, extract text, and split into semantically coherent chunks.

    Args:
        file_path: Absolute path to the saved file on disk.
        filename:  Original filename (used to determine loader + metadata).

    Returns:
        List of Document chunks, each with source/page metadata.
    """
    ext = os.path.splitext(filename)[1].lower()
    logger.info("Loading file=%s (ext=%s)", filename, ext)

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        raw_docs = loader.load()
    elif ext in (".txt", ".md"):
        loader = TextLoader(file_path, encoding="utf-8")
        raw_docs = loader.load()
        # Set a meaningful source metadata entry
        for doc in raw_docs:
            doc.metadata.setdefault("source", filename)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # Attach friendly filename to metadata for citation purposes
    for doc in raw_docs:
        doc.metadata["filename"] = filename

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks = splitter.split_documents(raw_docs)
    logger.info("file=%s → %d raw pages → %d chunks", filename, len(raw_docs), len(chunks))
    return chunks
