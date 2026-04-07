"""Document Analysis Agent — RAG pipeline: FAISS retrieval → grounded LLM answer."""
import logging
from typing import Any, Dict

from langchain_core.documents import Document

from config import settings
from core.embeddings import vector_store_manager
from core.llm import invoke_llm
from core.prompts import DOC_ANALYST_PROMPT
from core.state import AgentState

logger = logging.getLogger(__name__)


def _assemble_context(docs: list[Document]) -> str:
    """Build a numbered context string from retrieved chunks, with source metadata."""
    parts: list[str] = []
    for i, doc in enumerate(docs, start=1):
        filename = doc.metadata.get("filename", doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page", "")
        page_str = f", page {page + 1}" if page != "" else ""
        parts.append(f"[Chunk {i} — {filename}{page_str}]\n{doc.page_content}")
    return "\n\n".join(parts)


def doc_analyst_node(state: AgentState) -> Dict[str, Any]:
    """LangGraph node: retrieves relevant chunks from FAISS and answers the query."""
    query = state.get("query", "")
    session_id = state.get("session_id", "")
    logger.info("doc_analyst_node: session=%s query=%.80r", session_id, query)

    if not vector_store_manager.has_documents(session_id):
        return {
            "response": (
                "No documents have been uploaded for this session. "
                "Please upload a PDF, TXT, or Markdown file first using the sidebar."
            ),
            "sources": [],
        }

    docs = vector_store_manager.similarity_search(
        session_id, query, k=settings.TOP_K_RETRIEVAL
    )

    if not docs:
        return {
            "response": "No relevant content found in the uploaded documents for your query.",
            "sources": [],
        }

    context = _assemble_context(docs)

    try:
        result = invoke_llm(
            DOC_ANALYST_PROMPT,
            {"query": query, "context": context},
            temperature=0.1,
        )
        response = result.content
    except Exception as exc:
        logger.error("LLM call failed in doc_analyst_node: %s", exc)
        return {
            "response": "Document retrieval succeeded but answer generation failed. Please try again.",
            "sources": [],
            "error": str(exc),
        }

    # Build sources list for the response (unique filenames only)
    seen: set[str] = set()
    sources = []
    for doc in docs:
        fname = doc.metadata.get("filename", doc.metadata.get("source", "document"))
        if fname not in seen:
            seen.add(fname)
            sources.append({"title": fname, "url": "", "relevance_score": 1.0})

    return {"response": response, "sources": sources}
