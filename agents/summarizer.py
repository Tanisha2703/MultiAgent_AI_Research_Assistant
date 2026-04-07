"""Summarizer Agent — Map-reduce pattern for arbitrarily long documents."""
import logging
from typing import Any, Dict

from config import settings
from core.embeddings import vector_store_manager
from core.llm import invoke_llm, invoke_llm_raw
from core.prompts import SUMMARIZE_MAP_PROMPT, SUMMARIZE_REDUCE_PROMPT
from core.state import AgentState

logger = logging.getLogger(__name__)


def _map_chunk(chunk_text: str) -> str:
    """Summarize a single chunk (map phase)."""
    prompt = SUMMARIZE_MAP_PROMPT.format(text=chunk_text)
    result = invoke_llm_raw(prompt, temperature=0.3)
    return result.content


def summarizer_node(state: AgentState) -> Dict[str, Any]:
    """LangGraph node: map-reduce summarization of session documents.

    - For small documents (≤3 chunks): single-pass summarization via reduce prompt.
    - For larger documents: map each chunk independently, then reduce.
    """
    query = state.get("query", "")
    session_id = state.get("session_id", "")
    logger.info("summarizer_node: session=%s query=%.80r", session_id, query)

    if not vector_store_manager.has_documents(session_id):
        return {
            "response": (
                "No documents have been uploaded for this session. "
                "Please upload a PDF, TXT, or Markdown file first using the sidebar."
            ),
            "sources": [],
        }

    # Broad retrieval — we want coverage, not just top-5 relevance
    docs = vector_store_manager.similarity_search(
        session_id, query or "summarize the document", k=settings.SUMMARIZE_TOP_K
    )

    if not docs:
        return {
            "response": "No content found in the uploaded documents to summarize.",
            "sources": [],
        }

    try:
        if len(docs) <= 3:
            # Single-pass: stuff all chunks into reduce prompt directly
            combined = "\n\n".join(doc.page_content for doc in docs)
            result = invoke_llm(SUMMARIZE_REDUCE_PROMPT, {"summaries": combined}, temperature=0.3)
            response = result.content
        else:
            # Map phase: summarize each chunk independently
            chunk_summaries: list[str] = []
            for i, doc in enumerate(docs):
                logger.debug("map phase: chunk %d/%d", i + 1, len(docs))
                summary = _map_chunk(doc.page_content)
                chunk_summaries.append(f"Section {i + 1}: {summary}")

            # Reduce phase: merge all chunk summaries into final output
            combined_summaries = "\n\n".join(chunk_summaries)
            result = invoke_llm(SUMMARIZE_REDUCE_PROMPT, {"summaries": combined_summaries}, temperature=0.3)
            response = result.content

    except Exception as exc:
        logger.error("Summarization failed: %s", exc)
        return {
            "response": "Summarization failed. Please try again.",
            "sources": [],
            "error": str(exc),
        }

    # Source metadata
    seen: set[str] = set()
    sources = []
    for doc in docs:
        fname = doc.metadata.get("filename", doc.metadata.get("source", "document"))
        if fname not in seen:
            seen.add(fname)
            sources.append({"title": fname, "url": "", "relevance_score": 1.0})

    return {"response": response, "sources": sources}
