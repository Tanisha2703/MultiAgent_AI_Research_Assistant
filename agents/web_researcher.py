"""Web Research Agent — Tavily search → LLM synthesis → cited report."""
import logging
from typing import Any, Dict

from tavily import TavilyClient

from config import settings
from core.llm import invoke_llm
from core.prompts import WEB_RESEARCHER_PROMPT
from core.state import AgentState, Source

logger = logging.getLogger(__name__)

_tavily = TavilyClient(api_key=settings.TAVILY_API_KEY)


def _format_search_results(results: list) -> tuple[str, list[Source]]:
    """Format Tavily results into a numbered string for the prompt and a sources list."""
    formatted_lines: list[str] = []
    sources: list[Source] = []

    for i, r in enumerate(results, start=1):
        title = r.get("title", "No title")
        url = r.get("url", "")
        content = r.get("content", "")
        score = r.get("score", 0.0)

        formatted_lines.append(
            f"[{i}] Title: {title}\n    URL: {url}\n    Content: {content}\n"
        )
        sources.append(Source(title=title, url=url, relevance_score=score, content=content))

    return "\n".join(formatted_lines), sources


def web_researcher_node(state: AgentState) -> Dict[str, Any]:
    """LangGraph node: performs web search and synthesizes a cited report."""
    query = state.get("query", "")
    logger.info("web_researcher_node: query=%.80r", query)

    try:
        raw_results = _tavily.search(query, max_results=5).get("results", [])
    except Exception as exc:
        logger.error("Tavily search failed: %s", exc)
        return {
            "response": "I encountered an error while searching the web. Please try again.",
            "sources": [],
            "error": str(exc),
        }

    if not raw_results:
        return {
            "response": "No web results found for your query. Try rephrasing it.",
            "sources": [],
        }

    search_text, sources = _format_search_results(raw_results)

    try:
        result = invoke_llm(
            WEB_RESEARCHER_PROMPT,
            {"query": query, "search_results": search_text},
            temperature=0.2,
        )
        response = result.content
    except Exception as exc:
        logger.error("LLM synthesis failed: %s", exc)
        return {
            "response": "Search succeeded but response generation failed. Please try again.",
            "sources": sources,
            "error": str(exc),
        }

    return {"response": response, "sources": sources}
