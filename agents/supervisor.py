"""Supervisor Agent — LangGraph graph builder and orchestrator.

Flow:
  START → classify_intent → [web_researcher | doc_analyst | summarizer | general_response] → END
"""
import logging
import uuid
from typing import Any, Dict, List

from langgraph.graph import END, StateGraph

from agents.doc_analyst import doc_analyst_node
from agents.summarizer import summarizer_node
from agents.web_researcher import web_researcher_node
from core.llm import invoke_llm
from core.prompts import CLASSIFICATION_PROMPT, GENERAL_RESPONSE_PROMPT
from core.state import AgentState

logger = logging.getLogger(__name__)

# Valid intent values
VALID_INTENTS = {"web_research", "document_qa", "summarize", "general"}


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def classify_intent_node(state: AgentState) -> Dict[str, Any]:
    """Uses LLM to classify the query intent into one of 4 categories."""
    query = state.get("query", "")
    logger.info("classify_intent_node: query=%.80r", query)

    try:
        result = invoke_llm(CLASSIFICATION_PROMPT, {"query": query}, temperature=0.0)
        raw = result.content.strip().lower()

        # Normalize — extract the first matching intent keyword
        intent = "general"
        for candidate in VALID_INTENTS:
            if candidate in raw:
                intent = candidate
                break

    except Exception as exc:
        logger.error("Intent classification failed: %s", exc)
        intent = "general"

    logger.info("classify_intent_node: intent=%s", intent)
    return {"intent": intent, "session_id": state.get("session_id") or str(uuid.uuid4())}


def general_response_node(state: AgentState) -> Dict[str, Any]:
    """Handles greetings and ambiguous queries."""
    query = state.get("query", "")
    logger.info("general_response_node: query=%.80r", query)

    try:
        result = invoke_llm(GENERAL_RESPONSE_PROMPT, {"query": query}, temperature=0.2)
        return {"response": result.content, "sources": []}
    except Exception as exc:
        logger.error("general_response_node failed: %s", exc)
        return {
            "response": "I'm sorry, I encountered an error. Please try again.",
            "sources": [],
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Router (conditional edge function)
# ---------------------------------------------------------------------------

def route_by_intent(state: AgentState) -> str:
    """Maps the classified intent to the corresponding node name."""
    intent = state.get("intent", "general")
    mapping = {
        "web_research": "web_researcher",
        "document_qa": "doc_analyst",
        "summarize": "summarizer",
        "general": "general_response",
    }
    destination = mapping.get(intent, "general_response")
    logger.debug("route_by_intent: %s → %s", intent, destination)
    return destination


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def create_graph():
    """Build and compile the LangGraph StateGraph."""
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("web_researcher", web_researcher_node)
    workflow.add_node("doc_analyst", doc_analyst_node)
    workflow.add_node("summarizer", summarizer_node)
    workflow.add_node("general_response", general_response_node)

    # Entry point
    workflow.set_entry_point("classify_intent")

    # Conditional routing after classification
    workflow.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "web_researcher": "web_researcher",
            "doc_analyst": "doc_analyst",
            "summarizer": "summarizer",
            "general_response": "general_response",
        },
    )

    # All specialist nodes go to END
    workflow.add_edge("web_researcher", END)
    workflow.add_edge("doc_analyst", END)
    workflow.add_edge("summarizer", END)
    workflow.add_edge("general_response", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# Module-level compiled graph (lazy-initialised on first import after config)
# ---------------------------------------------------------------------------
graph = create_graph()


def run(query: str, session_id: str | None = None) -> AgentState:
    """Convenience function: run the graph for a single query.

    Args:
        query:      The user's query string.
        session_id: Optional session identifier. Generated if not provided.

    Returns:
        The final AgentState after graph execution.
    """
    initial_state: AgentState = {
        "query": query,
        "session_id": session_id or str(uuid.uuid4()),
        "response": "",
        "sources": [],
        "error": None,
        "conversation_history": [],
    }
    result: AgentState = graph.invoke(initial_state)
    return result
