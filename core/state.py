"""Shared AgentState TypedDict — the single data contract used across all nodes."""
from typing import TypedDict, List, Optional


class Source(TypedDict, total=False):
    title: str
    url: str
    relevance_score: float
    content: str


class AgentState(TypedDict, total=False):
    # Input
    query: str
    session_id: str

    # Routing
    intent: str  # web_research | document_qa | summarize | general

    # Output
    response: str
    sources: List[Source]

    # Error handling
    error: Optional[str]

    # Conversation context
    conversation_history: List[dict]
