"""Unit tests — all LLM calls are mocked to avoid real API costs.

Run with:
    pytest tests/ -v
"""
import os

# Set test credentials BEFORE importing any project modules
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")

import uuid
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# 1. AgentState schema validation
# ---------------------------------------------------------------------------

class TestAgentState:
    def test_valid_state(self):
        from core.state import AgentState

        state: AgentState = {
            "query": "What is LangGraph?",
            "session_id": str(uuid.uuid4()),
            "intent": "web_research",
            "response": "LangGraph is…",
            "sources": [],
            "error": None,
            "conversation_history": [],
        }
        assert state["query"] == "What is LangGraph?"
        assert state["intent"] == "web_research"

    def test_optional_fields(self):
        from core.state import AgentState

        # Minimal state (total=False allows partial dicts)
        state: AgentState = {"query": "hello"}
        assert state.get("intent") is None
        assert state.get("sources") is None


# ---------------------------------------------------------------------------
# 2. Intent classification signals (keyword matching logic)
# ---------------------------------------------------------------------------

class TestIntentSignals:
    web_signals = ["latest", "recent", "current", "2025", "today", "news"]
    doc_signals = ["document", "uploaded", "file", "the paper", "the report"]
    summarize_signals = ["summarize", "summary", "key points", "tl;dr"]

    def test_web_research_signals(self):
        query = "What are the latest AI trends in 2025?"
        assert any(s in query.lower() for s in self.web_signals)

    def test_document_qa_signals(self):
        query = "In the uploaded paper, what methodology was used?"
        assert any(s in query.lower() for s in self.doc_signals)

    def test_summarize_signals(self):
        query = "Summarize the document for me"
        assert any(s in query.lower() for s in self.summarize_signals)

    def test_general_no_signals(self):
        query = "Hello, how are you?"
        all_signals = self.web_signals + self.doc_signals + self.summarize_signals
        assert not any(s in query.lower() for s in all_signals)


# ---------------------------------------------------------------------------
# 3. Document chunking configuration
# ---------------------------------------------------------------------------

class TestDocumentChunking:
    def test_chunk_size_respected(self):
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " "],
        )
        # 5000-char text should split into multiple chunks
        text = "This is a sentence about artificial intelligence. " * 100
        chunks = splitter.split_text(text)
        assert len(chunks) > 1

    def test_chunks_within_size_limit(self):
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " "],
        )
        text = "Hello world. " * 300
        chunks = splitter.split_text(text)
        # Allow small overshoot due to separator handling
        for chunk in chunks:
            assert len(chunk) <= 1100

    def test_overlap_creates_context_continuity(self):
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            separators=[" "],
        )
        text = " ".join([f"word{i}" for i in range(50)])
        chunks = splitter.split_text(text)
        assert len(chunks) >= 2
        # Verify overlap: last words of chunk N appear at start of chunk N+1
        last_words = set(chunks[0].split()[-3:])
        first_words = set(chunks[1].split()[:5])
        assert last_words & first_words  # intersection is non-empty


# ---------------------------------------------------------------------------
# 4. API schema validation
# ---------------------------------------------------------------------------

class TestAPISchemas:
    def test_research_request_defaults(self):
        from api.schemas import ResearchRequest

        req = ResearchRequest(query="test query")
        assert req.query == "test query"
        # session_id should be auto-generated UUID
        assert req.session_id is not None
        assert len(req.session_id) == 36  # UUID format

    def test_research_request_custom_session(self):
        from api.schemas import ResearchRequest

        sid = str(uuid.uuid4())
        req = ResearchRequest(query="hello", session_id=sid)
        assert req.session_id == sid

    def test_research_response_structure(self):
        from api.schemas import ResearchResponse, SourceItem

        resp = ResearchResponse(
            response="Here is the research…",
            sources=[SourceItem(title="Example", url="https://example.com", relevance_score=0.9)],
            intent="web_research",
            session_id=str(uuid.uuid4()),
        )
        assert resp.intent == "web_research"
        assert len(resp.sources) == 1
        assert resp.sources[0].title == "Example"

    def test_upload_response_structure(self):
        from api.schemas import UploadResponse

        resp = UploadResponse(
            message="Successfully processed 'doc.pdf'",
            filename="doc.pdf",
            chunks_stored=42,
            session_id=str(uuid.uuid4()),
        )
        assert resp.chunks_stored == 42

    def test_research_request_rejects_empty_query(self):
        from pydantic import ValidationError
        from api.schemas import ResearchRequest

        with pytest.raises(ValidationError):
            ResearchRequest(query="")


# ---------------------------------------------------------------------------
# 5. Routing logic correctness
# ---------------------------------------------------------------------------

class TestRoutingLogic:
    """Tests the route_by_intent function without invoking the LLM."""

    def test_all_intents_are_routed(self):
        from agents.supervisor import route_by_intent
        from core.state import AgentState

        expected = {
            "web_research": "web_researcher",
            "document_qa": "doc_analyst",
            "summarize": "summarizer",
            "general": "general_response",
        }
        for intent, expected_node in expected.items():
            state: AgentState = {"intent": intent}
            assert route_by_intent(state) == expected_node

    def test_unknown_intent_falls_back_to_general(self):
        from agents.supervisor import route_by_intent
        from core.state import AgentState

        state: AgentState = {"intent": "unknown_intent_xyz"}
        assert route_by_intent(state) == "general_response"

    def test_missing_intent_falls_back_to_general(self):
        from agents.supervisor import route_by_intent
        from core.state import AgentState

        state: AgentState = {}  # no intent key
        assert route_by_intent(state) == "general_response"


# ---------------------------------------------------------------------------
# 6. VectorStoreManager unit tests (mocked embeddings)
# ---------------------------------------------------------------------------

class TestVectorStoreManager:
    @patch("core.embeddings.OpenAIEmbeddings")
    def test_has_documents_false_for_new_session(self, _mock_emb):
        from core.embeddings import VectorStoreManager

        manager = VectorStoreManager()
        assert manager.has_documents("non-existent-session") is False

    @patch("core.embeddings.OpenAIEmbeddings")
    def test_similarity_search_empty_without_index(self, _mock_emb):
        from core.embeddings import VectorStoreManager

        manager = VectorStoreManager()
        results = manager.similarity_search("no-session", "some query")
        assert results == []
