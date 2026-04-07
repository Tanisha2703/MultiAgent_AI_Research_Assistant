# CLAUDE.md — Multi-Agent AI Research Assistant

> Complete project context, architecture, setup instructions, and development guide.

---

## 📌 What This Project Is

A **production-grade agentic AI system** built with **LangGraph** that intelligently orchestrates multiple specialized AI agents to handle complex research tasks. A central **Supervisor Agent** classifies user intent and dynamically routes queries to the optimal sub-agent — **Web Research**, **Document Analysis (RAG)**, or **Summarization** — delivering accurate, cited, and well-structured responses.

**This is NOT a simple chatbot.** It's a multi-agent orchestration system with:
- Deterministic routing via LangGraph conditional edges
- Full RAG pipeline with FAISS vector store and recursive chunking
- Map-reduce summarization for handling arbitrarily long documents
- Production-ready FastAPI backend with async support
- Clean Streamlit chat UI with document upload

---

## 🛠️ Tech Stack

| Layer               | Technology                          | Why This Choice                                                  |
|---------------------|-------------------------------------|------------------------------------------------------------------|
| **Orchestration**   | LangGraph                           | Explicit state management, conditional routing, visual debugging |
| **LLM Framework**   | LangChain                           | Chains, prompt templates, document loaders, text splitters       |
| **Vector Store**    | FAISS                               | Zero-cost local inference, fast similarity search, easy to persist|
| **Embeddings**      | OpenAI `text-embedding-3-small`     | 1536-dim, best cost/quality ratio for retrieval tasks            |
| **LLM**            | OpenAI `gpt-4o-mini`                | Fast, cheap, strong reasoning — ideal for classification + QA    |
| **Web Search**      | Tavily API                          | Purpose-built for LLM apps, structured results, relevance scores |
| **Backend**         | FastAPI                             | Async-native, auto-generated OpenAPI docs, Pydantic validation   |
| **Frontend**        | Streamlit                           | Rapid prototyping, built-in chat components, file upload support |
| **Document Parsing**| PyPDF, Unstructured                 | PDF/TXT/Markdown support with page-level metadata                |
| **Language**        | Python 3.10+                        | Industry standard for ML/AI engineering                          |

---

## 🏗️ Architecture

### High-Level System Design

```
┌──────────────────────────────────────────────────────────────┐
│                      STREAMLIT UI                            │
│             Chat Interface + Document Upload                 │
└────────────────────────┬─────────────────────────────────────┘
                         │ HTTP (JSON)
┌────────────────────────▼─────────────────────────────────────┐
│                    FastAPI Server                             │
│          POST /research    POST /upload    GET /health        │
└────────────────────────┬─────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────┐
│               SUPERVISOR AGENT (LangGraph)                   │
│                                                              │
│   ┌──────────────┐    ┌────────────────┐                     │
│   │  Classify     │───▶│  Route by      │                    │
│   │  Intent       │    │  Intent        │                    │
│   └──────────────┘    └───┬────┬────┬──┘                    │
│                           │    │    │                         │
│        ┌──────────────────┘    │    └──────────────────┐     │
│        ▼                       ▼                       ▼     │
│  ┌───────────┐         ┌────────────┐          ┌──────────┐ │
│  │ 🌐 Web    │         │ 📄 Document│          │ 📝 Summa-│ │
│  │ Research  │         │ Analysis   │          │ rizer    │ │
│  │ Agent     │         │ Agent      │          │ Agent    │ │
│  │           │         │            │          │          │ │
│  │ Tavily    │         │ FAISS +    │          │ Map-     │ │
│  │ Search    │         │ RAG        │          │ Reduce   │ │
│  │ → Rank    │         │ Pipeline   │          │ Pattern  │ │
│  │ → Cite    │         │            │          │          │ │
│  └───────────┘         └────────────┘          └──────────┘ │
│                                                              │
│                    ┌──────────────┐                           │
│                    │ Format       │                           │
│                    │ Response     │                           │
│                    └──────┬───────┘                           │
│                           │                                  │
└───────────────────────────┼──────────────────────────────────┘
                            ▼
                     Final Response
                  (with citations + sources)
```

### LangGraph State Flow

```
START
  │
  ▼
classify_intent ─── LLM classifies query into one of 4 intents
  │
  ├── "web_research"  ──▶ web_researcher_node ──▶ END
  │
  ├── "document_qa"   ──▶ doc_analyst_node    ──▶ END
  │
  ├── "summarize"     ──▶ summarizer_node     ──▶ END
  │
  └── "general"       ──▶ general_response    ──▶ END
```

### RAG Pipeline Detail

```
INGESTION:
  PDF/TXT Upload
      │
      ▼
  Text Extraction (PyPDF / TextLoader)
      │
      ▼
  Recursive Chunking
  ├── chunk_size: 1000 chars
  ├── chunk_overlap: 200 chars
  └── separators: ["\n\n", "\n", ". ", " "]
      │
      ▼
  Embedding (text-embedding-3-small, 1536-dim)
      │
      ▼
  FAISS Index (persisted per session)


RETRIEVAL:
  User Query
      │
      ▼
  Query Embedding
      │
      ▼
  FAISS Similarity Search (top-k=5)
      │
      ▼
  Context Assembly (ranked chunks)
      │
      ▼
  LLM Generation (grounded on retrieved context)
      │
      ▼
  Cited Answer
```

---

## 📁 Project Structure

```
multi-agent-research-assistant/
│
├── README.md                    # Project overview & quick start
├── CLAUDE.md                    # ← You are here (full project context)
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variable template
├── .gitignore                   # Git ignore rules
├── config.py                    # Centralized configuration
│
├── agents/                      # Agent definitions
│   ├── __init__.py
│   ├── supervisor.py            # LangGraph graph builder + orchestrator
│   ├── web_researcher.py        # Web research via Tavily
│   ├── doc_analyst.py           # RAG-based document Q&A
│   └── summarizer.py            # Map-reduce summarization
│
├── core/                        # Core utilities
│   ├── __init__.py
│   ├── state.py                 # AgentState TypedDict (shared state schema)
│   ├── prompts.py               # All prompt templates (centralized)
│   ├── embeddings.py            # FAISS vector store manager
│   └── document_loader.py       # Document ingestion & chunking pipeline
│
├── api/                         # FastAPI backend
│   ├── __init__.py
│   ├── server.py                # FastAPI app with CORS & startup checks
│   ├── routes.py                # /research, /upload, /health endpoints
│   └── schemas.py               # Pydantic request/response models
│
├── ui/                          # Streamlit frontend
│   └── app.py                   # Chat UI with sidebar document upload
│
├── docs/                        # Documentation
│   └── ARCHITECTURE.md          # Detailed architecture deep dive
│
└── tests/                       # Test suite
    ├── __init__.py
    └── test_agents.py           # Unit tests with mocked LLM calls
```

---

## 🚀 Setup & Run Instructions

### Prerequisites

- Python 3.10+
- OpenAI API key ([platform.openai.com](https://platform.openai.com))
- Tavily API key ([tavily.com](https://tavily.com)) — free tier: 1000 requests/month

### Step 1 — Clone & Install

```bash
git clone https://github.com/Tanisha2703/multi-agent-research-assistant.git
cd multi-agent-research-assistant

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Step 2 — Configure Environment

```bash
cp .env.example .env
# Edit .env and add your API keys:
#   OPENAI_API_KEY=sk-...
#   TAVILY_API_KEY=tvly-...
```

### Step 3 — Start the API Server

```bash
uvicorn api.server:app --reload --port 8000
```

Swagger docs available at: `http://localhost:8000/docs`

### Step 4 — Start the Streamlit UI (separate terminal)

```bash
streamlit run ui/app.py
```

Open `http://localhost:8501` in your browser.

---

## 🔑 Key Architecture Decisions & Rationale

### Why LangGraph over vanilla LangChain Agents?

| Criteria             | LangChain Agents         | LangGraph                      |
|----------------------|--------------------------|--------------------------------|
| Routing              | LLM decides tool calls   | Explicit conditional edges     |
| State Management     | Implicit via memory      | Typed `AgentState` dict        |
| Error Handling       | Global try/catch         | Per-node failure isolation     |
| Debugging            | Opaque chain of calls    | Visual graph + state snapshots |
| Determinism          | Non-deterministic        | Deterministic routing paths    |

**Decision:** LangGraph gives us full control over the execution flow. The supervisor classifies intent once, then routes deterministically — no wasted LLM calls on tool selection.

### Why FAISS over ChromaDB or Pinecone?

- **Zero infrastructure cost** — runs locally, no external service dependency
- **Blazing fast** — optimized C++ backend for similarity search
- **Persistence** — serialize to disk with `save_local()` / `load_local()`
- **Sufficient scale** — handles up to ~1M vectors without sharding
- **Tradeoff accepted:** No metadata filtering (we handle this in application logic)

### Why Recursive Chunking over Fixed-Size?

Fixed-size chunking (e.g., split every 500 chars) breaks mid-sentence and mid-paragraph, producing incoherent chunks that hurt retrieval quality. RecursiveCharacterTextSplitter tries larger separators first:

```
"\n\n" (paragraph) → "\n" (line) → ". " (sentence) → " " (word)
```

This ensures each chunk is as semantically coherent as possible.

### Why Map-Reduce for Summarization?

- **Scales to any document length** — each chunk fits in the context window
- **Parallelizable** — map phase chunks can be processed concurrently
- **Quality** — reduce phase removes redundancy and creates coherent output
- **Alternative considered:** Stuff method (concat all chunks) — fails for long documents exceeding the context window

---

## 🔌 API Reference

### POST `/research`

Submit a research query to the multi-agent system.

**Request:**
```json
{
  "query": "What are the latest advances in RAG systems?",
  "session_id": "optional-uuid"
}
```

**Response:**
```json
{
  "response": "Based on recent research...",
  "sources": [
    { "title": "...", "url": "...", "relevance_score": 0.95 }
  ],
  "intent": "web_research",
  "session_id": "generated-or-provided-uuid"
}
```

### POST `/upload`

Upload a document (PDF/TXT/MD) for analysis. Multipart form data.

**Response:**
```json
{
  "message": "Successfully processed 'paper.pdf'",
  "filename": "paper.pdf",
  "chunks_stored": 47,
  "session_id": "uuid"
}
```

### GET `/health`

```json
{ "status": "healthy", "version": "1.0.0" }
```

---

## 🧠 Agent Deep Dive

### 1. Supervisor Agent (`agents/supervisor.py`)

The brain of the system. Responsibilities:

1. **Intent Classification** — Uses a structured-output LLM call to classify the query into `web_research`, `document_qa`, `summarize`, or `general`
2. **Routing** — Conditional edge function maps intent → agent node
3. **State Initialization** — Creates the `AgentState` with query, session_id, and conversation history
4. **Error Recovery** — If any agent fails, returns a graceful error message

Intent classification signals:
- `"latest"`, `"recent"`, `"current"`, `"2025"` → web_research
- `"document"`, `"uploaded"`, `"file"`, `"the paper"` → document_qa
- `"summarize"`, `"key points"`, `"TL;DR"` → summarize
- Greetings, ambiguous → general

### 2. Web Research Agent (`agents/web_researcher.py`)

Pipeline:
1. Receives query from supervisor
2. Calls Tavily Search API (max 5 results)
3. Formats results with title, URL, content, relevance score
4. Sends formatted results + query to LLM for synthesis
5. Returns structured research report with inline citations

### 3. Document Analysis Agent (`agents/doc_analyst.py`)

Full RAG pipeline:
1. Receives query + session_id from supervisor
2. Loads the session's FAISS index via `VectorStoreManager`
3. Performs similarity search (top-k=5)
4. Assembles context from ranked chunks (includes source file + page metadata)
5. Sends context + query to LLM with strict grounding prompt
6. Returns answer with chunk-level citations

### 4. Summarizer Agent (`agents/summarizer.py`)

Map-reduce pattern:
1. Retrieves up to 20 chunks from FAISS (broader retrieval for coverage)
2. **Map Phase**: Each chunk is independently summarized (3-5 sentences)
3. **Reduce Phase**: All chunk summaries are combined into a final summary
4. Extracts 3-5 bullet-point **Key Insights**
5. For small documents (≤3 chunks), uses single-pass summarization instead

---

## 📐 Code Conventions

- **Prompts are centralized** in `core/prompts.py` — never hardcode prompts in agent files
- **Agents follow a uniform interface**: `def agent_node(state: AgentState) -> dict`
- **State mutations are explicit** — agents return only the keys they modify
- **Type hints everywhere** — Pydantic for API, TypedDict for state
- **Logging via `logging` module** — no `print()` statements
- **Configuration via `config.py`** — reads from `.env`, single source of truth
- **Vector store isolation** — each session gets its own FAISS index (no cross-session leakage)

---

## 🧪 Testing

```bash
pytest tests/ -v
```

Tests use mocked LLM responses to avoid API costs. Coverage includes:
- Intent classification mapping
- AgentState schema validation
- Document chunking configuration
- API schema validation (Pydantic)
- Routing logic correctness

Set `OPENAI_API_KEY=test` for unit tests — no real API calls are made.

---

## ⚠️ Known Limitations & Future Improvements

| Current Limitation                    | Planned Improvement                              |
|---------------------------------------|--------------------------------------------------|
| Single-turn agent execution           | Multi-turn agent chains (agent A → agent B)      |
| No persistent conversation memory     | Redis/SQLite backed chat history                  |
| Synchronous LLM calls in map phase    | Async batch processing for map-reduce            |
| No authentication on API              | JWT/API key auth middleware                       |
| FAISS only (no metadata filtering)    | Hybrid search with ChromaDB or Qdrant            |
| Streamlit UI (prototype)              | React/Next.js production frontend                |

---

## 🔒 Security Considerations

- API keys loaded from `.env` via `python-dotenv` — **never committed to git**
- File uploads validated for type (`.pdf`, `.txt`, `.md` only) and size (<10MB)
- Session IDs are UUIDs — no sequential/guessable identifiers
- FAISS indices stored locally — **no document data leaves the server** except via LLM API calls
- CORS configured for development — restrict `allow_origins` in production

---

## 📚 References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [Tavily API Docs](https://docs.tavily.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

## 👤 Author

**Tanisha Kamboj**
- GitHub: [Tanisha2703](https://github.com/Tanisha2703)
- LinkedIn: [Tanisha](https://linkedin.com/in/tanisha)
- Email: tanishakamboj72@gmail.com
