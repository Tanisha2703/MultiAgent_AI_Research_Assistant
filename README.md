<div align="center">

# 🔬 Multi-Agent AI Research Assistant

### *An intelligent, autonomous research system powered by LangGraph*

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-FF6B35?style=for-the-badge&logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

*Ask anything. Upload documents. Get answers — powered by a swarm of specialized AI agents.*

</div>

---

## ✨ What It Does

This is a **production-grade multi-agent system** that routes your queries to the right specialist automatically:

| You Ask | Agent Triggered | What Happens |
|---|---|---|
| *"What is the latest on AI regulation?"* | 🌐 Web Researcher | Live Tavily search → cited answer |
| *"Summarize this 50-page PDF"* | 📝 Summarizer | Map-reduce across all chunks |
| *"What does section 3 of my contract say?"* | 📄 Doc Analyst | RAG retrieval → grounded answer |
| *"Hello / what can you do?"* | 💬 General | Direct LLM response |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│           Supervisor Agent              │
│   (LLM-based intent classification)    │
└──────┬──────────┬──────────┬────────────┘
       │          │          │          │
       ▼          ▼          ▼          ▼
  🌐 Web      📄 Doc      📝 Summ-   💬 General
  Researcher  Analyst     arizer     Response
  (Tavily)    (FAISS      (Map-      (Direct
              RAG)        Reduce)     LLM)
       │          │          │          │
       └──────────┴──────────┴──────────┘
                        │
                        ▼
                 Structured Response
              (answer + sources + intent)
```

**LangGraph** manages the state machine — each agent is a node with typed edges, enabling clean conditional routing, error recovery, and future extensibility.

---

## 🚀 Key Features

- **🧠 Intelligent Routing** — Supervisor LLM classifies intent (`web_research`, `document_qa`, `summarize`, `general`) and routes to the right agent every time
- **🌐 Real-Time Web Search** — Web Researcher uses Tavily to fetch live results with titles, URLs, and relevance scores
- **📄 RAG Document Pipeline** — Upload PDFs, TXT, or Markdown; FAISS indexes chunks for semantic retrieval; Doc Analyst answers with grounded citations
- **📝 Long-Doc Summarization** — Map-reduce pattern: parallel chunk summaries → single coherent final summary, no token-limit issues
- **💾 Persistent FAISS Indexes** — Vector stores saved to disk; document context survives server restarts
- **🔄 Dual LLM Provider Fallback** — OpenAI GPT-4o-mini as primary, NVIDIA NIM (Llama 3.3 70B, free tier) as automatic fallback — zero disruption on quota errors
- **🔌 Clean REST API** — FastAPI with full OpenAPI/Swagger docs at `/docs`
- **🖥️ Chat UI** — Streamlit interface with multi-document upload, session management, source citations
- **📦 Multi-Document Support** — Upload multiple files in a single session; all are indexed together in one FAISS store

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Agent Orchestration** | [LangGraph](https://github.com/langchain-ai/langgraph) — StateGraph with conditional edges |
| **Primary LLM** | OpenAI `gpt-4o-mini` |
| **Fallback LLM** | NVIDIA NIM `meta/llama-3.3-70b-instruct` (free tier) |
| **Primary Embeddings** | OpenAI `text-embedding-3-small` |
| **Fallback Embeddings** | NVIDIA NIM `nvidia/nv-embedqa-e5-v5` (free tier) |
| **Vector Store** | FAISS (persisted to disk, per-session isolation) |
| **Web Search** | Tavily Search API |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | Streamlit |
| **Config** | Pydantic Settings + `.env` |

---

## ⚡ Quick Start

### 1. Clone & install

```bash
git clone https://github.com/Tanisha2703/multi-agent-ai-research-assistant.git
cd multi-agent-ai-research-assistant

python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Required
TAVILY_API_KEY=tvly-...           # https://app.tavily.com/

# At least ONE of the following (NVIDIA is free):
OPENAI_API_KEY=sk-...             # https://platform.openai.com/
NVIDIA_API_KEY=nvapi-...          # https://build.nvidia.com/  ← FREE tier
```

> **No OpenAI credits?** Just set `NVIDIA_API_KEY` — all LLM and embedding calls automatically use NVIDIA's free tier.

### 3. Run the API server

```bash
uvicorn api.server:app --reload --port 8000
```

📖 Swagger UI → [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Run the Streamlit UI (new terminal)

```bash
streamlit run ui/app.py
```

🖥️ Open → [http://localhost:8501](http://localhost:8501)

---

## 📁 Project Structure

```
multi-agent-ai-research-assistant/
├── agents/
│   ├── supervisor.py        # LangGraph orchestrator + intent router
│   ├── web_researcher.py    # Tavily search agent
│   ├── doc_analyst.py       # RAG agent (FAISS retrieval)
│   └── summarizer.py        # Map-reduce summarizer
├── api/
│   ├── server.py            # FastAPI app + startup checks
│   ├── routes.py            # /research, /upload, /health endpoints
│   └── schemas.py           # Pydantic request/response models
├── core/
│   ├── llm.py               # LLM factory + OpenAI→NVIDIA fallback logic
│   ├── embeddings.py        # FAISS manager with disk persistence
│   ├── document_loader.py   # PDF/TXT/MD loader + chunker
│   ├── prompts.py           # All prompt templates
│   └── state.py             # LangGraph AgentState TypedDict
├── ui/
│   └── app.py               # Streamlit chat interface
├── tests/                   # pytest test suite
├── data/faiss_stores/        # Persisted FAISS indexes (git-ignored)
├── config.py                # Pydantic settings (reads .env)
├── requirements.txt
└── .env.example
```

---

## 🔌 API Reference

### `POST /research`
Submit a query — the supervisor classifies intent and routes to the right agent.

```json
{
  "query": "Summarize the uploaded contract",
  "session_id": "abc-123"
}
```

**Response:**
```json
{
  "response": "The contract covers...",
  "intent": "summarize",
  "sources": [{"title": "contract.pdf", "url": "", "relevance_score": 1.0}],
  "session_id": "abc-123"
}
```

### `POST /upload`
Upload a PDF, TXT, or Markdown file (max 10 MB). Chunks and indexes it into the session's FAISS store.

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@report.pdf" \
  -F "session_id=abc-123"
```

### `GET /health`
Returns `{"status": "healthy", "version": "1.0.0"}`.

Full interactive docs → [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🤖 Agent Deep Dive

### Supervisor
Uses an LLM prompt with strict output parsing to classify every query into one of four intents. Falls back to `general` if classification fails. Assigns a session UUID if none provided.

### Web Researcher
Calls Tavily's search API, formats the top results, and feeds them to the LLM to produce a cited, structured answer. Returns source URLs and relevance scores.

### Document Analyst
Performs cosine-similarity search in the session's FAISS index, assembles a numbered context window from the top-K chunks, and generates a grounded answer with chunk citations.

### Summarizer
Implements map-reduce: first summarizes each chunk independently in parallel (map), then reduces all summaries into a single coherent output (reduce). Handles documents of arbitrary length.

---

## 🔁 Provider Fallback

The system has a **zero-downtime fallback chain**:

```
LLM Call
  │
  ├─ OpenAI available? ──→ Try GPT-4o-mini
  │                           │
  │                     429/Error? ──→ NVIDIA NIM (Llama 3.3 70B)
  │
  └─ OpenAI not configured? ──→ NVIDIA NIM directly

Embeddings
  │
  ├─ Try OpenAI text-embedding-3-small
  │     │
  │   Fail? ──→ NVIDIA NIM nv-embedqa-e5-v5
  │
  └─ OpenAI not configured? ──→ NVIDIA NIM directly
```

NVIDIA's free tier requires no billing setup — just [grab a free key](https://build.nvidia.com/).

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit changes: `git commit -m "feat: add my feature"`
4. Push and open a Pull Request

---

## 👩‍💻 Author

**Tanisha Kamboj**

[![GitHub](https://img.shields.io/badge/GitHub-Tanisha2703-181717?style=flat-square&logo=github)](https://github.com/Tanisha2703)

---

<div align="center">

*Built with ❤️ using LangGraph, FastAPI, and Streamlit*

⭐ **Star this repo if you found it useful!** ⭐

</div>
