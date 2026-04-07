# Architecture Deep Dive

## System Overview

The Multi-Agent Research Assistant uses a **supervisor pattern** implemented with LangGraph to coordinate specialist agents. The key insight is separating *routing intelligence* (supervisor) from *task execution* (agents).

---

## LangGraph State Machine

```
START
  │
  ▼
classify_intent          ← LLM call (gpt-4o-mini, temp=0.0)
  │                         Returns one of: web_research | document_qa | summarize | general
  │
  ├── web_research  ──▶  web_researcher_node   ──▶  END
  │                        Tavily(5 results) → synthesis LLM
  │
  ├── document_qa   ──▶  doc_analyst_node      ──▶  END
  │                        FAISS(top-5) → grounded LLM
  │
  ├── summarize     ──▶  summarizer_node       ──▶  END
  │                        FAISS(top-20) → map-reduce
  │
  └── general       ──▶  general_response_node ──▶  END
                           Direct LLM response
```

### Why conditional edges?

Conditional edges in LangGraph map a *router function* output → next node. Unlike LangChain `AgentExecutor` where the LLM decides which tool to call each step, here the routing decision is made **once** by the classify_intent node, then the graph takes the deterministic path to the right specialist.

Benefit: **No wasted LLM calls on tool selection loops.**

---

## RAG Pipeline

```
Upload flow:
  File (PDF/TXT/MD)
    → PyPDFLoader / TextLoader
    → RecursiveCharacterTextSplitter (chunk=1000, overlap=200)
    → OpenAIEmbeddings (text-embedding-3-small, 1536-dim)
    → FAISS.from_documents()  ← stored in VectorStoreManager[session_id]

Query flow:
  User query
    → OpenAIEmbeddings (query)
    → FAISS.similarity_search(top-k=5)
    → Assemble context with source/page metadata
    → ChatOpenAI with DOC_ANALYST_PROMPT
    → Grounded answer with citations
```

### Session isolation

Each session gets its own `FAISS` index stored in `VectorStoreManager._stores[session_id]`. This prevents document data from one user leaking into another session's retrieval results.

---

## Map-Reduce Summarization

```
FAISS.similarity_search(top-20 chunks)
  │
  ├── Chunk 1 → SUMMARIZE_MAP_PROMPT → partial summary 1
  ├── Chunk 2 → SUMMARIZE_MAP_PROMPT → partial summary 2
  ├── ...
  └── Chunk N → SUMMARIZE_MAP_PROMPT → partial summary N
                                │
                  SUMMARIZE_REDUCE_PROMPT(all partial summaries)
                                │
                          Final Summary + Key Insights
```

For ≤3 chunks, the map phase is skipped and chunks go directly into the reduce prompt (single-pass).

---

## Data Flow: End-to-End Request

```
Streamlit UI  --POST /research {query, session_id}-->  FastAPI
FastAPI       --run_graph(query, session_id)-------->  LangGraph
LangGraph     --classify_intent------------------->    GPT-4o-mini
LangGraph     --route to specialist---------------->   Agent Node
Agent Node    --(Tavily / FAISS / LLM)------------>   Response
LangGraph     --return final AgentState------------>  FastAPI
FastAPI       --JSON response-------------------->     Streamlit UI
Streamlit UI  --render in st.chat_message--------->   Browser
```

---

## Security Design

| Concern | Mitigation |
|---|---|
| API key exposure | Loaded from `.env`, never in source code; `.env` in `.gitignore` |
| File upload abuse | Extension allowlist (.pdf, .txt, .md); 10 MB size limit |
| Session collisions | UUIDs for session IDs — cryptographically unpredictable |
| Data isolation | Per-session FAISS index — no cross-session retrieval |
| Prompt injection | Grounded prompts instruct LLM to use only retrieved context |
| CORS | Wildcard in dev; restrict `allow_origins` before production deploy |

---

## Configuration Reference

All configuration lives in `config.py` and reads from `.env`:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | required | OpenAI API authentication |
| `TAVILY_API_KEY` | required | Tavily search API |
| `LLM_MODEL` | gpt-4o-mini | LLM for all agent calls |
| `EMBEDDING_MODEL` | text-embedding-3-small | Embedding model for FAISS |
| `CHUNK_SIZE` | 1000 | Character chunk size for splitting |
| `CHUNK_OVERLAP` | 200 | Overlap between consecutive chunks |
| `TOP_K_RETRIEVAL` | 5 | Number of chunks retrieved for doc Q&A |
| `SUMMARIZE_TOP_K` | 20 | Number of chunks retrieved for summarization |
| `MAX_FILE_SIZE_MB` | 10 | Maximum upload file size |
