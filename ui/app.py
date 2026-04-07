"""Streamlit chat UI — connects to the FastAPI backend.

Run with:
    streamlit run ui/app.py
"""
import uuid

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
API_BASE = "http://localhost:8000"
RESEARCH_URL = f"{API_BASE}/research"
UPLOAD_URL = f"{API_BASE}/upload"
HEALTH_URL = f"{API_BASE}/health"

INTENT_LABELS = {
    "web_research": "🌐 Web Research",
    "document_qa": "📄 Document Q&A",
    "summarize": "📝 Summarize",
    "general": "💬 General",
}

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []


# ---------------------------------------------------------------------------
# Sidebar — document upload + session info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🔬 Research Assistant")
    st.caption("Powered by LangGraph + GPT-4o-mini")
    st.divider()

    # Health indicator
    try:
        health = requests.get(HEALTH_URL, timeout=3).json()
        st.success(f"API: {health.get('status', 'unknown')} v{health.get('version', '')}")
    except Exception:
        st.error("API server is not reachable. Start it with:\n`uvicorn api.server:app --reload`")

    st.divider()

    # Document upload
    st.subheader("📂 Upload Documents")
    st.caption("Supported: PDF, TXT, Markdown (max 10 MB each). Select multiple files at once.")

    uploaded_list = st.file_uploader(
        label="Choose files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        key="file_uploader",
        label_visibility="collapsed",
    )

    for uploaded in (uploaded_list or []):
        # Only process each file once per session
        if uploaded.name not in st.session_state.uploaded_files:
            with st.spinner(f"Processing {uploaded.name}…"):
                try:
                    resp = requests.post(
                        UPLOAD_URL,
                        files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                        data={"session_id": st.session_state.session_id},
                        timeout=60,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(
                            f"✅ **{data['filename']}** — {data['chunks_stored']} chunks indexed"
                        )
                        st.session_state.uploaded_files.append(uploaded.name)
                    else:
                        try:
                            detail = resp.json().get("detail", resp.text)
                        except Exception:
                            detail = resp.text
                        st.error(f"❌ Upload failed for **{uploaded.name}**: {detail}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to the API server.")

    if st.session_state.uploaded_files:
        st.divider()
        st.subheader("📋 Indexed Documents")
        for fname in st.session_state.uploaded_files:
            st.markdown(f"- 📄 {fname}")

    st.divider()
    st.caption(f"Session ID\n`{st.session_state.session_id[:8]}…`")

    if st.button("🗑️ New Session", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.uploaded_files = []
        st.rerun()


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.title("Multi-Agent Research Assistant")
st.caption(
    "Ask me to research anything on the web, answer questions about your documents, "
    "or summarize uploaded files."
)

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("intent"):
            st.caption(f"Routed to: {INTENT_LABELS.get(msg['intent'], msg['intent'])}")
        if msg.get("sources"):
            with st.expander("📚 Sources", expanded=False):
                for src in msg["sources"]:
                    title = src.get("title") or "Source"
                    url = src.get("url", "")
                    score = src.get("relevance_score", 0.0)
                    if url:
                        st.markdown(f"- [{title}]({url}) — relevance: {score:.2f}")
                    else:
                        st.markdown(f"- **{title}**")

# Chat input
if prompt := st.chat_input("Ask a question…"):
    # Render user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Call the API
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                resp = requests.post(
                    RESEARCH_URL,
                    json={
                        "query": prompt,
                        "session_id": st.session_state.session_id,
                    },
                    timeout=120,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    answer = data.get("response", "No response received.")
                    intent = data.get("intent", "general")
                    sources = data.get("sources", [])

                    st.markdown(answer)
                    st.caption(f"Routed to: {INTENT_LABELS.get(intent, intent)}")

                    if sources:
                        with st.expander("📚 Sources", expanded=False):
                            for src in sources:
                                title = src.get("title") or "Source"
                                url = src.get("url", "")
                                score = src.get("relevance_score", 0.0)
                                if url:
                                    st.markdown(f"- [{title}]({url}) — relevance: {score:.2f}")
                                else:
                                    st.markdown(f"- **{title}**")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "intent": intent,
                        "sources": sources,
                    })

                else:
                    detail = resp.json().get("detail", resp.text)
                    err_msg = f"Error {resp.status_code}: {detail}"
                    st.error(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})

            except requests.exceptions.ConnectionError:
                err_msg = (
                    "Cannot reach the API server. Make sure it's running:\n\n"
                    "```\nuvicorn api.server:app --reload --port 8000\n```"
                )
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
            except Exception as exc:
                err_msg = f"Unexpected error: {exc}"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
