"""FAISS vector store manager — per-session index isolation with disk persistence."""
import json
import logging
import os
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from core.llm import _openai_available, _nvidia_available

logger = logging.getLogger(__name__)

# Persist FAISS indexes here so server restarts (--reload) don't lose data
_PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "faiss_stores")


class VectorStoreManager:
    """Manages one FAISS index per session_id with disk persistence."""

    def __init__(self) -> None:
        self._stores: dict[str, FAISS] = {}
        self._providers: dict[str, str] = {}   # session_id → "openai" | "nvidia"
        os.makedirs(_PERSIST_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _session_path(self, session_id: str) -> str:
        return os.path.join(_PERSIST_DIR, session_id)

    def _make_embeddings(self, provider: str):
        from config import settings
        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                openai_api_key=settings.OPENAI_API_KEY,
                max_retries=0,
            )
        from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
        return NVIDIAEmbeddings(
            model=settings.NVIDIA_EMBEDDING_MODEL,
            nvidia_api_key=settings.NVIDIA_API_KEY,
        )

    def _save(self, session_id: str) -> None:
        path = self._session_path(session_id)
        os.makedirs(path, exist_ok=True)
        self._stores[session_id].save_local(path)
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump({"provider": self._providers[session_id]}, f)
        logger.debug("session=%s persisted to disk", session_id)

    def _try_load(self, session_id: str) -> bool:
        """Load from disk into memory if not already loaded. Returns True if available."""
        if session_id in self._stores:
            return True
        path = self._session_path(session_id)
        meta_file = os.path.join(path, "meta.json")
        if not os.path.exists(meta_file):
            return False
        try:
            with open(meta_file) as f:
                meta = json.load(f)
            provider = meta.get("provider", "nvidia")
            emb = self._make_embeddings(provider)
            store = FAISS.load_local(path, emb, allow_dangerous_deserialization=True)
            self._stores[session_id] = store
            self._providers[session_id] = provider
            logger.info("session=%s restored from disk (provider=%s)", session_id, provider)
            return True
        except Exception as exc:
            logger.warning("session=%s failed to load from disk: %s", session_id, exc)
            return False

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_documents(self, session_id: str, documents: List[Document]) -> int:
        """Add pre-chunked documents to the session's FAISS index.

        Creates a new index if none exists yet for this session.
        Persists to disk so the index survives server restarts.
        Returns the number of chunks stored.
        """
        if not documents:
            return 0

        # Restore from disk if server restarted and wiped memory
        self._try_load(session_id)

        if session_id in self._stores:
            # Append to existing index — reuses the same embeddings object it was built with
            self._stores[session_id].add_documents(documents)
            self._save(session_id)
            logger.info(
                "session=%s appended %d chunks (provider=%s)",
                session_id, len(documents), self._providers.get(session_id),
            )
            return len(documents)

        # New session — try OpenAI first, fall back directly to NVIDIA on any error
        store = None
        provider = None

        if _openai_available():
            try:
                emb = self._make_embeddings("openai")
                store = FAISS.from_documents(documents, emb)
                provider = "openai"
                logger.info("session=%s created with OpenAI embeddings", session_id)
            except Exception as exc:
                logger.warning(
                    "OpenAI embeddings failed [%s] -- falling back to NVIDIA",
                    str(exc)[:150],
                )

        if store is None and _nvidia_available():
            emb = self._make_embeddings("nvidia")
            store = FAISS.from_documents(documents, emb)
            provider = "nvidia"
            logger.info("session=%s created with NVIDIA embeddings", session_id)

        if store is None:
            raise RuntimeError(
                "No embeddings provider available. "
                "Set OPENAI_API_KEY or NVIDIA_API_KEY in .env"
            )

        self._stores[session_id] = store
        self._providers[session_id] = provider
        self._save(session_id)
        logger.info("session=%s stored %d chunks", session_id, len(documents))
        return len(documents)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        session_id: str,
        query: str,
        k: int = 5,
    ) -> List[Document]:
        """Perform a cosine-similarity search against the session's index."""
        self._try_load(session_id)
        store = self._stores.get(session_id)
        if store is None:
            logger.warning("session=%s has no FAISS index; returning empty list", session_id)
            return []
        results = store.similarity_search(query, k=k)
        logger.debug("session=%s retrieved %d chunks for query=%.60r", session_id, len(results), query)
        return results

    def has_documents(self, session_id: str) -> bool:
        """Return True if the session has at least one document indexed."""
        if session_id in self._stores:
            return True
        return self._try_load(session_id)

    def get_store(self, session_id: str) -> Optional[FAISS]:
        self._try_load(session_id)
        return self._stores.get(session_id)


# Module-level singleton — shared across the FastAPI app lifetime
vector_store_manager = VectorStoreManager()
