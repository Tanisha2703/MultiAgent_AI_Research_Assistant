"""FastAPI route handlers — /research, /upload, /health."""
import logging
import os
import tempfile
import uuid

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional

from agents.supervisor import run as run_graph
from api.schemas import HealthResponse, ResearchRequest, ResearchResponse, UploadResponse
from config import settings
from core.document_loader import load_and_chunk
from core.embeddings import vector_store_manager

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_BYTES = settings.MAX_FILE_SIZE_MB * 1024 * 1024


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    """Returns server health status."""
    return HealthResponse(status="healthy", version=settings.APP_VERSION)


# ---------------------------------------------------------------------------
# Research query
# ---------------------------------------------------------------------------

@router.post("/research", response_model=ResearchResponse, tags=["Research"])
async def research(request: ResearchRequest) -> ResearchResponse:
    """Submit a research query to the multi-agent system.

    The supervisor agent classifies the intent and routes to the
    appropriate specialist: web research, document Q&A, or summarization.
    """
    logger.info("POST /research session=%s query=%.80r", request.session_id, request.query)

    result = run_graph(query=request.query, session_id=request.session_id)

    sources_out = [
        {
            "title": s.get("title", ""),
            "url": s.get("url", ""),
            "relevance_score": s.get("relevance_score", 0.0),
            "content": s.get("content", ""),
        }
        for s in (result.get("sources") or [])
    ]

    return ResearchResponse(
        response=result.get("response", ""),
        sources=sources_out,
        intent=result.get("intent", "general"),
        session_id=result.get("session_id", request.session_id),
    )


# ---------------------------------------------------------------------------
# Document upload
# ---------------------------------------------------------------------------

@router.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(default=None),
) -> UploadResponse:
    """Upload a document (PDF / TXT / MD) for subsequent analysis.

    The document is chunked and indexed into a session-scoped FAISS store.
    File size limit: 10 MB. Allowed types: .pdf, .txt, .md
    """
    effective_session_id = session_id or str(uuid.uuid4())
    filename = file.filename or "uploaded_file"
    logger.info("POST /upload session=%s filename=%s", effective_session_id, filename)

    # Validate extension
    ext = os.path.splitext(filename)[1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}",
        )

    # Read and validate size
    content = await file.read()
    if len(content) > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {settings.MAX_FILE_SIZE_MB} MB limit.",
        )

    # Write to a temp file for the loader (loaders require a file path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        chunks = load_and_chunk(tmp_path, filename)
        chunks_stored = vector_store_manager.add_documents(effective_session_id, chunks)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Document processing failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to process document.")
    finally:
        os.unlink(tmp_path)

    return UploadResponse(
        message=f"Successfully processed '{filename}'",
        filename=filename,
        chunks_stored=chunks_stored,
        session_id=effective_session_id,
    )
