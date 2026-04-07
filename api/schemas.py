"""Pydantic request/response schemas for the FastAPI API."""
import uuid
from typing import List, Optional

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="The research query")
    session_id: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Session identifier — generated if not provided",
    )


class SourceItem(BaseModel):
    title: str = ""
    url: str = ""
    relevance_score: float = 0.0
    content: str = ""


class ResearchResponse(BaseModel):
    response: str
    sources: List[SourceItem] = []
    intent: str
    session_id: str


class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_stored: int
    session_id: str


class HealthResponse(BaseModel):
    status: str
    version: str
