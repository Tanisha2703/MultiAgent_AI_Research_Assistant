"""Centralized configuration — reads from .env, single source of truth."""
import os
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    # ----- API Keys -----
    # OpenAI is preferred. Set NVIDIA_API_KEY as a free-tier fallback.
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API key (preferred)")
    TAVILY_API_KEY: str = Field(..., description="Tavily search API key")

    # NVIDIA free-tier — get key at https://build.nvidia.com/
    NVIDIA_API_KEY: Optional[str] = Field(default=None, description="NVIDIA NIM API key (free tier fallback)")

    # ----- OpenAI model selection -----
    LLM_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # ----- NVIDIA fallback model selection -----
    # Free tier models: https://build.nvidia.com/explore/discover
    NVIDIA_LLM_MODEL: str = "meta/llama-3.3-70b-instruct"
    NVIDIA_EMBEDDING_MODEL: str = "nvidia/nv-embedqa-e5-v5"

    # ----- RAG / chunking -----
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RETRIEVAL: int = 5
    SUMMARIZE_TOP_K: int = 20          # broader retrieval for summarization

    # ----- File upload constraints -----
    MAX_FILE_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: tuple = (".pdf", ".txt", ".md")

    # ----- App metadata -----
    APP_VERSION: str = "1.0.0"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
