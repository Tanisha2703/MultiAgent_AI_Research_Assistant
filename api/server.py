"""FastAPI application — startup validation, CORS, and route registration."""
import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from config import settings

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Multi-Agent Research Assistant",
    description=(
        "A production-grade agentic AI system that orchestrates multiple specialized agents "
        "for web research, document analysis (RAG), and summarization."
    ),
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — restrict allow_origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this for production deployments
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


# ---------------------------------------------------------------------------
# Startup event — validate API keys are loaded
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_checks() -> None:
    """Fail fast if no usable LLM key is available."""
    from core.llm import _openai_available, _nvidia_available, active_provider

    openai_ok = _openai_available()
    nvidia_ok = _nvidia_available()

    if openai_ok:
        logger.info("LLM provider: OpenAI (%s)", settings.LLM_MODEL)
    elif nvidia_ok:
        logger.warning(
            "OPENAI_API_KEY not set — using NVIDIA free-tier fallback (%s). "
            "Set OPENAI_API_KEY for best results.",
            settings.NVIDIA_LLM_MODEL,
        )
    else:
        logger.error(
            "No LLM API key found. Set OPENAI_API_KEY (preferred) or "
            "NVIDIA_API_KEY (free tier: https://build.nvidia.com/) in your .env file."
        )
        sys.exit(1)

    if not settings.TAVILY_API_KEY or settings.TAVILY_API_KEY == "tvly-...":
        logger.warning(
            "TAVILY_API_KEY is not set. Web research queries will fail. "
            "Get a free key at https://tavily.com"
        )

    logger.info(
        "Multi-Agent Research Assistant v%s started. Active provider: %s. "
        "Swagger: http://localhost:%d/docs",
        settings.APP_VERSION,
        active_provider(),
        settings.API_PORT,
    )
