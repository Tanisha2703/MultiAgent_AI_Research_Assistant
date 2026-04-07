"""LLM utilities with OpenAI-to-NVIDIA runtime fallback.

OpenAI is tried first on every call. If it fails for ANY reason
(429 quota, auth error, timeout), NVIDIA is tried automatically.
No magic -- just explicit try/except.
"""
import logging

from langchain_core.messages import HumanMessage

from config import settings

logger = logging.getLogger(__name__)


def _openai_available():
    key = (settings.OPENAI_API_KEY or "").strip()
    return len(key) > 10 and key.startswith("sk-") and key != "sk-..."


def _nvidia_available():
    key = (settings.NVIDIA_API_KEY or "").strip()
    return len(key) > 10 and key.startswith("nvapi-") and key != "nvapi-..."


def _openai_llm(temperature):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        openai_api_key=settings.OPENAI_API_KEY,
        temperature=temperature,
        max_retries=0,
    )


def _nvidia_llm(temperature):
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    return ChatNVIDIA(
        model=settings.NVIDIA_LLM_MODEL,
        nvidia_api_key=settings.NVIDIA_API_KEY,
        temperature=temperature,
    )


def _invoke_with_fallback(messages, temperature):
    if _openai_available():
        try:
            result = _openai_llm(temperature).invoke(messages)
            logger.debug("LLM call succeeded via OpenAI")
            return result
        except Exception as exc:
            if _nvidia_available():
                logger.warning(
                    "OpenAI failed [%s] -- falling back to NVIDIA",
                    str(exc)[:150],
                )
            else:
                raise

    if _nvidia_available():
        result = _nvidia_llm(temperature).invoke(messages)
        logger.debug("LLM call succeeded via NVIDIA")
        return result

    raise RuntimeError(
        "No LLM key configured. Set OPENAI_API_KEY or NVIDIA_API_KEY in .env"
    )


def invoke_llm(prompt, input_dict, temperature=0.2):
    """Invoke a ChatPromptTemplate with automatic fallback."""
    formatted = prompt.invoke(input_dict)
    return _invoke_with_fallback(formatted, temperature)


def invoke_llm_raw(text, temperature=0.2):
    """Invoke LLM with a plain string prompt."""
    messages = [HumanMessage(content=text)]
    return _invoke_with_fallback(messages, temperature)


def active_provider():
    openai_ok = _openai_available()
    nvidia_ok = _nvidia_available()
    if openai_ok and nvidia_ok:
        return f"OpenAI ({settings.LLM_MODEL}) + NVIDIA fallback ({settings.NVIDIA_LLM_MODEL})"
    if openai_ok:
        return f"OpenAI ({settings.LLM_MODEL})"
    if nvidia_ok:
        return f"NVIDIA ({settings.NVIDIA_LLM_MODEL})"
    return "none"


def get_embeddings():
    """Return embeddings model. Tries OpenAI first, then NVIDIA."""
    if _openai_available():
        try:
            from langchain_openai import OpenAIEmbeddings
            logger.info("Embeddings: OpenAI (%s)", settings.EMBEDDING_MODEL)
            return OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                openai_api_key=settings.OPENAI_API_KEY,
            )
        except Exception as exc:
            logger.warning("OpenAI embeddings failed (%s) -- trying NVIDIA", exc)

    if _nvidia_available():
        from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
        logger.info("Embeddings: NVIDIA (%s)", settings.NVIDIA_EMBEDDING_MODEL)
        return NVIDIAEmbeddings(
            model=settings.NVIDIA_EMBEDDING_MODEL,
            nvidia_api_key=settings.NVIDIA_API_KEY,
        )

    raise RuntimeError(
        "No embeddings key configured. Set OPENAI_API_KEY or NVIDIA_API_KEY in .env"
    )
