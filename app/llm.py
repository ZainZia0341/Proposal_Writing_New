from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_llm():
    if settings.llm_provider == "groq" and settings.groq_api_key:
        from langchain_groq import ChatGroq

        return ChatGroq(
            temperature=0.2,
            model_name="openai/gpt-oss-120b",
            api_key=settings.groq_api_key,
            timeout=12,
            max_retries=0,
        )

    if settings.google_api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0.2,
            google_api_key=settings.google_api_key,
            timeout=12,
            max_retries=0,
        )

    return None


def llm_available() -> bool:
    return get_llm() is not None


def invoke_text(system_prompt: str, user_prompt: str, fallback: str | None = None) -> str:
    model = get_llm()
    if model is None:
        if fallback is not None:
            return fallback
        raise RuntimeError("No LLM is configured. Add GROQ_API_KEY or GOOGLE_API_KEY.")

    try:
        response = model.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        return str(response.content).strip()
    except Exception as exc:  # pragma: no cover - network/provider failures
        logger.warning("LLM invocation failed, using fallback when available: %s", exc)
        if fallback is not None:
            return fallback
        raise


def invoke_json(system_prompt: str, user_prompt: str, fallback: dict[str, Any]) -> dict[str, Any]:
    fallback_text = json.dumps(fallback)
    raw_response = invoke_text(system_prompt, user_prompt, fallback=fallback_text)

    try:
        return json.loads(raw_response)
    except json.JSONDecodeError:
        start = raw_response.find("{")
        end = raw_response.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw_response[start : end + 1])
            except json.JSONDecodeError:
                pass
        return fallback
