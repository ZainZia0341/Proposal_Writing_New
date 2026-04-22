from __future__ import annotations

import json
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .config import settings
from .logging_utils import get_logger

logger = get_logger(__name__)

_model_usage: ContextVar[tuple[str, ...]] = ContextVar("llm_model_usage", default=())
_provider_override: ContextVar[str | None] = ContextVar("llm_provider_override", default=None)
_google_model_override: ContextVar[str | None] = ContextVar("llm_google_model_override", default=None)


def _debug_print(title: str, payload: Any | None = None) -> None:
    print(f"\n============= {title} =============", flush=True)
    if payload is not None:
        if isinstance(payload, str):
            print(payload, flush=True)
        else:
            try:
                print(json.dumps(payload, indent=2, ensure_ascii=False, default=str), flush=True)
            except TypeError:
                print(str(payload), flush=True)
    print("========================================\n", flush=True)


@dataclass(frozen=True)
class ResolvedLLM:
    provider: str
    model_name: str
    client: Any

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model_name}"


@lru_cache(maxsize=1)
def _get_groq_chat_model() -> ResolvedLLM | None:
    if not settings.groq_api_key:
        return None

    from langchain_groq import ChatGroq

    logger.info("Configured Groq chat model", extra={"provider": "groq", "model_name": settings.groq_model_name})
    return ResolvedLLM(
        provider="groq",
        model_name=settings.groq_model_name,
        client=ChatGroq(
            temperature=0.2,
            model_name=settings.groq_model_name,
            api_key=settings.groq_api_key,
            timeout=12,
            max_retries=0,
        ),
    )


@lru_cache(maxsize=8)
def _get_google_chat_model(model_name: str) -> ResolvedLLM | None:
    if not settings.google_api_key:
        return None

    from langchain_google_genai import ChatGoogleGenerativeAI

    logger.info("Configured Google Gemini chat model", extra={"provider": "google", "model_name": model_name})
    return ResolvedLLM(
        provider="google",
        model_name=model_name,
        client=ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.2,
            google_api_key=settings.google_api_key,
            timeout=12,
            max_retries=0,
        ),
    )


def reset_llm_runtime() -> None:
    _model_usage.set(())
    _provider_override.set(None)
    _google_model_override.set(None)
    groq_cache_clear = getattr(_get_groq_chat_model, "cache_clear", None)
    if callable(groq_cache_clear):
        groq_cache_clear()
    google_cache_clear = getattr(_get_google_chat_model, "cache_clear", None)
    if callable(google_cache_clear):
        google_cache_clear()


def reset_llm_request_state() -> None:
    _model_usage.set(())
    _provider_override.set(None)
    _google_model_override.set(None)


def get_model_used() -> str | None:
    unique_usage: list[str] = []
    for label in _model_usage.get():
        if label not in unique_usage:
            unique_usage.append(label)
    if not unique_usage:
        return None
    return " -> ".join(unique_usage)


def _record_model_usage(model: ResolvedLLM) -> None:
    usage = _model_usage.get()
    _model_usage.set((*usage, model.label))


def _is_groq_rate_limit_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)

    exc_name = exc.__class__.__name__.lower()
    message = str(exc).lower()
    return bool(
        status_code == 429
        or "ratelimit" in exc_name
        or "rate limit" in message
        or "too many requests" in message
        or "429" in message
    )


def _google_candidate_names() -> list[str]:
    preferred_name = _google_model_override.get() or settings.google_model_name
    names: list[str] = [preferred_name]
    for model_name in settings.google_fallback_models:
        if model_name not in names:
            names.append(model_name)
    return names


def _invoke_model(model: ResolvedLLM, system_prompt: str, user_prompt: str) -> str:
    _debug_print(
        "starting llm call",
        {
            "provider": model.provider,
            "model_name": model.model_name,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        },
    )
    response = model.client.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    _record_model_usage(model)
    response_text = str(response.content).strip()
    _debug_print(
        "llm response",
        {
            "provider": model.provider,
            "model_name": model.model_name,
            "response_text": response_text,
        },
    )
    return response_text


def _invoke_google_chain(system_prompt: str, user_prompt: str, fallback: str | None = None) -> str:
    last_error: Exception | None = None
    for model_name in _google_candidate_names():
        model = _get_google_chat_model(model_name)
        if model is None:
            continue
        try:
            response_text = _invoke_model(model, system_prompt, user_prompt)
            _provider_override.set("google")
            _google_model_override.set(model_name)
            return response_text
        except Exception as exc:  # pragma: no cover - network/provider failures
            last_error = exc
            _debug_print(
                "llm call failed",
                {
                    "provider": "google",
                    "model_name": model_name,
                    "error": str(exc),
                },
            )
            logger.warning(
                "Google Gemini invocation failed for model %s, trying next candidate when available: %s",
                model_name,
                exc,
            )

    if fallback is not None:
        logger.warning("All Google Gemini candidates failed, using fallback response")
        return fallback
    if last_error is not None:
        raise last_error
    raise RuntimeError("No Google Gemini model is configured. Add GOOGLE_API_KEY.")


def get_llm():
    if _provider_override.get() == "google":
        google_model = _get_google_chat_model(_google_model_override.get() or settings.google_model_name)
        return google_model.client if google_model is not None else None

    if settings.llm_provider == "groq" and settings.groq_api_key:
        groq_model = _get_groq_chat_model()
        return groq_model.client if groq_model is not None else None

    if settings.google_api_key:
        google_model = _get_google_chat_model(settings.google_model_name)
        return google_model.client if google_model is not None else None

    return None


def llm_available() -> bool:
    return bool(settings.groq_api_key or settings.google_api_key)


def invoke_text(system_prompt: str, user_prompt: str, fallback: str | None = None) -> str:
    if _provider_override.get() == "google" and settings.google_api_key:
        return _invoke_google_chain(system_prompt, user_prompt, fallback=fallback)

    if settings.llm_provider == "groq" and settings.groq_api_key:
        groq_model = _get_groq_chat_model()
        if groq_model is not None:
            try:
                return _invoke_model(groq_model, system_prompt, user_prompt)
            except Exception as exc:  # pragma: no cover - network/provider failures
                if _is_groq_rate_limit_error(exc) and settings.google_api_key:
                    _debug_print(
                        "llm provider switch",
                        {
                            "reason": "Groq rate limit detected",
                            "from": groq_model.label,
                            "to_google_candidates": _google_candidate_names(),
                            "error": str(exc),
                        },
                    )
                    logger.warning(
                        "Groq rate limit detected for model %s. Switching this request to Google Gemini fallback models.",
                        groq_model.model_name,
                    )
                    _provider_override.set("google")
                    return _invoke_google_chain(system_prompt, user_prompt, fallback=fallback)
                _debug_print(
                    "llm call failed",
                    {
                        "provider": "groq",
                        "model_name": groq_model.model_name,
                        "error": str(exc),
                        "using_fallback": fallback is not None,
                    },
                )
                logger.warning("LLM invocation failed, using fallback when available: %s", exc)
                if fallback is not None:
                    return fallback
                raise

    if settings.google_api_key:
        return _invoke_google_chain(system_prompt, user_prompt, fallback=fallback)

    if fallback is not None:
        return fallback
    raise RuntimeError("No LLM is configured. Add GROQ_API_KEY or GOOGLE_API_KEY.")


def invoke_json(system_prompt: str, user_prompt: str, fallback: dict[str, Any]) -> dict[str, Any]:
    fallback_text = json.dumps(fallback)
    raw_response = invoke_text(system_prompt, user_prompt, fallback=fallback_text)

    try:
        payload = json.loads(raw_response)
        _debug_print("parsed llm json response", payload)
        return payload
    except json.JSONDecodeError:
        start = raw_response.find("{")
        end = raw_response.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                payload = json.loads(raw_response[start : end + 1])
                _debug_print("parsed llm json response", payload)
                return payload
            except json.JSONDecodeError:
                pass
        logger.warning("Could not parse LLM JSON response, using fallback payload")
        _debug_print("using llm json fallback payload", fallback)
        return fallback
