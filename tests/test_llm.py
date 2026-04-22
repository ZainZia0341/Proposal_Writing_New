from __future__ import annotations

from types import SimpleNamespace


class _FakeRateLimitError(Exception):
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message)
        self.status_code = 429


class _StaticResponse:
    def __init__(self, content: str):
        self.content = content


def test_groq_rate_limit_switches_request_to_google(monkeypatch):
    import app.llm as llm

    calls = {"groq": 0, "google": 0}

    class GroqClient:
        def invoke(self, messages):
            calls["groq"] += 1
            raise _FakeRateLimitError()

    class GoogleClient:
        def __init__(self, model_name: str):
            self.model_name = model_name

        def invoke(self, messages):
            calls["google"] += 1
            return _StaticResponse(f"reply from {self.model_name}")

    monkeypatch.setattr(
        llm,
        "settings",
        SimpleNamespace(
            llm_provider="groq",
            groq_api_key="groq-key",
            google_api_key="google-key",
            groq_model_name="openai/gpt-oss-120b",
            google_model_name="gemini-2.5-pro",
            google_fallback_models=("gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"),
        ),
    )
    monkeypatch.setattr(llm, "_get_groq_chat_model", lambda: llm.ResolvedLLM("groq", "openai/gpt-oss-120b", GroqClient()))
    monkeypatch.setattr(
        llm,
        "_get_google_chat_model",
        lambda model_name: llm.ResolvedLLM("google", model_name, GoogleClient(model_name)),
    )

    llm.reset_llm_request_state()
    first = llm.invoke_text("system", "user")
    second = llm.invoke_text("system", "user again")

    assert first == "reply from gemini-2.5-pro"
    assert second == "reply from gemini-2.5-pro"
    assert calls["groq"] == 1
    assert calls["google"] == 2
    assert llm.get_model_used() == "google:gemini-2.5-pro"


def test_non_rate_limit_groq_error_does_not_switch_to_google(monkeypatch):
    import app.llm as llm

    calls = {"groq": 0, "google": 0}

    class GroqClient:
        def invoke(self, messages):
            calls["groq"] += 1
            raise RuntimeError("Temporary upstream failure")

    class GoogleClient:
        def __init__(self, model_name: str):
            self.model_name = model_name

        def invoke(self, messages):
            calls["google"] += 1
            return _StaticResponse(f"reply from {self.model_name}")

    monkeypatch.setattr(
        llm,
        "settings",
        SimpleNamespace(
            llm_provider="groq",
            groq_api_key="groq-key",
            google_api_key="google-key",
            groq_model_name="openai/gpt-oss-120b",
            google_model_name="gemini-2.5-pro",
            google_fallback_models=("gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"),
        ),
    )
    monkeypatch.setattr(llm, "_get_groq_chat_model", lambda: llm.ResolvedLLM("groq", "openai/gpt-oss-120b", GroqClient()))
    monkeypatch.setattr(
        llm,
        "_get_google_chat_model",
        lambda model_name: llm.ResolvedLLM("google", model_name, GoogleClient(model_name)),
    )

    llm.reset_llm_request_state()
    response = llm.invoke_text("system", "user", fallback="fallback response")

    assert response == "fallback response"
    assert calls["groq"] == 1
    assert calls["google"] == 0
    assert llm.get_model_used() is None
