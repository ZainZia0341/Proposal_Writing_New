from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.llm import reset_llm_runtime
from app.main import app
from app.repositories import reset_in_memory_repositories
from app.vector_store import reset_project_store


BASE_DIR = Path(__file__).resolve().parent.parent
PAYLOAD_DIR = BASE_DIR / "test_payloads"


@pytest.fixture(autouse=True)
def reset_local_state(monkeypatch):
    import app.repositories as repositories
    import app.vector_store as vector_store

    reset_llm_runtime()
    monkeypatch.setattr(repositories, "repositories_mode", lambda: "in_memory")
    monkeypatch.setattr(vector_store, "project_store_mode", lambda: "in_memory")
    reset_in_memory_repositories()
    reset_project_store()
    yield
    reset_llm_runtime()
    reset_in_memory_repositories()
    reset_project_store()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def load_payload():
    def _load(name: str) -> dict:
        with (PAYLOAD_DIR / name).open("r", encoding="utf-8") as file:
            return json.load(file)

    return _load
