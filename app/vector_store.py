from __future__ import annotations

import copy
import re
from functools import lru_cache
from typing import Protocol

from langchain_core.documents import Document

from app.config import settings
from app.schemas import ProjectRecord
from app.seed_data import DUMMY_PROJECTS, DUMMY_USER


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z0-9\.\+#]+", text.lower()) if len(token) > 1}


class ProjectVectorStore(Protocol):
    def upsert_projects(self, user_id: str, projects: list[ProjectRecord]) -> int: ...

    def search_projects(self, user_id: str, query: str, top_k: int) -> list[ProjectRecord]: ...


class InMemoryProjectVectorStore:
    def __init__(self) -> None:
        self._projects_by_user: dict[str, list[dict]] = {
            DUMMY_USER.user_id: [project.model_dump(mode="json") for project in DUMMY_PROJECTS]
        }

    def upsert_projects(self, user_id: str, projects: list[ProjectRecord]) -> int:
        serialized = [project.model_dump(mode="json") for project in projects]
        self._projects_by_user[user_id] = serialized
        return len(serialized)

    def search_projects(self, user_id: str, query: str, top_k: int) -> list[ProjectRecord]:
        items = self._projects_by_user.get(user_id, [])
        if not items:
            return []

        query_tokens = _tokenize(query)
        scored: list[tuple[int, dict]] = []
        for item in items:
            haystack = " ".join(
                [
                    item["title"],
                    item["description"],
                    ", ".join(item.get("tech_stack", [])),
                    item["role"],
                ]
            )
            project_tokens = _tokenize(haystack)
            overlap = len(query_tokens.intersection(project_tokens))
            scored.append((overlap, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        top_items = [item for score, item in scored if score > 0][:top_k]
        if not top_items:
            top_items = [item for _, item in scored[:top_k]]
        return [ProjectRecord.model_validate(copy.deepcopy(item)) for item in top_items]


class PineconeProjectVectorStore:
    def __init__(self) -> None:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from langchain_pinecone import PineconeVectorStore

        if not settings.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY is required for Pinecone embeddings.")

        self._vector_store = PineconeVectorStore(
            index_name=settings.pinecone_index_name,
            embedding=GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-001",
                google_api_key=settings.google_api_key,
                output_dimensionality=768,
            ),
        )

    def upsert_projects(self, user_id: str, projects: list[ProjectRecord]) -> int:
        documents = [
            Document(
                page_content=(
                    f"Title: {project.title}\n"
                    f"Description: {project.description}\n"
                    f"Role: {project.role}\n"
                    f"Tech Stack: {', '.join(project.tech_stack)}"
                ),
                metadata={
                    "user_id": user_id,
                    "project_id": project.project_id,
                    "title": project.title,
                    "role": project.role,
                    "tech_stack": project.tech_stack,
                    "description": project.description,
                },
            )
            for project in projects
        ]
        ids = [f"{user_id}:{project.project_id}" for project in projects]
        self._vector_store.add_documents(documents=documents, ids=ids, namespace=user_id)
        return len(projects)

    def search_projects(self, user_id: str, query: str, top_k: int) -> list[ProjectRecord]:
        documents = self._vector_store.similarity_search(
            query=query,
            k=top_k,
            namespace=user_id,
            filter={"user_id": user_id},
        )
        return [
            ProjectRecord(
                project_id=str(doc.metadata.get("project_id")),
                title=str(doc.metadata.get("title", "Untitled Project")),
                description=str(doc.metadata.get("description", doc.page_content)),
                tech_stack=list(doc.metadata.get("tech_stack", [])),
                role=str(doc.metadata.get("role", "Contributor")),
            )
            for doc in documents
        ]


@lru_cache(maxsize=1)
def get_project_store() -> ProjectVectorStore:
    if settings.pinecone_api_key and settings.google_api_key:
        try:  # pragma: no cover - external service bootstrap
            return PineconeProjectVectorStore()
        except Exception:
            return InMemoryProjectVectorStore()
    return InMemoryProjectVectorStore()
