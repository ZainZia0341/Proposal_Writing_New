from __future__ import annotations

import json
from io import BytesIO
from typing import Any

from pydantic import BaseModel
from pypdf import PdfReader

from .config import settings
from .llm import get_model_used, invoke_json, reset_llm_request_state
from .logging_utils import get_logger
from .schemas import PortfolioPdfParseResponse, ProjectRecord, StructuredPortfolioProjects

logger = get_logger(__name__)

PROJECT_ANNOTATION_PROMPT = (
    "Extract freelance portfolio projects from this document. Return only projects that are useful for proposal "
    "retrieval. Each project must include project_id, title, description, tech_stack, and role. "
    "Use concise but evidence-rich descriptions. If a field is missing, infer only a neutral placeholder, not fake facts."
)


def _mistral_client():
    if not settings.mistral_api_key:
        raise RuntimeError("MISTRAL_API_KEY is required for Mistral OCR parsing.")
    try:
        from mistralai import Mistral
    except ImportError:  # pragma: no cover - SDK version compatibility
        from mistralai.client import Mistral

    return Mistral(api_key=settings.mistral_api_key)


def _response_to_dict(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    if isinstance(response, BaseModel):
        return response.model_dump(mode="json")
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "model_dump_json"):
        return json.loads(response.model_dump_json())
    return {}


def _annotation_to_projects(annotation: Any) -> list[ProjectRecord]:
    if annotation is None:
        return []
    if isinstance(annotation, StructuredPortfolioProjects):
        return annotation.projects
    if isinstance(annotation, str):
        try:
            annotation = json.loads(annotation)
        except json.JSONDecodeError:
            return []
    elif isinstance(annotation, BaseModel):
        annotation = annotation.model_dump(mode="json")
    elif hasattr(annotation, "model_dump"):
        annotation = annotation.model_dump()

    if not isinstance(annotation, dict):
        return []
    payload = StructuredPortfolioProjects.model_validate(annotation)
    return payload.projects


def _extract_ocr_markdown(response: Any) -> str:
    payload = _response_to_dict(response)
    pages = payload.get("pages", [])
    markdown_parts: list[str] = []
    for page in pages:
        if isinstance(page, dict) and page.get("markdown"):
            markdown_parts.append(str(page["markdown"]))
    return "\n\n".join(markdown_parts).strip()


def extract_projects_with_mistral_ocr(file_name: str, content: bytes) -> PortfolioPdfParseResponse:
    try:
        from mistralai.extra import response_format_from_pydantic_model
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Installed mistralai package does not expose structured OCR helpers.") from exc

    client = _mistral_client()
    uploaded_file = client.files.upload(
        file={
            "file_name": file_name,
            "content": content,
        },
        purpose="ocr",
    )
    try:
        signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
        response = client.ocr.process(
            model=settings.mistral_ocr_model,
            document={
                "type": "document_url",
                "document_url": signed_url.url,
            },
            document_annotation_format=response_format_from_pydantic_model(StructuredPortfolioProjects),
            document_annotation_prompt=PROJECT_ANNOTATION_PROMPT,
            table_format="markdown",
            include_image_base64=False,
        )
    finally:
        delete_file = getattr(client.files, "delete", None)
        if callable(delete_file):
            try:
                delete_file(file_id=uploaded_file.id)
            except Exception as exc:  # pragma: no cover - cleanup best effort
                logger.warning("Could not delete uploaded Mistral OCR file: %s", exc)

    payload = _response_to_dict(response)
    projects = _annotation_to_projects(payload.get("document_annotation"))
    extracted_markdown = _extract_ocr_markdown(response)
    if not projects:
        raise RuntimeError("Mistral OCR did not return any structured project records.")
    return PortfolioPdfParseResponse(
        user_id="",
        projects=projects,
        extracted_markdown=extracted_markdown,
        model_used=f"mistral:{settings.mistral_ocr_model}",
    )


def extract_text_from_pdf_bytes(content: bytes) -> str:
    reader = PdfReader(BytesIO(content))
    parts = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(part for part in parts if part).strip()


def _rough_project_fallback(text: str) -> list[ProjectRecord]:
    stripped = text.strip()
    if not stripped:
        return []
    first_line = next((line.strip() for line in stripped.splitlines() if line.strip()), "Imported Portfolio Project")
    return [
        ProjectRecord(
            project_id="pdf_project_1",
            title=first_line[:120],
            description=stripped[:4000],
            tech_stack=[],
            role="Imported Portfolio Project",
        )
    ]


def structure_projects_from_text_with_llm(text: str) -> list[ProjectRecord]:
    fallback = {"projects": [project.model_dump(mode="json") for project in _rough_project_fallback(text)]}
    payload = invoke_json(
        system_prompt=(
            "Convert extracted portfolio PDF text into JSON for vector storage. Return JSON with a projects array. "
            "Each project must contain project_id, title, description, tech_stack, and role. "
            "Do not invent tools, achievements, clients, or roles not supported by the text."
        ),
        user_prompt=text[:12000],
        fallback=fallback,
    )
    return [ProjectRecord.model_validate(project) for project in payload.get("projects", [])]


def parse_portfolio_pdf(user_id: str, file_name: str, content: bytes) -> PortfolioPdfParseResponse:
    reset_llm_request_state()
    try:
        parsed = extract_projects_with_mistral_ocr(file_name, content)
        return parsed.model_copy(update={"user_id": user_id})
    except Exception as exc:
        logger.warning("Mistral OCR portfolio parse failed, falling back to local PDF text extraction: %s", exc)

    try:
        extracted_text = extract_text_from_pdf_bytes(content)
        projects = structure_projects_from_text_with_llm(extracted_text)
    except Exception as exc:
        logger.exception("Portfolio PDF fallback parsing failed")
        raise RuntimeError(f"Portfolio PDF parsing failed with Mistral OCR and local fallback: {exc}") from exc

    if not projects:
        raise RuntimeError("Portfolio PDF parsing did not produce any project records.")
    return PortfolioPdfParseResponse(
        user_id=user_id,
        projects=projects,
        extracted_markdown=extracted_text,
        model_used=get_model_used() or "local:pypdf",
    )
