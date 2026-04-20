from __future__ import annotations

import re

from fastapi import HTTPException

from app.config import settings
from app.llm import invoke_text, llm_available
from app.logging_utils import get_logger
from app.repositories import (
    get_proposals_repository,
    get_tasks_repository,
    get_users_repository,
    mark_task_processing,
)
from app.schemas import (
    GenerateProposalRequest,
    GenerateProposalResponse,
    OptimizeProposalRequest,
    OptimizeProposalResponse,
    ProposalThreadRecord,
    TaskRecord,
    TaskStatus,
    TaskStatusResponse,
    TemplateRecord,
    TemplateSummary,
    TemplateType,
    UserProfile,
    UserSignupRequest,
    UserSignupResponse,
)
from app.seed_data import DEFAULT_TEMPLATES, template_summaries
from app.vector_store import get_project_store

logger = get_logger(__name__)


def list_templates() -> list[TemplateSummary]:
    return template_summaries()


def get_template_record(template_id: str) -> TemplateRecord:
    template = DEFAULT_TEMPLATES.get(template_id)
    if template is None:
        raise HTTPException(status_code=404, detail=f"Template '{template_id}' was not found.")
    return template


def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).strip()


def _prepare_custom_template(raw_text: str) -> str:
    collapsed = _normalize_text(raw_text)
    collapsed = collapsed[: settings.custom_template_char_limit]
    collapsed = _truncate_words(collapsed, settings.custom_template_word_limit)

    fallback = collapsed
    if not llm_available():
        return fallback

    return invoke_text(
        system_prompt=(
            "You clean freelance proposal templates before storage. "
            "Return a concise reusable template, keep placeholders intact, remove repetition, "
            "and do not add new claims."
        ),
        user_prompt=(
            f"Clean this template and keep it under {settings.custom_template_word_limit} words:\n\n{collapsed}"
        ),
        fallback=fallback,
    )


def _generate_ai_template(context: str) -> str:
    context = _normalize_text(context)[: settings.custom_template_char_limit]
    fallback = (
        "Hi [Client Name], I read your job post for [Job Title]. I have strong experience with "
        "[Skills] and recently delivered [Relevant Project], which is closely aligned with your needs. "
        "I can help you move this forward with a clean implementation and clear communication. "
        "Would you like me to share the approach I would take for this project?\n\nBest,\n[Name]"
    )
    return invoke_text(
        system_prompt=(
            "You create concise reusable proposal templates. "
            "Use placeholders like [Job Title], [Skills], [Relevant Project], [Client Name], and [Name]."
        ),
        user_prompt=f"Create a reusable proposal template from this guidance:\n\n{context}",
        fallback=fallback,
    )


def signup_user(request: UserSignupRequest) -> UserSignupResponse:
    user_id = request.user_id
    logger.info("Starting signup flow", extra={"user_id": user_id})

    if request.custom_template_text:
        template_type = TemplateType.CUSTOM
        template_id = "custom-template-1"
        template_text = _prepare_custom_template(request.custom_template_text)
        custom_template_text = template_text
        template_summary = TemplateSummary(
            template_id=template_id,
            label="Custom Template",
            description="User-provided template cleaned before persistence.",
            best_for="Personalized proposal writing style.",
            template_type=template_type,
        )
    elif request.ai_template_context:
        template_type = TemplateType.AI_GENERATED
        template_id = "generated-template-1"
        template_text = _generate_ai_template(request.ai_template_context)
        custom_template_text = template_text
        template_summary = TemplateSummary(
            template_id=template_id,
            label="AI Generated Template",
            description="Reusable template generated from user guidance.",
            best_for="Users who want the system to create a proposal style for them.",
            template_type=template_type,
        )
    else:
        selected_template_id = request.selected_template_id or "geeksvisor_classic"
        template_record = get_template_record(selected_template_id)
        template_type = template_record.template_type
        template_id = template_record.template_id
        template_text = template_record.body
        custom_template_text = None
        template_summary = TemplateSummary.model_validate(template_record.model_dump(mode="json"))

    scraped_profile_text = (request.scraped_profile_text or "")[: settings.scrape_text_char_limit]
    notes = {
        "frontend_payload": request.frontend_payload or {},
        "scraped_profile_text": scraped_profile_text,
    }

    user = UserProfile(
        user_id=user_id,
        full_name=request.full_name,
        designation=request.designation,
        expertise_areas=request.expertise_areas,
        experience_languages=request.experience_languages,
        experience_years=request.experience_years,
        template_type=template_type,
        template_id=template_id,
        selected_template_text=template_text,
        custom_template_text=custom_template_text,
        portfolio_projects=request.previous_projects,
        notes=notes,
    )

    users_repo = get_users_repository()
    stored_user = users_repo.upsert(user)
    stored_projects = get_project_store().upsert_projects(stored_user.user_id, request.previous_projects)

    logger.info(
        "Signup flow completed",
        extra={"user_id": stored_user.user_id, "stored_projects": stored_projects, "template_id": template_id},
    )
    return UserSignupResponse(
        user=stored_user,
        stored_projects=stored_projects,
        namespace=stored_user.user_id,
        template=template_summary,
    )


def build_generation_task(request: GenerateProposalRequest) -> TaskRecord:
    task = TaskRecord(thread_id=request.thread_id)
    logger.info("Creating generation task", extra={"thread_id": request.thread_id, "task_id": task.task_id})
    return get_tasks_repository().create(task)


def build_optimization_task(request: OptimizeProposalRequest) -> TaskRecord:
    task = TaskRecord(thread_id=request.thread_id)
    logger.info("Creating optimization task", extra={"thread_id": request.thread_id, "task_id": task.task_id})
    return get_tasks_repository().create(task)


def get_task_status(task_id: str) -> TaskStatusResponse:
    task = get_tasks_repository().get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' was not found.")
    return TaskStatusResponse(
        task_id=task.task_id,
        thread_id=task.thread_id,
        status=task.status,
        result=task.result,
        error_message=task.error_message,
    )


def get_thread_or_404(thread_id: str) -> ProposalThreadRecord:
    thread = get_proposals_repository().get(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' was not found.")
    return thread


def validate_thread_ownership(thread_id: str, user_id: str | None) -> ProposalThreadRecord:
    thread = get_thread_or_404(thread_id)
    if user_id is not None and thread.user_id != user_id:
        raise HTTPException(status_code=403, detail="Thread does not belong to the provided user_id.")
    return thread


def finalize_generation_result(task_id: str, response: GenerateProposalResponse) -> GenerateProposalResponse:
    logger.info("Finalizing generation task", extra={"task_id": task_id, "thread_id": response.thread_id})
    get_tasks_repository().update(
        task_id,
        thread_id=response.thread_id,
        status=TaskStatus.COMPLETED,
        result=response.model_dump(mode="json"),
        error_message=None,
    )
    return response


def finalize_optimization_result(task_id: str, response: OptimizeProposalResponse) -> OptimizeProposalResponse:
    logger.info("Finalizing optimization task", extra={"task_id": task_id, "thread_id": response.thread_id})
    get_tasks_repository().update(
        task_id,
        thread_id=response.thread_id,
        status=TaskStatus.COMPLETED,
        result=response.model_dump(mode="json"),
        error_message=None,
    )
    return response


def fail_task(task_id: str, error: Exception) -> None:
    logger.error("Marking task as failed", extra={"task_id": task_id, "error": str(error)})
    get_tasks_repository().update(
        task_id,
        status=TaskStatus.FAILED,
        error_message=str(error),
    )


def mark_task_started(task_id: str, thread_id: str | None = None) -> None:
    logger.info("Marking task as started", extra={"task_id": task_id, "thread_id": thread_id})
    mark_task_processing(task_id, thread_id=thread_id)
