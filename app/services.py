from __future__ import annotations

from fastapi import HTTPException

from .logging_utils import get_logger
from .repositories import (
    get_proposals_repository,
    get_tasks_repository,
    mark_task_processing,
)
from .schemas import (
    GenerateProposalRequest,
    GenerateProposalResponse,
    OptimizeProposalRequest,
    OptimizeProposalResponse,
    PortfolioSyncRequest,
    PortfolioSyncResponse,
    ProposalThreadRecord,
    TaskRecord,
    TaskStatus,
    TaskStatusResponse,
    TemplateRecord,
)
from .seed_data import DEFAULT_TEMPLATES
from .vector_store import get_project_store

logger = get_logger(__name__)


def list_templates() -> list[TemplateRecord]:
    return list(DEFAULT_TEMPLATES.values())


def sync_portfolio_for_ai_dev(request: PortfolioSyncRequest) -> PortfolioSyncResponse:
    try:
        stored_projects = get_project_store().upsert_projects(request.user_id, request.projects)
    except Exception as exc:
        logger.exception("Portfolio sync failed during Pinecone upsert")
        raise HTTPException(
            status_code=500,
            detail=f"Portfolio sync failed during Pinecone upsert or embedding generation: {exc}",
        ) from exc
    logger.info(
        "Full stack portfolio sync completed for AI dev retrieval",
        extra={"user_id": request.user_id, "stored_projects": stored_projects},
    )
    return PortfolioSyncResponse(
        user_id=request.user_id,
        stored_projects=stored_projects,
        namespace=request.user_id,
        received_scraped_profile_text=bool(request.scraped_profile_text),
        model_used=None,
    )


def build_generation_task(thread_id: str) -> TaskRecord:
    task = TaskRecord(thread_id=thread_id)
    logger.info("Creating generation task", extra={"thread_id": thread_id, "task_id": task.task_id})
    return get_tasks_repository().create(task)


def build_optimization_task(request: OptimizeProposalRequest) -> TaskRecord:
    task = TaskRecord(thread_id=request.thread_id)
    logger.info("Creating optimization task", extra={"thread_id": request.thread_id, "task_id": task.task_id})
    return get_tasks_repository().create(task)


def get_task_status(task_id: str) -> TaskStatusResponse:
    task = get_tasks_repository().get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' was not found.")
    model_used = None
    if isinstance(task.result, dict):
        model_used = task.result.get("model_used")
    return TaskStatusResponse(
        task_id=task.task_id,
        thread_id=task.thread_id,
        status=task.status,
        result=task.result,
        error_message=task.error_message,
        model_used=model_used,
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
