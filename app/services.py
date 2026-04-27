from __future__ import annotations

from fastapi import HTTPException

from .bid_examples import build_bid_style_record
from .document_processing import parse_portfolio_pdf
from .logging_utils import get_logger
from .repositories import (
    get_proposals_repository,
    get_tasks_repository,
    mark_task_processing,
)
from .schemas import (
    BidExampleDraftRequest,
    BidExampleDraftResponse,
    BidSyncRequest,
    BidSyncResponse,
    GenerateProposalRequest,
    GenerateProposalResponse,
    OptimizeProposalRequest,
    OptimizeProposalResponse,
    PortfolioPdfParseResponse,
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


def sync_bid_examples_for_style_learning(request: BidSyncRequest) -> BidSyncResponse:
    proposals_repo = get_proposals_repository()
    if not request.bids:
        proposals_repo.delete_bid_style(request.user_id)
        logger.info("Cleared stored bid style examples", extra={"user_id": request.user_id})
        return BidSyncResponse(user_id=request.user_id, stored_bids=0, model_used=None)

    record = build_bid_style_record(request.user_id, request.bids)
    stored = proposals_repo.upsert_bid_style(record)
    logger.info(
        "Stored bid style examples for proposal generation",
        extra={"user_id": request.user_id, "stored_bids": len(stored.bids)},
    )
    return BidSyncResponse(user_id=request.user_id, stored_bids=len(stored.bids), model_used=None)


def build_generation_task(user_id: str, thread_id: str) -> TaskRecord:
    task = TaskRecord(user_id=user_id, thread_id=thread_id)
    logger.info("Creating generation task", extra={"thread_id": thread_id, "task_id": task.task_id})
    return get_tasks_repository().create(task)


def build_optimization_task(request: OptimizeProposalRequest) -> TaskRecord:
    task = TaskRecord(user_id=request.user_id, thread_id=request.thread_id)
    logger.info("Creating optimization task", extra={"thread_id": request.thread_id, "task_id": task.task_id})
    return get_tasks_repository().create(task)


def build_bid_example_task(user_id: str, thread_id: str) -> TaskRecord:
    task = TaskRecord(user_id=user_id, thread_id=thread_id)
    logger.info("Creating bid example task", extra={"thread_id": thread_id, "task_id": task.task_id})
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


def get_thread_or_404(user_id: str, thread_id: str) -> ProposalThreadRecord:
    thread = get_proposals_repository().get(user_id, thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' was not found.")
    return thread


def validate_thread_ownership(thread_id: str, user_id: str) -> ProposalThreadRecord:
    return get_thread_or_404(user_id, thread_id)


def validate_bid_example_draft_request(request: BidExampleDraftRequest) -> None:
    if request.thread_id:
        draft = get_proposals_repository().get_bid_example_draft(request.user_id, request.thread_id)
        if draft is None:
            raise HTTPException(status_code=404, detail=f"Bid example draft '{request.thread_id}' was not found.")
        return
    if request.user_profile is None:
        raise HTTPException(status_code=422, detail="user_profile is required when thread_id is not provided.")


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


def finalize_bid_example_result(task_id: str, response: BidExampleDraftResponse) -> BidExampleDraftResponse:
    logger.info("Finalizing bid example task", extra={"task_id": task_id, "thread_id": response.thread_id})
    get_tasks_repository().update(
        task_id,
        thread_id=response.thread_id,
        status=TaskStatus.COMPLETED,
        result=response.model_dump(mode="json"),
        error_message=None,
    )
    return response


def parse_portfolio_pdf_for_ai_dev(user_id: str, file_name: str, content: bytes) -> PortfolioPdfParseResponse:
    if not file_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Only PDF files are supported for portfolio parsing.")
    if not content:
        raise HTTPException(status_code=422, detail="Uploaded PDF file is empty.")
    try:
        response = parse_portfolio_pdf(user_id=user_id, file_name=file_name, content=content)
    except Exception as exc:
        logger.exception("Portfolio PDF parse endpoint failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    logger.info(
        "Portfolio PDF parsed into structured projects",
        extra={"user_id": user_id, "projects": len(response.projects), "model_used": response.model_used},
    )
    return response


def sync_structured_portfolio_for_ai_dev(request: PortfolioSyncRequest) -> PortfolioSyncResponse:
    return sync_portfolio_for_ai_dev(request)


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
