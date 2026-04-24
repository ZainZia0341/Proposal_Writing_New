from __future__ import annotations

import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

from .config import settings
from .graph import run_bid_example_flow, run_generate_flow, run_optimize_flow
from .logging_utils import configure_logging, get_logger
from .repositories import repositories_mode
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
    TaskStatusResponse,
    TemplateRecord,
)
from .services import (
    build_bid_example_task,
    build_generation_task,
    build_optimization_task,
    fail_task,
    finalize_bid_example_result,
    finalize_generation_result,
    finalize_optimization_result,
    get_task_status,
    list_templates,
    mark_task_started,
    parse_portfolio_pdf_for_ai_dev,
    sync_bid_examples_for_style_learning,
    sync_portfolio_for_ai_dev,
    sync_structured_portfolio_for_ai_dev,
    validate_bid_example_draft_request,
    validate_thread_ownership,
)
from .vector_store import project_store_mode

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info(
        "FastAPI app started",
        extra={
            "repositories_mode": repositories_mode(),
            "project_store_mode": project_store_mode(),
            "log_file": str(settings.log_file_path),
        },
    )
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
handler = Mangum(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    logger.info("Request started", extra={"method": request.method, "path": request.url.path})
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        "Request completed",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )
    return response


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get(f"{settings.api_prefix}/templates", response_model=list[TemplateRecord])
async def get_templates() -> list[TemplateRecord]:
    return list_templates()


@app.post(f"{settings.api_prefix}/portfolio/sync", response_model=PortfolioSyncResponse)
async def portfolio_sync_endpoint(request: PortfolioSyncRequest) -> PortfolioSyncResponse:
    try:
        return sync_portfolio_for_ai_dev(request)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Portfolio sync endpoint failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post(f"{settings.api_prefix}/portfolio/pdf/parse", response_model=PortfolioPdfParseResponse)
async def portfolio_pdf_parse_endpoint(
    user_id: str = Form(...),
    file: UploadFile = File(...),
) -> PortfolioPdfParseResponse:
    try:
        content = await file.read()
        return parse_portfolio_pdf_for_ai_dev(user_id=user_id, file_name=file.filename or "portfolio.pdf", content=content)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Portfolio PDF parse endpoint failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post(f"{settings.api_prefix}/portfolio/structured/sync", response_model=PortfolioSyncResponse)
async def portfolio_structured_sync_endpoint(request: PortfolioSyncRequest) -> PortfolioSyncResponse:
    try:
        return sync_structured_portfolio_for_ai_dev(request)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Structured portfolio sync endpoint failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post(f"{settings.api_prefix}/proposals/bids/sync", response_model=BidSyncResponse)
async def bids_sync_endpoint(request: BidSyncRequest) -> BidSyncResponse:
    try:
        return sync_bid_examples_for_style_learning(request)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Bid sync endpoint failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post(f"{settings.api_prefix}/proposals/bids/example", response_model=BidExampleDraftResponse)
async def bid_example_endpoint(request: BidExampleDraftRequest) -> BidExampleDraftResponse:
    thread_id = request.thread_id or str(uuid4())
    task = build_bid_example_task(thread_id)
    try:
        validate_bid_example_draft_request(request)
        mark_task_started(task.task_id, thread_id=thread_id)
        response = run_bid_example_flow(
            task.task_id,
            {
                "user_id": request.user_id,
                "thread_id": thread_id,
                "user_profile": request.user_profile.model_dump(mode="json") if request.user_profile else None,
                "feedback_msg": request.feedback_msg,
            },
        )
        return finalize_bid_example_result(task.task_id, response)
    except HTTPException as exc:
        fail_task(task.task_id, exc)
        raise
    except PermissionError as exc:
        fail_task(task.task_id, exc)
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        fail_task(task.task_id, exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Bid example endpoint failed")
        fail_task(task.task_id, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post(f"{settings.api_prefix}/proposals/generate", response_model=GenerateProposalResponse)
async def generate_proposal_endpoint(request: GenerateProposalRequest) -> GenerateProposalResponse:
    thread_id = request.thread_id or str(uuid4())
    task = build_generation_task(thread_id)
    try:
        mark_task_started(task.task_id, thread_id=thread_id)
        response = run_generate_flow(
            task.task_id,
            {
                "user_id": request.user_id,
                "thread_id": thread_id,
                "user_profile": request.user_profile.model_dump(mode="json"),
                "job_details": request.job_details.model_dump(mode="json"),
            },
        )
        return finalize_generation_result(task.task_id, response)
    except HTTPException as exc:
        fail_task(task.task_id, exc)
        raise
    except Exception as exc:
        logger.exception("Generate proposal endpoint failed")
        fail_task(task.task_id, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post(f"{settings.api_prefix}/proposals/optimize", response_model=OptimizeProposalResponse)
async def optimize_proposal_endpoint(request: OptimizeProposalRequest) -> OptimizeProposalResponse:
    task = build_optimization_task(request)
    try:
        validate_thread_ownership(request.thread_id, request.user_id)
        mark_task_started(task.task_id, thread_id=request.thread_id)
        response = run_optimize_flow(
            task.task_id,
            {
                "thread_id": request.thread_id,
                "selected_proposal_id": request.selected_proposal_id,
                "feedback_msg": request.feedback_msg,
            },
        )
        return finalize_optimization_result(task.task_id, response)
    except HTTPException as exc:
        fail_task(task.task_id, exc)
        raise
    except Exception as exc:
        logger.exception("Optimize proposal endpoint failed")
        fail_task(task.task_id, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get(f"{settings.api_prefix}/tasks/{{task_id}}", response_model=TaskStatusResponse)
async def task_status_endpoint(task_id: str) -> TaskStatusResponse:
    return get_task_status(task_id)


@app.post("/generate_proposal", response_model=GenerateProposalResponse)
async def legacy_generate_endpoint(request: GenerateProposalRequest) -> GenerateProposalResponse:
    return await generate_proposal_endpoint(request)


@app.post("/chat_proposal", response_model=OptimizeProposalResponse)
async def legacy_chat_endpoint(request: OptimizeProposalRequest) -> OptimizeProposalResponse:
    return await optimize_proposal_endpoint(request)
