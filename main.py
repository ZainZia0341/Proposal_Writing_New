from __future__ import annotations

from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

from app.config import settings
from app.graph import run_generate_flow, run_optimize_flow
from app.schemas import (
    GenerateProposalRequest,
    GenerateProposalResponse,
    OptimizeProposalRequest,
    OptimizeProposalResponse,
    TaskStatusResponse,
    TemplateSummary,
    UserSignupRequest,
    UserSignupResponse,
)
from app.services import (
    build_generation_task,
    build_optimization_task,
    fail_task,
    finalize_generation_result,
    finalize_optimization_result,
    get_task_status,
    list_templates,
    mark_task_started,
    signup_user,
)

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get(f"{settings.api_prefix}/templates", response_model=list[TemplateSummary])
async def get_templates() -> list[TemplateSummary]:
    return list_templates()


@app.post(f"{settings.api_prefix}/signup", response_model=UserSignupResponse)
async def signup_endpoint(request: UserSignupRequest) -> UserSignupResponse:
    try:
        return signup_user(request)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post(f"{settings.api_prefix}/proposals/generate", response_model=GenerateProposalResponse)
async def generate_proposal_endpoint(request: GenerateProposalRequest) -> GenerateProposalResponse:
    task = build_generation_task(request)
    thread_id = request.thread_id or str(uuid4())
    try:
        mark_task_started(task.task_id, thread_id=thread_id)
        response = run_generate_flow(
            task.task_id,
            {
                "user_id": request.user_id,
                "thread_id": thread_id,
                "template_id": request.template_id,
                "job_details": request.job_details.model_dump(mode="json"),
            },
        )
        return finalize_generation_result(task.task_id, response)
    except HTTPException as exc:
        fail_task(task.task_id, exc)
        raise
    except Exception as exc:
        fail_task(task.task_id, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post(f"{settings.api_prefix}/proposals/optimize", response_model=OptimizeProposalResponse)
async def optimize_proposal_endpoint(request: OptimizeProposalRequest) -> OptimizeProposalResponse:
    task = build_optimization_task(request)
    try:
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


handler = Mangum(app)
