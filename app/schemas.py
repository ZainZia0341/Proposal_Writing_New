from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TemplateType(str, Enum):
    PROVIDED = "provided"
    CUSTOM = "custom"
    AI_GENERATED = "ai_generated"


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ResponseType(str, Enum):
    PROPOSALS = "proposals"
    DIRECT_ANSWER = "direct_answer"
    PROPOSAL_UPDATE = "proposal_update"


class ProjectRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    project_id: str
    title: str
    description: str
    tech_stack: list[str] = Field(default_factory=list)
    role: str

    @field_validator("tech_stack", mode="before")
    @classmethod
    def normalize_tech_stack(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return [str(value).strip()]


class JobDetails(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    title: str
    description: str
    budget: str | None = None
    required_skills: list[str] = Field(default_factory=list, alias="skills_required")
    client_info: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_input_fields(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        if "description" not in values and "job_description" in values:
            values["description"] = values["job_description"]
        if "skills_required" not in values and "required_skills" in values:
            values["skills_required"] = values["required_skills"]
        if "skills_required" not in values and "skills" in values:
            values["skills_required"] = values["skills"]
        return values

    @field_validator("required_skills", mode="before")
    @classmethod
    def normalize_required_skills(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return [str(value).strip()]

    def as_search_text(self) -> str:
        parts = [
            self.title,
            self.description,
            self.budget or "",
            ", ".join(self.required_skills),
            self.client_info or "",
        ]
        return "\n".join(part for part in parts if part)


class ConversationMessage(BaseModel):
    role: MessageRole
    content: str
    created_at: str = Field(default_factory=utc_now)


class TemplateSummary(BaseModel):
    template_id: str
    label: str
    description: str
    best_for: str
    template_type: TemplateType = TemplateType.PROVIDED


class TemplateRecord(TemplateSummary):
    body: str


class FullStackUserProfile(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    full_name: str
    designation: str
    expertise_areas: list[str] = Field(default_factory=list)
    experience_languages: list[str] = Field(default_factory=list)
    experience_years: int | None = None
    tone_preference: str | None = None
    notes: dict[str, Any] = Field(default_factory=dict)

    @field_validator("expertise_areas", "experience_languages", mode="before")
    @classmethod
    def normalize_snapshot_lists(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return [str(value).strip()]


class TemplateSnapshot(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    template_id: str
    template_type: TemplateType = TemplateType.PROVIDED
    template_text: str
    label: str | None = None
    description: str | None = None


class PortfolioSyncRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    user_id: str
    projects: list[ProjectRecord] = Field(default_factory=list)
    scraped_profile_text: str | None = None
    full_stack_metadata: dict[str, Any] | None = None


class PortfolioSyncResponse(BaseModel):
    user_id: str
    stored_projects: int
    namespace: str
    received_scraped_profile_text: bool = False
    model_used: str | None = None


class ProposalOption(BaseModel):
    id: str
    label: str
    text: str


class RetrieverToolMessage(BaseModel):
    query: str
    matched_project_ids: list[str] = Field(default_factory=list)
    matched_project_titles: list[str] = Field(default_factory=list)
    accepted: bool = False
    rationale: str = ""
    attempt: int = 0


class ProposalThreadRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_id: str
    thread_id: str
    job_details: JobDetails
    user_profile_snapshot: FullStackUserProfile | None = None
    template_snapshot: TemplateSnapshot | None = None
    template_id: str
    template_text: str
    proposals: list[ProposalOption] = Field(default_factory=list)
    selected_proposal_id: str | None = None
    latest_response_type: ResponseType = ResponseType.PROPOSALS
    messages: list[ConversationMessage] = Field(default_factory=list)
    summary: str | None = None
    last_retriever_tool_message: RetrieverToolMessage | None = None
    status: TaskStatus = TaskStatus.COMPLETED
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)


class GenerateProposalRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_id: str
    thread_id: str | None = None
    user_profile: FullStackUserProfile
    template: TemplateSnapshot
    job_details: JobDetails
    async_mode: bool = False


class GenerateProposalResponse(BaseModel):
    thread_id: str
    task_id: str
    status: TaskStatus
    proposals: list[ProposalOption] = Field(default_factory=list)
    retrieval_used: bool = False
    fallback_used: bool = False
    retrieved_project_ids: list[str] = Field(default_factory=list)
    summary: str | None = None
    model_used: str | None = None


class OptimizeProposalRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    thread_id: str
    selected_proposal_id: str
    feedback_msg: str
    user_id: str | None = None
    async_mode: bool = False


class OptimizeProposalResponse(BaseModel):
    thread_id: str
    task_id: str
    status: TaskStatus
    response_type: ResponseType
    updated_proposal: str | None = None
    direct_answer: str | None = None
    retrieval_used: bool = False
    fallback_used: bool = False
    selected_proposal_id: str
    summary: str | None = None
    model_used: str | None = None


class TaskRecord(BaseModel):
    task_id: str = Field(default_factory=lambda: f"task_{uuid4()}")
    thread_id: str | None = None
    status: TaskStatus = TaskStatus.PENDING
    result: dict[str, Any] | None = None
    error_message: str | None = None
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)


class TaskStatusResponse(BaseModel):
    task_id: str
    thread_id: str | None = None
    status: TaskStatus
    result: dict[str, Any] | None = None
    error_message: str | None = None
    model_used: str | None = None
