from __future__ import annotations

import copy
from functools import lru_cache
from typing import Protocol

from .config import settings
from .logging_utils import get_logger
from .schemas import ProposalThreadRecord, TaskRecord, TaskStatus, utc_now

logger = get_logger(__name__)

try:  # pragma: no cover - optional dependency
    import boto3
except ImportError:  # pragma: no cover - optional dependency
    boto3 = None


class ProposalsRepository(Protocol):
    def get(self, thread_id: str) -> ProposalThreadRecord | None: ...

    def upsert(self, record: ProposalThreadRecord) -> ProposalThreadRecord: ...


class TasksRepository(Protocol):
    def create(self, task: TaskRecord) -> TaskRecord: ...

    def get(self, task_id: str) -> TaskRecord | None: ...

    def update(self, task_id: str, **fields) -> TaskRecord | None: ...


class _InMemoryStore:
    proposals: dict[str, dict] = {}
    tasks: dict[str, dict] = {}


class InMemoryProposalsRepository:
    def get(self, thread_id: str) -> ProposalThreadRecord | None:
        record = _InMemoryStore.proposals.get(thread_id)
        return ProposalThreadRecord.model_validate(copy.deepcopy(record)) if record else None

    def upsert(self, record: ProposalThreadRecord) -> ProposalThreadRecord:
        payload = record.model_copy(update={"updated_at": utc_now()})
        _InMemoryStore.proposals[payload.thread_id] = payload.model_dump(mode="json")
        return payload


class InMemoryTasksRepository:
    def create(self, task: TaskRecord) -> TaskRecord:
        payload = task.model_copy(update={"updated_at": utc_now()})
        _InMemoryStore.tasks[payload.task_id] = payload.model_dump(mode="json")
        return payload

    def get(self, task_id: str) -> TaskRecord | None:
        record = _InMemoryStore.tasks.get(task_id)
        return TaskRecord.model_validate(copy.deepcopy(record)) if record else None

    def update(self, task_id: str, **fields) -> TaskRecord | None:
        task = self.get(task_id)
        if task is None:
            return None
        updated = task.model_copy(update={**fields, "updated_at": utc_now()})
        _InMemoryStore.tasks[task_id] = updated.model_dump(mode="json")
        return updated


class DynamoProposalsRepository:
    def __init__(self) -> None:
        resource = boto3.resource("dynamodb", region_name=settings.aws_region)
        self.table = resource.Table(settings.proposals_table_name)

    def get(self, thread_id: str) -> ProposalThreadRecord | None:
        result = self.table.get_item(Key={"thread_id": thread_id})
        item = result.get("Item")
        return ProposalThreadRecord.model_validate(item) if item else None

    def upsert(self, record: ProposalThreadRecord) -> ProposalThreadRecord:
        payload = record.model_copy(update={"updated_at": utc_now()})
        self.table.put_item(Item=payload.model_dump(mode="json", exclude_none=True))
        return payload


class DynamoTasksRepository:
    def __init__(self) -> None:
        resource = boto3.resource("dynamodb", region_name=settings.aws_region)
        self.table = resource.Table(settings.tasks_table_name)

    def create(self, task: TaskRecord) -> TaskRecord:
        payload = task.model_copy(update={"updated_at": utc_now()})
        self.table.put_item(Item=payload.model_dump(mode="json", exclude_none=True))
        return payload

    def get(self, task_id: str) -> TaskRecord | None:
        result = self.table.get_item(Key={"task_id": task_id})
        item = result.get("Item")
        return TaskRecord.model_validate(item) if item else None

    def update(self, task_id: str, **fields) -> TaskRecord | None:
        task = self.get(task_id)
        if task is None:
            return None
        updated = task.model_copy(update={**fields, "updated_at": utc_now()})
        self.table.put_item(Item=updated.model_dump(mode="json", exclude_none=True))
        return updated


def repositories_mode() -> str:
    if settings.use_dynamodb and boto3 is not None:
        return "dynamodb"
    return "in_memory"


@lru_cache(maxsize=1)
def get_proposals_repository() -> ProposalsRepository:
    mode = repositories_mode()
    logger.info("Proposals repository mode selected: %s", mode)
    if mode == "dynamodb":
        return DynamoProposalsRepository()
    return InMemoryProposalsRepository()


@lru_cache(maxsize=1)
def get_tasks_repository() -> TasksRepository:
    mode = repositories_mode()
    logger.info("Tasks repository mode selected: %s", mode)
    if mode == "dynamodb":
        return DynamoTasksRepository()
    return InMemoryTasksRepository()


def mark_task_processing(task_id: str, thread_id: str | None = None) -> TaskRecord | None:
    logger.info("Marking task as processing", extra={"task_id": task_id, "thread_id": thread_id})
    return get_tasks_repository().update(task_id, status=TaskStatus.PROCESSING, thread_id=thread_id)


def reset_in_memory_repositories() -> None:
    _InMemoryStore.proposals = {}
    _InMemoryStore.tasks = {}
    get_proposals_repository.cache_clear()
    get_tasks_repository.cache_clear()
