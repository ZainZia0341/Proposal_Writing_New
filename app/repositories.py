from __future__ import annotations

import copy
from functools import lru_cache
from typing import Protocol

from app.config import settings
from app.logging_utils import get_logger
from app.schemas import ProposalThreadRecord, TaskRecord, TaskStatus, UserProfile, utc_now
from app.seed_data import DUMMY_USER

logger = get_logger(__name__)

try:  # pragma: no cover - optional dependency
    import boto3
except ImportError:  # pragma: no cover - optional dependency
    boto3 = None


class UsersRepository(Protocol):
    def get(self, user_id: str) -> UserProfile | None: ...

    def upsert(self, user: UserProfile) -> UserProfile: ...


class ProposalsRepository(Protocol):
    def get(self, thread_id: str) -> ProposalThreadRecord | None: ...

    def upsert(self, record: ProposalThreadRecord) -> ProposalThreadRecord: ...


class TasksRepository(Protocol):
    def create(self, task: TaskRecord) -> TaskRecord: ...

    def get(self, task_id: str) -> TaskRecord | None: ...

    def update(self, task_id: str, **fields) -> TaskRecord | None: ...


class _InMemoryStore:
    users: dict[str, dict] = {}
    proposals: dict[str, dict] = {}
    tasks: dict[str, dict] = {}
    seeded: bool = False


def _seed_defaults() -> None:
    if _InMemoryStore.seeded:
        return
    _InMemoryStore.users[DUMMY_USER.user_id] = DUMMY_USER.model_dump(mode="json")
    _InMemoryStore.seeded = True


class InMemoryUsersRepository:
    def __init__(self) -> None:
        _seed_defaults()

    def get(self, user_id: str) -> UserProfile | None:
        record = _InMemoryStore.users.get(user_id)
        return UserProfile.model_validate(copy.deepcopy(record)) if record else None

    def upsert(self, user: UserProfile) -> UserProfile:
        payload = user.model_copy(update={"updated_at": utc_now()})
        _InMemoryStore.users[user.user_id] = payload.model_dump(mode="json")
        return payload


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


class DynamoUsersRepository:
    def __init__(self) -> None:
        resource = boto3.resource("dynamodb", region_name=settings.aws_region)
        self.table = resource.Table(settings.users_table_name)

    def get(self, user_id: str) -> UserProfile | None:
        result = self.table.get_item(Key={"user_id": user_id})
        item = result.get("Item")
        return UserProfile.model_validate(item) if item else None

    def upsert(self, user: UserProfile) -> UserProfile:
        payload = user.model_copy(update={"updated_at": utc_now()})
        self.table.put_item(Item=payload.model_dump(mode="json"))
        return payload


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
        self.table.put_item(Item=payload.model_dump(mode="json"))
        return payload


class DynamoTasksRepository:
    def __init__(self) -> None:
        resource = boto3.resource("dynamodb", region_name=settings.aws_region)
        self.table = resource.Table(settings.tasks_table_name)

    def create(self, task: TaskRecord) -> TaskRecord:
        payload = task.model_copy(update={"updated_at": utc_now()})
        self.table.put_item(Item=payload.model_dump(mode="json"))
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
        self.table.put_item(Item=updated.model_dump(mode="json"))
        return updated


def repositories_mode() -> str:
    if settings.use_dynamodb and boto3 is not None:
        return "dynamodb"
    return "in_memory"


@lru_cache(maxsize=1)
def get_users_repository() -> UsersRepository:
    mode = repositories_mode()
    logger.info("Users repository mode selected: %s", mode)
    if mode == "dynamodb":
        return DynamoUsersRepository()
    return InMemoryUsersRepository()


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
    _InMemoryStore.users = {}
    _InMemoryStore.proposals = {}
    _InMemoryStore.tasks = {}
    _InMemoryStore.seeded = False
    get_users_repository.cache_clear()
    get_proposals_repository.cache_clear()
    get_tasks_repository.cache_clear()
