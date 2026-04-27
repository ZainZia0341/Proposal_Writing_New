from __future__ import annotations

import copy
from functools import lru_cache
from typing import Protocol

from .config import settings
from .logging_utils import get_logger
from .schemas import BidExampleDraftRecord, ProposalThreadRecord, TaskRecord, TaskStatus, UserBidStyleRecord, utc_now

logger = get_logger(__name__)

try:  # pragma: no cover - optional dependency
    import boto3
    from boto3.dynamodb.conditions import Key as DynamoKey
except ImportError:  # pragma: no cover - optional dependency
    boto3 = None
    DynamoKey = None


def user_pk(user_id: str) -> str:
    return f"USER#{user_id}"


def proposal_thread_sk(thread_id: str) -> str:
    return f"THREAD#{thread_id}"


def bid_example_draft_sk(thread_id: str) -> str:
    return f"BID_EXAMPLE_DRAFT#{thread_id}"


def task_sk(task_id: str) -> str:
    return f"TASK#{task_id}"


BID_STYLE_SK = "BID_STYLE"


class ProposalsRepository(Protocol):
    def get(self, user_id: str, thread_id: str) -> ProposalThreadRecord | None: ...

    def upsert(self, record: ProposalThreadRecord) -> ProposalThreadRecord: ...

    def get_bid_example_draft(self, user_id: str, thread_id: str) -> BidExampleDraftRecord | None: ...

    def upsert_bid_example_draft(self, record: BidExampleDraftRecord) -> BidExampleDraftRecord: ...

    def get_bid_style(self, user_id: str) -> UserBidStyleRecord | None: ...

    def upsert_bid_style(self, record: UserBidStyleRecord) -> UserBidStyleRecord: ...

    def delete_bid_style(self, user_id: str) -> None: ...


class TasksRepository(Protocol):
    def create(self, task: TaskRecord) -> TaskRecord: ...

    def get(self, task_id: str) -> TaskRecord | None: ...

    def update(self, task_id: str, **fields) -> TaskRecord | None: ...


class _InMemoryStore:
    proposals: dict[str, dict] = {}
    tasks: dict[str, dict] = {}
    bid_styles: dict[str, dict] = {}
    bid_example_drafts: dict[str, dict] = {}


class InMemoryProposalsRepository:
    def get(self, user_id: str, thread_id: str) -> ProposalThreadRecord | None:
        record = _InMemoryStore.proposals.get(f"{user_id}:{thread_id}")
        return ProposalThreadRecord.model_validate(copy.deepcopy(record)) if record else None

    def upsert(self, record: ProposalThreadRecord) -> ProposalThreadRecord:
        payload = record.model_copy(update={"updated_at": utc_now()})
        _InMemoryStore.proposals[f"{payload.user_id}:{payload.thread_id}"] = payload.model_dump(mode="json")
        return payload

    def get_bid_example_draft(self, user_id: str, thread_id: str) -> BidExampleDraftRecord | None:
        record = _InMemoryStore.bid_example_drafts.get(f"{user_id}:{thread_id}")
        return BidExampleDraftRecord.model_validate(copy.deepcopy(record)) if record else None

    def upsert_bid_example_draft(self, record: BidExampleDraftRecord) -> BidExampleDraftRecord:
        payload = record.model_copy(update={"updated_at": utc_now()})
        _InMemoryStore.bid_example_drafts[f"{payload.user_id}:{payload.thread_id}"] = payload.model_dump(mode="json")
        return payload

    def get_bid_style(self, user_id: str) -> UserBidStyleRecord | None:
        record = _InMemoryStore.bid_styles.get(f"{user_id}:{BID_STYLE_SK}")
        return UserBidStyleRecord.model_validate(copy.deepcopy(record)) if record else None

    def upsert_bid_style(self, record: UserBidStyleRecord) -> UserBidStyleRecord:
        payload = record.model_copy(update={"updated_at": utc_now()})
        _InMemoryStore.bid_styles[f"{payload.user_id}:{BID_STYLE_SK}"] = payload.model_dump(mode="json")
        return payload

    def delete_bid_style(self, user_id: str) -> None:
        _InMemoryStore.bid_styles.pop(f"{user_id}:{BID_STYLE_SK}", None)


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
        self.table = resource.Table(settings.single_table_name)

    def get(self, user_id: str, thread_id: str) -> ProposalThreadRecord | None:
        result = self.table.get_item(Key={"PK": user_pk(user_id), "SK": proposal_thread_sk(thread_id)})
        item = result.get("Item")
        if item and item.get("record_type"):
            return None
        return ProposalThreadRecord.model_validate(item) if item else None

    def upsert(self, record: ProposalThreadRecord) -> ProposalThreadRecord:
        payload = record.model_copy(update={"updated_at": utc_now()})
        item = {
            "PK": user_pk(payload.user_id),
            "SK": proposal_thread_sk(payload.thread_id),
            "entity_type": "proposal_thread",
            **payload.model_dump(mode="json", exclude_none=True),
        }
        self.table.put_item(Item=item)
        return payload

    def get_bid_example_draft(self, user_id: str, thread_id: str) -> BidExampleDraftRecord | None:
        result = self.table.get_item(Key={"PK": user_pk(user_id), "SK": bid_example_draft_sk(thread_id)})
        item = result.get("Item")
        if item and item.get("record_type") != "bid_example_draft":
            return None
        return BidExampleDraftRecord.model_validate(item) if item else None

    def upsert_bid_example_draft(self, record: BidExampleDraftRecord) -> BidExampleDraftRecord:
        payload = record.model_copy(update={"updated_at": utc_now()})
        item = {
            "PK": user_pk(payload.user_id),
            "SK": bid_example_draft_sk(payload.thread_id),
            **payload.model_dump(mode="json", exclude_none=True),
        }
        self.table.put_item(Item=item)
        return payload

    def get_bid_style(self, user_id: str) -> UserBidStyleRecord | None:
        result = self.table.get_item(Key={"PK": user_pk(user_id), "SK": BID_STYLE_SK})
        item = result.get("Item")
        return UserBidStyleRecord.model_validate(item) if item else None

    def upsert_bid_style(self, record: UserBidStyleRecord) -> UserBidStyleRecord:
        payload = record.model_copy(update={"updated_at": utc_now()})
        item = {
            "PK": user_pk(payload.user_id),
            "SK": BID_STYLE_SK,
            **payload.model_dump(mode="json", exclude_none=True),
        }
        self.table.put_item(Item=item)
        return payload

    def delete_bid_style(self, user_id: str) -> None:
        self.table.delete_item(Key={"PK": user_pk(user_id), "SK": BID_STYLE_SK})


class DynamoTasksRepository:
    def __init__(self) -> None:
        resource = boto3.resource("dynamodb", region_name=settings.aws_region)
        self.table = resource.Table(settings.single_table_name)

    def create(self, task: TaskRecord) -> TaskRecord:
        payload = task.model_copy(update={"updated_at": utc_now()})
        item = {
            "PK": user_pk(payload.user_id),
            "SK": task_sk(payload.task_id),
            "GSI1PK": task_sk(payload.task_id),
            "GSI1SK": user_pk(payload.user_id),
            "entity_type": "task",
            **payload.model_dump(mode="json", exclude_none=True),
        }
        self.table.put_item(Item=item)
        return payload

    def get(self, task_id: str) -> TaskRecord | None:
        result = self.table.query(
            IndexName="GSI1",
            KeyConditionExpression=DynamoKey("GSI1PK").eq(task_sk(task_id)),
            Limit=1,
        )
        items = result.get("Items", [])
        item = items[0] if items else None
        return TaskRecord.model_validate(item) if item else None

    def update(self, task_id: str, **fields) -> TaskRecord | None:
        task = self.get(task_id)
        if task is None:
            return None
        updated = task.model_copy(update={**fields, "updated_at": utc_now()})
        item = {
            "PK": user_pk(updated.user_id),
            "SK": task_sk(updated.task_id),
            "GSI1PK": task_sk(updated.task_id),
            "GSI1SK": user_pk(updated.user_id),
            "entity_type": "task",
            **updated.model_dump(mode="json", exclude_none=True),
        }
        self.table.put_item(Item=item)
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
    _InMemoryStore.bid_styles = {}
    _InMemoryStore.bid_example_drafts = {}
    get_proposals_repository.cache_clear()
    get_tasks_repository.cache_clear()
