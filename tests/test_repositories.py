from __future__ import annotations

from app.repositories import (
    BID_STYLE_SK,
    DynamoTasksRepository,
    bid_example_draft_sk,
    get_tasks_repository,
    proposal_thread_sk,
    task_sk,
    user_pk,
)
from app.schemas import TaskRecord, TaskStatus


def test_single_table_key_helpers_match_access_pattern_sheet():
    assert user_pk("zain_zia_001") == "USER#zain_zia_001"
    assert proposal_thread_sk("thread_123") == "THREAD#thread_123"
    assert bid_example_draft_sk("draft_123") == "BID_EXAMPLE_DRAFT#draft_123"
    assert task_sk("task_123") == "TASK#task_123"
    assert BID_STYLE_SK == "BID_STYLE"


def test_task_repository_stores_user_owned_tasks_and_finds_by_task_id():
    task = get_tasks_repository().create(TaskRecord(user_id="zain_zia_001", thread_id="thread_123"))
    updated = get_tasks_repository().update(
        task.task_id,
        status=TaskStatus.COMPLETED,
        result={"task_id": task.task_id, "ok": True},
    )

    loaded = get_tasks_repository().get(task.task_id)

    assert updated is not None
    assert loaded is not None
    assert loaded.user_id == "zain_zia_001"
    assert loaded.thread_id == "thread_123"
    assert loaded.status == TaskStatus.COMPLETED
    assert loaded.result == {"task_id": task.task_id, "ok": True}


def test_dynamo_task_update_preserves_single_table_keys():
    class FakeTable:
        def __init__(self):
            self.put_items = []

        def query(self, **kwargs):
            return {
                "Items": [
                    {
                        "PK": "USER#zain_zia_001",
                        "SK": "TASK#task_123",
                        "GSI1PK": "TASK#task_123",
                        "GSI1SK": "USER#zain_zia_001",
                        "task_id": "task_123",
                        "user_id": "zain_zia_001",
                        "thread_id": "thread_123",
                        "status": "processing",
                    }
                ]
            }

        def put_item(self, *, Item):
            self.put_items.append(Item)

    repo = DynamoTasksRepository.__new__(DynamoTasksRepository)
    repo.table = FakeTable()

    updated = repo.update("task_123", status=TaskStatus.COMPLETED, result={"ok": True})

    assert updated is not None
    assert repo.table.put_items[0]["PK"] == "USER#zain_zia_001"
    assert repo.table.put_items[0]["SK"] == "TASK#task_123"
    assert repo.table.put_items[0]["GSI1PK"] == "TASK#task_123"
    assert repo.table.put_items[0]["GSI1SK"] == "USER#zain_zia_001"
    assert repo.table.put_items[0]["status"] == "completed"
    assert repo.table.put_items[0]["result"] == {"ok": True}
