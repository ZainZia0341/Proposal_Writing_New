# Single-Table DB Access Patterns

This file explains the proposal API storage flow in simple words and gives sheet-ready rows for the single DynamoDB table design.

## Simple Terms

- `PK` means partition key. It is the main group where an item lives.
- `SK` means sort key. It is the unique item name inside that group.
- `GSI1PK` means partition key for Global Secondary Index 1.
- `GSI1SK` means sort key for Global Secondary Index 1.
- A GSI is an extra lookup path. The base table groups task records by user, but `GSI1` lets the API find a task when it only has `task_id`.

## Table

Table name:

```text
Proposal-Studio
```

Main key:

```text
PK
SK
```

Task lookup index:

```text
GSI1
GSI1PK
GSI1SK
```

## Item Shapes

Proposal thread:

```json
{
  "PK": "USER#zain_zia_001",
  "SK": "THREAD#thread_123",
  "entity_type": "proposal_thread",
  "user_id": "zain_zia_001",
  "thread_id": "thread_123",
  "job_details": {
    "title": "AI Chatbot",
    "description": "Need a chatbot for a website",
    "budget": "$1000",
    "required_skills": ["Python", "OpenAI"]
  },
  "proposals": [
    {
      "id": "alt_1",
      "label": "Balanced",
      "text": "Proposal text..."
    }
  ],
  "messages": [],
  "summary": null,
  "status": "completed"
}
```

Bid style examples:

```json
{
  "PK": "USER#zain_zia_001",
  "SK": "BID_STYLE",
  "record_type": "user_bid_style",
  "thread_id": "bids_profile#zain_zia_001",
  "user_id": "zain_zia_001",
  "bids": [
    {
      "job_details": {
        "title": "AI chatbot",
        "description": "Need RAG workflow"
      },
      "proposal_text": "Hi, I have built similar AI workflows...",
      "markdown": "## Job Title\nAI chatbot\n\n## Sent Proposal\nHi..."
    }
  ]
}
```

Bid example draft:

```json
{
  "PK": "USER#zain_zia_001",
  "SK": "BID_EXAMPLE_DRAFT#draft_thread_123",
  "record_type": "bid_example_draft",
  "thread_id": "draft_thread_123",
  "user_id": "zain_zia_001",
  "example_bid": {
    "proposal_text": "Generated editable bid...",
    "markdown": "## Job Title\n..."
  },
  "messages": [],
  "summary": null,
  "status": "completed"
}
```

Task status:

```json
{
  "PK": "USER#zain_zia_001",
  "SK": "TASK#task_123",
  "GSI1PK": "TASK#task_123",
  "GSI1SK": "USER#zain_zia_001",
  "entity_type": "task",
  "task_id": "task_123",
  "user_id": "zain_zia_001",
  "thread_id": "thread_123",
  "status": "completed",
  "result": {
    "thread_id": "thread_123",
    "task_id": "task_123",
    "status": "completed"
  },
  "error_message": null
}
```

## Endpoint Flow

| Endpoint | DynamoDB behavior | Other storage |
|---|---|---|
| `POST /api/v1/portfolio/sync` | None | Upserts structured projects into Pinecone |
| `POST /api/v1/portfolio/pdf/parse` | None | No persistence, only returns parsed projects/text |
| `POST /api/v1/portfolio/structured/sync` | None | Upserts structured projects into Pinecone |
| `POST /api/v1/proposals/bids/sync` | Put/delete `USER#<id> / BID_STYLE` | None |
| `POST /api/v1/proposals/bids/example` | Put/update `USER#<id> / BID_EXAMPLE_DRAFT#<thread_id>` and `USER#<id> / TASK#<task_id>` | None |
| `POST /api/v1/proposals/generate` | Put/update `USER#<id> / THREAD#<thread_id>` and `USER#<id> / TASK#<task_id>` | Reads portfolio evidence from Pinecone |
| `POST /api/v1/proposals/optimize` | Get/update `USER#<id> / THREAD#<thread_id>` and `USER#<id> / TASK#<task_id>` | May read Pinecone if feedback needs more project evidence |
| `GET /api/v1/tasks/{task_id}` | Query `GSI1PK=TASK#<task_id>` | None |

## Sheet-Ready Rows

| Domain | Use case | Operation | PK | SK | Notes |
|---|---|---|---|---|---|
| Bid Style | Save user bid style examples | PutItem | `USER#<id>` | `BID_STYLE` | Replace-all sync, max 5 examples, stores generated markdown |
| Bid Style | Clear user bid style examples | DeleteItem | `USER#<id>` | `BID_STYLE` | Happens when bids array is empty |
| Bid Example Draft | Create editable AI bid draft | PutItem | `USER#<id>` | `BID_EXAMPLE_DRAFT#<threadId>` | Stores example bid, markdown, messages, summary |
| Bid Example Draft | Update editable AI bid draft | GetItem + PutItem | `USER#<id>` | `BID_EXAMPLE_DRAFT#<threadId>` | Requires user_id and thread_id |
| Proposal Thread | Save generated proposal options | PutItem | `USER#<id>` | `THREAD#<threadId>` | Stores job details, user profile snapshot, `alt_1` to `alt_3`, messages, summary |
| Proposal Thread | Refine selected proposal | GetItem + PutItem | `USER#<id>` | `THREAD#<threadId>` | Optimize endpoint now requires user_id |
| Proposal Thread | Direct answer from thread | GetItem + PutItem | `USER#<id>` | `THREAD#<threadId>` | Stores answer messages and latest response type |
| Task | Create task status | PutItem | `USER#<id>` | `TASK#<taskId>` | Writes `GSI1PK=TASK#<taskId>`, `GSI1SK=USER#<id>` |
| Task | Mark task processing | Query GSI1 + PutItem | `USER#<id>` | `TASK#<taskId>` | Internal update |
| Task | Mark task completed or failed | Query GSI1 + PutItem | `USER#<id>` | `TASK#<taskId>` | Stores final result or error |
| Task | Get task status by task id | Query | `GSI1PK=TASK#<taskId>` | `GSI1SK=USER#<id>` | Powers `GET /api/v1/tasks/{task_id}` |

## Proposal IDs

The API still uses these option IDs inside each thread:

```text
alt_1
alt_2
alt_3
```

These are local option IDs, not global database IDs. The backend normalizes AI output before saving so the frontend can safely send `selected_proposal_id`.
