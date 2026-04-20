# Proposal Writer API

FastAPI-first backend for:

- user signup and template selection
- user-scoped project ingestion
- proposal generation with LangGraph routing
- proposal optimization with LangGraph routing
- rolling thread summaries to control context size

## What Is Implemented

### FastAPI-only runtime

- The app runs locally with `FastAPI + Uvicorn`
- `Mangum` is no longer part of the live application path
- Main app entrypoint: `app.main:app`
- Convenience launcher: `python main.py`

### Signup flow

- `POST /api/v1/signup`
- Stores user details in `Users-Table` shape through repository abstraction
- Requires frontend to send a stable `user_id`; the backend no longer auto-generates it during signup
- Supports:
  - provided templates
  - custom templates
  - AI-generated templates
- Rejects payloads that send more than one template source among:
  - `selected_template_id`
  - `custom_template_text`
  - `ai_template_context`
- Stores structured previous projects on the user record
- Upserts project records into a user-scoped vector store namespace/filter
- Reuses an existing Pinecone namespace for the same `user_id` instead of trying to create a duplicate namespace
- Uses in-memory seeded fallback when cloud services are not configured

### Proposal generation flow

- `POST /api/v1/proposals/generate`
- Creates or reuses a thread id
- Loads user details and selected template
- Uses LangGraph nodes for:
  - context initialization
  - optional message summarization
  - query planning
  - retrieval
  - retrieval verification
  - retry loop up to 3 attempts
  - proposal generation
  - LLM-based fallback generation when retrieval never validates, using stored user/job/template context
  - persistence
- Returns 3 indexed proposal alternatives:
  - `alt_1`
  - `alt_2`
  - `alt_3`

### Proposal optimization flow

- `POST /api/v1/proposals/optimize`
- Validates thread ownership when `user_id` is supplied
- Loads thread, job details, selected proposal, and stored summary
- Uses LLM-driven routing through LangGraph to decide:
  - direct answer from stored context
  - revise only
  - retrieve then revise
- Keeps pinned job/user/template context separate from rolling conversation history
- Persists only the last accepted retriever tool payload

### Memory and summary handling

- Uses LangGraph message-aware state
- Uses LangGraph `Command` routing
- Uses LangGraph message removal pattern to keep the latest message window
- Summarizes older messages and stores the summary on the thread record

### Observability

- Structured logging to:
  - stdout
  - `logs/app.log`
- Logs:
  - request start/end
  - storage mode selection
  - routing decisions
  - retrieval retries
  - summary creation
  - fallback generation
  - task lifecycle events

## Main Endpoints

- `GET /health`
- `GET /api/v1/templates`
- `POST /api/v1/signup`
- `POST /api/v1/proposals/generate`
- `POST /api/v1/proposals/optimize`
- `GET /api/v1/tasks/{task_id}`

Legacy compatibility routes are still present:

- `POST /generate_proposal`
- `POST /chat_proposal`

## Local Run

Run with Uvicorn directly:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Or use the helper launcher:

```bash
python main.py
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## Persistence Modes

### Local fallback mode

Used when DynamoDB / Pinecone config is missing.

- user data: in-memory repository
- proposal threads: in-memory repository
- tasks: in-memory repository
- project search: in-memory vector store seeded with dummy data

### Cloud mode

Used when configured.

- DynamoDB repositories for users, proposals, and tasks
- Pinecone vector search with user namespace / metadata filter
- Use the modern Python SDK package name `pinecone`
- Do not install deprecated `pinecone-client`; if it is already installed in your venv, uninstall it

## Environment Variables

Important settings:

- `LLM_PROVIDER`
- `GROQ_API_KEY`
- `GOOGLE_API_KEY`
- `USE_DYNAMODB`
- `AWS_REGION`
- `USERS_TABLE_NAME`
- `USERS_PROPOSALS_TABLE_NAME`
- `TASKS_TABLE_NAME`
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`
- `RETRIEVAL_MAX_RETRIES`
- `SUMMARY_TRIGGER_MESSAGES`
- `RECENT_MESSAGES_TO_KEEP`
- `CUSTOM_TEMPLATE_CHAR_LIMIT`
- `CUSTOM_TEMPLATE_WORD_LIMIT`
- `LOG_LEVEL`
- `LOG_DIR`
- `LOG_FILE_NAME`

## Test Payload Files

Payload files are split under `test_payloads/`:

- `signup_selected_template.json`
- `signup_custom_template.json`
- `generate_proposal.json`
- `generate_proposal_job_description_alias.json`
- `optimize_proposal_direct_answer.json`
- `optimize_proposal_revise.json`
- `task_status_example.json`

There is also a root index file:

- `test_payloads.json`

## Testing

Run the test suite:

```bash
python -m pytest -q
```

Current automated coverage includes:

- API endpoint tests with `FastAPI TestClient`
- graph routing tests with mocked LLM decisions
- payload parsing / schema validation tests
- summary window behavior
- retriever retry and fallback behavior

## Notes

- The backend is intentionally local-friendly and does not require cloud services for development.
- Tasks still execute inline for FastAPI local testing.
- The task-status model is already in place for a future async worker / background execution setup.
