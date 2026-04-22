# Proposal Writer API

FastAPI backend for AI-assisted freelance proposal generation and proposal optimization.

This backend now follows the current architecture:

- Full Stack owns the user account table and user profile state
- AI backend owns:
  - built-in proposal templates
  - Pinecone portfolio ingestion and retrieval
  - proposal thread persistence
  - proposal generation and optimization
  - task lifecycle persistence

## What Is Implemented

### Runtime

- Local runtime: `FastAPI + Uvicorn`
- AWS runtime: `FastAPI + Mangum` through `app.main.handler`
- Main ASGI app: `app.main:app`
- Local helper launcher: `python main.py`

### Built-in Templates

- `GET /api/v1/templates`
- Returns the canonical built-in templates including:
  - `template_id`
  - `label`
  - `description`
  - `best_for`
  - `template_type`
  - full template `body`
- Full Stack should fetch templates from this endpoint instead of hard-coding them separately

### Portfolio Sync

- `POST /api/v1/portfolio/sync`
- Used by Full Stack or the Chrome extension to upsert project history into Pinecone
- Uses `user_id` as the namespace / user scope
- Accepts empty `projects`
- Reuses the same namespace safely for later refreshes
- Works with Pinecone when configured, otherwise uses an in-memory fallback store

### Proposal Generation

- `POST /api/v1/proposals/generate`
- Full Stack sends:
  - `user_id`
  - `user_profile`
  - `template`
  - `job_details`
- The AI backend stores:
  - `user_profile_snapshot`
  - `template_snapshot`
  - `job_details`
  - generated alternatives
  - retriever state
  - rolling summary
- The response includes `model_used` so the caller can see which model handled the request
- Returns 3 indexed proposal alternatives:
  - `alt_1`
  - `alt_2`
  - `alt_3`

### Proposal Optimization

- `POST /api/v1/proposals/optimize`
- Full Stack sends only:
  - `thread_id`
  - `selected_proposal_id`
  - `feedback_msg`
- Optional:
  - `user_id` for extra ownership validation
- Loads everything else from the stored proposal thread snapshot
- The response includes `model_used`

### LangGraph Behavior

- LLM-guided query planning
- LLM-guided retrieval verification
- LLM-guided optimization routing:
  - direct answer
  - revise only
  - retrieve then revise
- Rolling message summarization when thread history grows
- Fallback proposal generation when retrieval never validates
- If Groq returns a rate limit error during a request, the backend switches that request to Google Gemini fallback models

### Observability

- Structured logs to:
  - stdout
  - `logs/app.log`
- Logs include:
  - request start/end
  - storage mode selection
  - routing decisions
  - retrieval retries
  - summary creation
  - fallback generation
  - task lifecycle updates

## Main Endpoints

- `GET /health`
- `GET /api/v1/templates`
- `POST /api/v1/portfolio/sync`
- `POST /api/v1/proposals/generate`
- `POST /api/v1/proposals/optimize`
- `GET /api/v1/tasks/{task_id}`

Legacy compatibility routes are still present:

- `POST /generate_proposal`
- `POST /chat_proposal`

Removed from AI backend:

- `POST /api/v1/signup`

Signup and user-profile persistence now belong to the Full Stack side.

## Local Run

Run with Uvicorn:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Or use the helper:

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

- proposal threads: in-memory repository
- tasks: in-memory repository
- project retrieval: in-memory vector store seeded with demo projects

### Cloud mode

Used when configured.

- DynamoDB for:
  - `Users-Proposals`
  - `Proposal-Tasks`
- Pinecone for project retrieval
- Google embedding model: `gemini-embedding-001`

## AWS Deployment

`serverless.yml` builds a Lambda container image and routes HTTP API requests to `app.main.handler`.

The stack creates these DynamoDB tables:

- `Users-Proposals`
- `Proposal-Tasks`

### DynamoDB Table Shapes

`Users-Proposals` stores one proposal thread per workflow, keyed by `thread_id`.

Example item:

```json
{
  "thread_id": "test_thread_555",
  "user_id": "zain_zia_001",
  "user_profile_snapshot": {
    "full_name": "Zain Zia",
    "designation": "Generative AI Developer",
    "expertise_areas": ["LLMs", "RAG systems", "Vector Databases"],
    "experience_languages": ["Node.js", "Python", "React.js"],
    "experience_years": 5,
    "tone_preference": "upwork"
  },
  "template_snapshot": {
    "template_id": "consultative_expert",
    "template_type": "provided",
    "template_text": "Hi, this aligns closely with my recent project..."
  },
  "job_details": {
    "title": "Senior AI Developer for Children's Educational App",
    "description": "We are looking for a developer to build an interactive AI platform...",
    "budget": "$3,000",
    "required_skills": ["Node.js", "Generative AI", "API Integration"],
    "client_info": "EdTech Startup in London"
  },
  "template_id": "consultative_expert",
  "template_text": "Hi, this aligns closely with my recent project...",
  "proposals": [
    { "id": "alt_1", "label": "Balanced", "text": "Proposal option 1..." },
    { "id": "alt_2", "label": "Consultative", "text": "Proposal option 2..." },
    { "id": "alt_3", "label": "Fast Mover", "text": "Proposal option 3..." }
  ],
  "selected_proposal_id": "alt_2",
  "latest_response_type": "proposal_update",
  "summary": "User asked to justify the budget using AWS experience.",
  "last_retriever_tool_message": {
    "query": "aws rag backend proposal",
    "matched_project_ids": ["p2"],
    "matched_project_titles": ["Aha-doc - Document Intelligence"],
    "accepted": true,
    "rationale": "Relevant backend AI delivery evidence",
    "attempt": 2
  }
}
```

`Proposal-Tasks` stores task lifecycle state, keyed by `task_id`.

Example item:

```json
{
  "task_id": "task_123",
  "thread_id": "test_thread_555",
  "status": "completed",
  "result": {
    "thread_id": "test_thread_555",
    "status": "completed"
  },
  "error_message": null
}
```

## Full Stack Contract Summary

Full Stack should already own and store:

- `user_id`
- `full_name`
- `designation`
- `expertise_areas`
- `experience_languages`
- optional:
  - `experience_years`
  - `tone_preference`
  - current template snapshot

Full Stack should send to AI backend for portfolio sync:

- `user_id`
- `projects`
- optional `scraped_profile_text`

Full Stack should send to AI backend for proposal generation:

- `user_id`
- `user_profile`
- `template`
- `job_details`

Full Stack should send to AI backend for proposal optimization:

- `thread_id`
- `selected_proposal_id`
- `feedback_msg`

The detailed contract is documented in [FULLSTACK_AI_CONTRACT.md](./FULLSTACK_AI_CONTRACT.md).

## Environment Variables

Important settings:

- `LLM_PROVIDER`
- `GROQ_API_KEY`
- `GROQ_MODEL_NAME`
- `GOOGLE_API_KEY`
- `GOOGLE_MODEL_NAME`
- `GOOGLE_FALLBACK_MODELS`
- `USE_DYNAMODB`
- `AWS_REGION`
- `USERS_PROPOSALS_TABLE_NAME`
- `TASKS_TABLE_NAME`
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`
- `RETRIEVAL_TOP_K`
- `RETRIEVAL_MAX_RETRIES`
- `SUMMARY_TRIGGER_MESSAGES`
- `RECENT_MESSAGES_TO_KEEP`
- `API_TIMEOUT_SECONDS`
- `LOG_LEVEL`
- `LOG_DIR`
- `LOG_FILE_NAME`

## Test Payload Files

Payload files live under `test_payloads/`:

- `portfolio_sync.json`
- `generate_proposal.json`
- `generate_proposal_job_description_alias.json`
- `optimize_proposal_direct_answer.json`
- `optimize_proposal_revise.json`
- `task_status_example.json`

There is also a root index file:

- `test_payloads.json`

## Testing

Run the suite:

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

- The backend is local-friendly and does not require cloud services for development.
- Tasks still execute inline for local and current Lambda use.
- The task-status model is already in place for a future async worker.



  ANY - https://ma27f4xhy4.execute-api.us-east-1.amazonaws.com/
  ANY - https://ma27f4xhy4.execute-api.us-east-1.amazonaws.com/{proxy+}