# Proposal Writer API

FastAPI backend for AI-assisted freelance proposal generation, proposal optimization, Pinecone portfolio retrieval, and user-specific writing-style learning from previous bids.

## Current Architecture

- Full Stack owns:
  - signup
  - auth
  - main users table
  - profile UI/state
  - Chrome extension flow
- AI backend owns:
  - built-in template catalog endpoint
  - Pinecone portfolio ingestion and retrieval
  - previous bid storage for style learning
  - proposal generation and optimization
  - proposal thread persistence
  - task lifecycle persistence

Important note:

- proposal prompting is no longer template-driven
- generation now learns style from up to 5 previous bids synced through the API
- `GET /api/v1/templates` still exists for UI/product use, but template text is not used in prompt construction anymore

## Runtime

- Local runtime: `FastAPI + Uvicorn`
- AWS runtime: `FastAPI + Mangum` through `app.main.handler`
- Main ASGI app: `app.main:app`
- Local helper launcher: `python main.py`

## Main Endpoints

- `GET /health`
- `GET /api/v1/templates`
- `POST /api/v1/portfolio/sync`
- `POST /api/v1/portfolio/pdf/parse`
- `POST /api/v1/portfolio/structured/sync`
- `POST /api/v1/proposals/bids/sync`
- `POST /api/v1/proposals/bids/example`
- `POST /api/v1/proposals/generate`
- `POST /api/v1/proposals/optimize`
- `GET /api/v1/tasks/{task_id}`

Legacy compatibility routes are still present:

- `POST /generate_proposal`
- `POST /chat_proposal`

Removed from the AI backend:

- `POST /api/v1/signup`

## What Is Implemented

### Templates

- `GET /api/v1/templates`
- returns the canonical built-in templates including:
  - `template_id`
  - `label`
  - `description`
  - `best_for`
  - `template_type`
  - full template `body`
- this endpoint is still useful for product/UI, but not for proposal prompting

### Portfolio Sync

- `POST /api/v1/portfolio/sync`
- upserts projects into Pinecone under `user_id` namespace
- accepts empty `projects`
- safely reuses the same namespace for future refreshes
- uses Pinecone when configured, otherwise falls back to the in-memory vector store

### PDF Portfolio Import

- `POST /api/v1/portfolio/pdf/parse`
- accepts multipart `user_id` + PDF `file`
- uses Mistral OCR/document annotation when configured
- falls back to local PDF text extraction plus the existing LLM structured parser when Mistral fails
- returns editable `projects` shaped like `ProjectRecord`
- `POST /api/v1/portfolio/structured/sync` accepts the reviewed projects and reuses the existing portfolio sync path

### Bid Style Sync

- `POST /api/v1/proposals/bids/sync`
- Full Stack sends up to 5 previous job + sent proposal pairs
- backend deterministically converts them into clean markdown
- stores them in `Users-Proposals` under a dedicated synthetic key:
  - `bids_profile#{user_id}`
- the stored examples are later used as few-shot style references during generation

### Bid Example Drafts

- `POST /api/v1/proposals/bids/example`
- creates an editable example bid when `thread_id` is not provided
- revises the stored example bid when `thread_id + feedback_msg` are provided
- stores draft records separately from synced bid-style examples
- uses LLM intent judgment, not keyword routing
- unrelated requests return:
  - `I can only generate an example bid i can not help you with that`

### Proposal Generation

- `POST /api/v1/proposals/generate`
- Full Stack sends:
  - `user_id`
  - `user_profile`
  - `job_details`
- backend loads:
  - stored previous bids for style learning
  - accepted retrieved projects from Pinecone
- backend stores:
  - `user_profile_snapshot`
  - `job_details`
  - generated alternatives
  - retriever state
  - rolling summary
- returns 3 indexed proposal alternatives:
  - `alt_1`
  - `alt_2`
  - `alt_3`
- response includes `model_used`

### Proposal Optimization

- `POST /api/v1/proposals/optimize`
- Full Stack sends only:
  - `thread_id`
  - `selected_proposal_id`
  - `feedback_msg`
- optional:
  - `user_id` for extra ownership validation
- backend loads everything else from the stored thread snapshot
- optimization routing remains LLM-driven:
  - direct answer
  - revise only
  - retrieve then revise
- response includes `model_used`

### LangGraph Behavior

- LLM-guided query planning
- LLM-guided retrieval verification
- LLM-guided optimization routing
- rolling message summarization when thread history grows
- fallback proposal generation when retrieval never validates
- proposal generation uses all stored previous bids together in one prompt
- if Groq rate-limits a request, the backend switches that request to Google Gemini fallback models

### Observability

- logs to:
  - stdout
  - `logs/app.log`
- logs include:
  - request start/end
  - storage mode selection
  - routing decisions
  - retrieval retries
  - summary creation
  - fallback generation
  - task lifecycle updates

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
- bid-style records: in-memory repository
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

`Users-Proposals` stores two record types.

Proposal thread item example:

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
  "job_details": {
    "title": "Senior AI Developer for Children's Educational App",
    "description": "We are looking for a developer to build an interactive AI platform...",
    "budget": "$3,000",
    "required_skills": ["Node.js", "Generative AI", "API Integration"],
    "client_info": "EdTech Startup in London"
  },
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

Bid-style item example:

```json
{
  "thread_id": "bids_profile#zain_zia_001",
  "user_id": "zain_zia_001",
  "record_type": "user_bid_style",
  "bids": [
    {
      "job_details": {
        "title": "AI chatbot for education",
        "description": "Need RAG + LLM workflow...",
        "budget": "$2500",
        "required_skills": ["Python", "LangChain", "Pinecone"],
        "client_info": "EdTech startup"
      },
      "proposal_text": "Hi, this aligns closely with my recent AI work...",
      "markdown": "## Job Title\nAI chatbot for education\n..."
    }
  ]
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

Full Stack should store in its own users table:

- `user_id`
- `full_name`
- `designation`
- `expertise_areas`
- `experience_languages`
- optional:
  - `experience_years`
  - `tone_preference`

Full Stack should send to AI backend for portfolio sync:

- `user_id`
- `projects`
- optional `scraped_profile_text`

Full Stack should send to AI backend for bid sync:

- `user_id`
- `bids`

Full Stack should send to AI backend for proposal generation:

- `user_id`
- `user_profile`
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
- `MISTRAL_API_KEY`
- `MISTRAL_OCR_MODEL`
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
- `portfolio_structured_sync.json`
- `portfolio_pdf_parse_form.json`
- `bids_sync.json`
- `generate_bid_example.json`
- `update_bid_example.json`
- `bid_example_unrelated_request.json`
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
- bid-style sync and markdown formatting tests
- payload parsing / schema validation tests
- summary window behavior
- retriever retry and fallback behavior

## Notes

- the backend is local-friendly and does not require cloud services for development
- tasks still execute inline for local and current Lambda use
- old clients that still send `template` to generate are tolerated, but the backend ignores it
- the task-status model is already in place for a future async worker
