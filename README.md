# Proposal Writer API

FastAPI + LangGraph + Pinecone + Gemini/Groq

## Project Structure

```
proposal_api/
├── pyproject.toml
├── .env.example
├── README.md
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app, middleware, routers
│   ├── config.py                # Settings from .env
│   ├── dependencies.py          # Shared dependencies (LLM, Pinecone)
│   ├── data/
│   │   └── projects.py          # All project details as structured data
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── indexer.py           # Embed & upsert projects into Pinecone
│   │   └── retriever.py         # Query Pinecone for relevant projects
│   ├── graphs/
│   │   ├── __init__.py
│   │   ├── proposal_graph.py    # LangGraph for single proposal generation
│   │   └── chat_graph.py        # LangGraph for chat-based proposal refinement
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── proposal.py          # POST /api/v1/proposal/generate
│   │   └── chat.py              # POST/GET /api/v1/chat/...
│   └── schemas/
│       ├── __init__.py
│       ├── proposal.py          # Request/Response models
│       └── chat.py
```

## Setup

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install
uv venv
source .venv/bin/activate
uv sync

# Copy and fill env
cp .env.example .env

# Index projects into Pinecone (run once)
uv run python -m app.rag.indexer

# Start server
uv run uvicorn app.main:app --reload --port 8000
```

## API Endpoints

### POST /api/v1/proposal/generate
One-shot proposal generation.

### POST /api/v1/chat/start
Start a new proposal chat session. Returns thread_id.

### POST /api/v1/chat/message
Send message in existing thread (refine proposal).

### GET /api/v1/chat/{thread_id}
Get full chat history for a thread.

### DELETE /api/v1/chat/{thread_id}
Delete a chat thread.