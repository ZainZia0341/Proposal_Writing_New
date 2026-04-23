# Full Stack <-> AI Dev Contract

This document defines the current integration contract between the Full Stack app and the AI proposal backend.

## Ownership

- Full Stack owns:
  - auth
  - signup/profile UI
  - the main users table
  - Chrome extension flow
  - saving user personal/profile data
  - collecting previous bids from the product side
- AI backend owns:
  - Pinecone portfolio ingestion and retrieval
  - bid-style example storage for proposal writing
  - proposal generation
  - proposal optimization
  - proposal thread persistence
  - task lifecycle persistence

Important architecture note:

- the AI backend does not expose signup
- the AI backend does not fetch the Full Stack users table during generate or optimize
- Full Stack sends a snapshot of `user_profile` on generate
- Full Stack syncs previous bids separately through the bids endpoint

## Recommended Flow

1. Full Stack stores the user profile in its own users table.
2. Full Stack or the Chrome extension calls `POST /api/v1/portfolio/sync` to upsert projects into Pinecone.
3. Full Stack calls `POST /api/v1/proposals/bids/sync` with up to 5 previous job + sent proposal pairs.
4. AI backend cleans those bids into markdown and stores them in `Users-Proposals` under a user-style record.
5. Full Stack calls `POST /api/v1/proposals/generate` with `user_profile + job_details`.
6. AI backend loads the stored previous bids, passes all of them together to the LLM, and lets the LLM infer the best hook/style for the current job.
7. AI backend stores the generated proposal thread in `Users-Proposals`.
8. Full Stack calls `POST /api/v1/proposals/optimize` with only `thread_id + selected_proposal_id + feedback_msg`.

## Main Endpoints

### `GET /api/v1/templates`

Purpose:

- returns the canonical built-in templates
- kept for product/UI use
- no longer used by proposal prompting

Response:

```json
[
  {
    "template_id": "consultative_expert",
    "label": "Consultative Expert",
    "description": "Diagnosis-first technical style for AI, backend, and architecture-heavy jobs.",
    "best_for": "AI, RAG, backend, AWS, and complex systems proposals.",
    "template_type": "provided",
    "body": "Hi, this aligns closely with my recent project..."
  }
]
```

Built-in template ids currently available:

- `geeksvisor_classic`
- `consultative_expert`
- `the_fast_mover`

Note:

- this endpoint is still supported
- template text is no longer part of the required `generate` contract

### `POST /api/v1/portfolio/sync`

Purpose:

- send AI-dev-relevant portfolio records for Pinecone upsert
- can be called at signup, after Chrome extension scraping, or during later refresh

Request:

```json
{
  "user_id": "user_123",
  "projects": [
    {
      "project_id": "p1",
      "title": "StoryBloom",
      "description": "Built an AI storybook generator with Node.js and Gemini APIs.",
      "tech_stack": ["Node.js", "OpenAI", "Serverless"],
      "role": "Lead Generative AI Developer"
    }
  ],
  "scraped_profile_text": "Optional raw scraped text",
  "full_stack_metadata": {
    "source": "chrome-extension"
  }
}
```

Request field notes:

- required:
  - `user_id`
- optional:
  - `projects`
  - `scraped_profile_text`
  - `full_stack_metadata`
- fields that can contain multiple values:
  - `projects`
  - `projects[].tech_stack`

Response:

```json
{
  "user_id": "user_123",
  "stored_projects": 1,
  "namespace": "user_123",
  "received_scraped_profile_text": true,
  "model_used": null
}
```

### `POST /api/v1/proposals/bids/sync`

Purpose:

- send up to 5 previous job + sent proposal pairs
- teach the AI backend the user's previous writing behavior, hooks, tone, and structure
- replace the previously stored set for that user

Request:

```json
{
  "user_id": "user_123",
  "bids": [
    {
      "job_details": {
        "title": "AI chatbot for education",
        "description": "Need RAG + LLM workflow...",
        "budget": "$2500",
        "required_skills": ["Python", "LangChain", "Pinecone"],
        "client_info": "EdTech startup"
      },
      "proposal_text": "Hi, this aligns closely with my recent AI work..."
    }
  ]
}
```

Request field notes:

- required:
  - `user_id`
- optional:
  - `bids`
- `bids` can be empty:
  - `[]` means clear the stored examples for that user
- max allowed bids:
  - `5`
- if Full Stack sends more than 5:
  - API returns `422`
- fields that can contain multiple values:
  - `bids`
  - `bids[].job_details.required_skills`
- accepted aliases inside `job_details`:
  - `description` or `job_description`
  - `required_skills` or `skills_required` or `skills`

Response:

```json
{
  "user_id": "user_123",
  "stored_bids": 1,
  "max_examples": 5,
  "model_used": null
}
```

Terminology note:

- when your team says `bids`, the AI backend treats that as:
  - previous job details
  - plus the proposal/response the user actually sent for that job

### `POST /api/v1/proposals/generate`

Purpose:

- generate three proposal alternatives
- use current user profile + current job details + accepted retrieved projects
- use all stored previous bids together as style examples
- store the proposal thread in `Users-Proposals`

Request:

```json
{
  "user_id": "user_123",
  "user_profile": {
    "full_name": "Zain Zia",
    "designation": "Generative AI Developer",
    "expertise_areas": ["LLMs", "RAG systems", "Vector Databases"],
    "experience_languages": ["Node.js", "Python", "React.js"],
    "experience_years": 5,
    "tone_preference": "upwork"
  },
  "job_details": {
    "title": "Senior Node.js Developer for Fintech",
    "description": "We need an expert to build secure API gateways and handle financial transactions.",
    "budget": "$5000",
    "required_skills": ["Node.js", "AWS", "PostgreSQL"],
    "client_info": "Based in UK, focus on high security"
  }
}
```

Request field notes:

- required top-level fields:
  - `user_id`
  - `user_profile`
  - `job_details`
- optional top-level fields:
  - `thread_id`
- `user_profile` required fields:
  - `full_name`
  - `designation`
- `user_profile` optional fields:
  - `expertise_areas`
  - `experience_languages`
  - `experience_years`
  - `tone_preference`
  - `notes`
- `job_details` required fields:
  - `title`
  - `description`
- `job_details` optional fields:
  - `budget`
  - `required_skills`
  - `client_info`
- accepted aliases inside `job_details`:
  - `description` or `job_description`
  - `required_skills` or `skills_required` or `skills`
- fields that can contain multiple values:
  - `user_profile.expertise_areas`
  - `user_profile.experience_languages`
  - `job_details.required_skills`

Compatibility note:

- old clients that still send `template` are tolerated during rollout
- the backend ignores that extra field
- Full Stack should stop depending on template-driven prompting

How style learning works:

- the backend loads all stored bids for that user, up to 5
- all examples are passed together in one prompt
- the LLM decides which hook/style suits the current job best
- the LLM must not copy old facts, client names, budgets, or unsupported skills into the new proposal

Response:

```json
{
  "thread_id": "uuid-789-101",
  "task_id": "task_123",
  "status": "completed",
  "proposals": [
    {
      "id": "alt_1",
      "label": "Balanced",
      "text": "Formal version..."
    },
    {
      "id": "alt_2",
      "label": "Consultative",
      "text": "Technical version..."
    },
    {
      "id": "alt_3",
      "label": "Fast Mover",
      "text": "Short version..."
    }
  ],
  "retrieval_used": true,
  "fallback_used": false,
  "retrieved_project_ids": ["p1", "p2"],
  "summary": null,
  "model_used": "groq:openai/gpt-oss-120b"
}
```

### `POST /api/v1/proposals/optimize`

Purpose:

- optimize a selected proposal
- answer direct questions from stored thread context
- optionally rerun retriever if the AI decides more project context is needed

Request:

```json
{
  "thread_id": "uuid-789-101",
  "selected_proposal_id": "alt_2",
  "feedback_msg": "Make this shorter and justify the budget better."
}
```

Request field notes:

- required:
  - `thread_id`
  - `selected_proposal_id`
  - `feedback_msg`
- optional:
  - `user_id`
- allowed `selected_proposal_id` values:
  - `alt_1`
  - `alt_2`
  - `alt_3`

Response when the AI returns a direct answer:

```json
{
  "thread_id": "uuid-789-101",
  "task_id": "task_456",
  "status": "completed",
  "response_type": "direct_answer",
  "updated_proposal": null,
  "direct_answer": "The stored budget for this job is $5000.",
  "retrieval_used": false,
  "fallback_used": false,
  "selected_proposal_id": "alt_2",
  "summary": null,
  "model_used": "groq:openai/gpt-oss-120b"
}
```

Response when the AI revises the proposal:

```json
{
  "thread_id": "uuid-789-101",
  "task_id": "task_456",
  "status": "completed",
  "response_type": "proposal_update",
  "updated_proposal": "Revised proposal text...",
  "direct_answer": null,
  "retrieval_used": true,
  "fallback_used": false,
  "selected_proposal_id": "alt_2",
  "summary": null,
  "model_used": "google:gemini-2.5-pro"
}
```

### `GET /api/v1/tasks/{task_id}`

Purpose:

- fetch task result or failure state

Response:

```json
{
  "task_id": "task_123",
  "thread_id": "uuid-789-101",
  "status": "completed",
  "result": {},
  "error_message": null,
  "model_used": "groq:openai/gpt-oss-120b"
}
```

## What Full Stack Should Store In Its Users Table

The AI backend does not need to fetch this table directly, but Full Stack should keep these fields because they are the source of truth for the snapshot sent to generate:

- `user_id`
- `full_name`
- `designation`
- `expertise_areas`
- `experience_languages`
- `experience_years`
- `tone_preference`

Useful optional fields:

- `portfolio_last_synced_at`
- `bids_last_synced_at`
- `chrome_extension_connected`
- `preferred_platform`
- standard app fields like:
  - `email`
  - `phoneNumber`
  - `username`

## What Full Stack Sends To AI Backend

For portfolio sync:

- `user_id`
- `projects`
- optional `scraped_profile_text`

For bid sync:

- `user_id`
- `bids`

For proposal generation:

- `user_id`
- `user_profile`
- `job_details`

For proposal optimization:

- `thread_id`
- `selected_proposal_id`
- `feedback_msg`

## What AI Backend Stores In `Users-Proposals`

This table now stores two record types.

Proposal thread records store:

- `thread_id`
- `user_id`
- `user_profile_snapshot`
- `job_details`
- `proposals`
- `selected_proposal_id`
- `messages`
- `summary`
- `last_retriever_tool_message`
- `latest_response_type`

User bid-style records store:

- `thread_id = bids_profile#{user_id}`
- `user_id`
- `record_type = user_bid_style`
- cleaned markdown bid examples
- original structured bid examples

## What AI Backend Stores In `Proposal-Tasks`

- `task_id`
- `thread_id`
- `status`
- `result`
- `error_message`

## Notes For Full Stack

- `generate` no longer needs `template`.
- `GET /api/v1/templates` is still available, but it is no longer part of proposal prompting.
- proposal ids are always:
  - `alt_1`
  - `alt_2`
  - `alt_3`
- bid sync is replace-all:
  - the newest sync overwrites the previously stored examples for that user
- if no stored bids exist for a user, generation still works
- the AI backend is designed to avoid claiming unsupported skills or project details
