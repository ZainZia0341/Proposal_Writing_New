# Full Stack <-> AI Dev Contract

This document defines the recommended integration contract between the Full Stack side and the AI dev backend.

## Ownership

- Full Stack owns:
  - auth
  - user account records
  - profile UI
  - template selection UI
  - Chrome extension flow
  - the main users table
- AI dev backend owns:
  - Pinecone ingestion and retrieval
  - proposal generation
  - proposal optimization
  - proposal thread persistence
  - task lifecycle persistence

Important architecture note:
- the AI dev backend does not expose a signup endpoint anymore
- user signup, profile save, and user-table persistence happen on the Full Stack side

## Recommended Flow

1. Full Stack stores user profile and template selection in its own users table.
2. Full Stack fetches canonical built-in template ids plus full text from `GET /api/v1/templates`.
3. If the user chooses a built-in template, Full Stack keeps that selected template snapshot on its side.
4. Full Stack or Chrome extension calls `POST /api/v1/portfolio/sync` to upsert project history into Pinecone.
5. Full Stack calls `POST /api/v1/proposals/generate` with `user_profile + template + job_details`.
6. AI dev backend stores the Full Stack snapshot in `Users-Proposals` with the thread.
7. Full Stack calls `POST /api/v1/proposals/optimize` with only `thread_id + selected_proposal_id + feedback_msg`.

## Main Endpoints

### `GET /api/v1/templates`

Purpose:
- give Full Stack the canonical built-in proposal templates
- return both the template id and the full template text
- avoid duplicating hard-coded template text in the Full Stack app

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
  },
  {
    "template_id": "geeksvisor_classic",
    "label": "GeeksVisor Classic",
    "description": "Balanced and portfolio-led proposal for general SaaS and full-stack jobs.",
    "best_for": "General full-stack, SaaS, and product-engineering proposals.",
    "template_type": "provided",
    "body": "Hi, this sounds like a perfect fit..."
  }
]
```

How Full Stack should use this:
- show the returned built-in templates in the UI
- when the user selects one, keep:
  - `template_id`
  - `template_type`
  - the returned `body`
- when calling generate, map that `body` into `template.template_text`
- this means Full Stack does not need to hard-code the three provided template texts separately

Built-in template ids currently available:
- `geeksvisor_classic`
- `consultative_expert`
- `the_fast_mover`

### `POST /api/v1/portfolio/sync`

Purpose:
- send AI-dev-relevant portfolio records for Pinecone upsert
- can be called:
  - at signup
  - when Chrome extension finishes scraping
  - when user refreshes portfolio later

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

Request field notes:
- required:
  - `user_id`
- optional:
  - `projects`
  - `scraped_profile_text`
  - `full_stack_metadata`
- fields that can contain multiple values:
  - `projects` is an array and can be empty
  - inside each project, `tech_stack` can contain multiple values
- `scraped_profile_text` is accepted for future AI/cleaning use and operational context; it is not currently required for proposal writing
- each project item should include:
  - `project_id`
  - `title`
  - `description`
  - `role`
  - `tech_stack`

### `POST /api/v1/proposals/generate`

Purpose:
- generate three proposal alternatives
- store the Full Stack snapshot inside `Users-Proposals`

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
  "template": {
    "template_id": "consultative_expert",
    "template_type": "provided",
    "template_text": "Hi, this aligns closely with my recent project..."
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

Notes:
- `thread_id` is optional. If not provided, AI dev backend creates one.
- `job_details.description` can also be sent as `job_description`.
- `job_details.required_skills` can also be sent as `skills_required` or `skills`.
- for built-in templates, `template.template_text` should come from the selected `body` returned by `GET /api/v1/templates`
- for custom or AI-generated templates, `template.template_text` should come from Full Stack storage
- response includes `model_used`
- if Groq rate-limits a request, the backend may switch that request to a Google Gemini fallback model and `model_used` will reflect that

Request field notes:
- required top-level fields:
  - `user_id`
  - `user_profile`
  - `template`
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
- `template` required fields:
  - `template_id`
  - `template_type`
  - `template_text`
- `template` optional fields:
  - `label`
  - `description`
- `job_details` required fields:
  - `title`
  - `description`
- `job_details` optional fields:
  - `budget`
  - `required_skills`
  - `client_info`
- fields that can contain multiple values:
  - `user_profile.expertise_areas`
  - `user_profile.experience_languages`
  - `job_details.required_skills`

Allowed `template.template_type` values:
- `provided`
  - use this when the user selected one of the built-in templates returned by `GET /api/v1/templates`
  - current built-in `template_id` values are:
    - `geeksvisor_classic`
    - `consultative_expert`
    - `the_fast_mover`
- `custom`
  - use this when the user wrote or saved a custom template
  - recommended `template_id` format: `custom-template-1`, `custom-template-2`, etc.
- `ai_generated`
  - use this when Full Stack created or saved a template generated by AI for that user
  - recommended `template_id` format: `generated-template-1`, `generated-template-2`, etc.

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
- optionally rerun retriever if AI decides more project context is needed

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
- recommended flow:
  - Full Stack should send only `thread_id`, `selected_proposal_id`, and `feedback_msg`
  - `user_id` is optional and can be sent only if Full Stack also wants extra ownership validation on the request

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

The AI dev backend no longer needs to fetch this table during generate or optimize, but Full Stack should still keep these fields because they are the source of truth for the user snapshot it sends:

- `user_id`
- `full_name`
- `designation`
- `expertise_areas`
- `experience_languages`
- `experience_years`
- `tone_preference`
- `template_id`
- `template_type`
- `template_text`

For built-in templates:
- `template_id` and `template_text` should be sourced from `GET /api/v1/templates`

For custom templates:
- Full Stack should store the final custom template text in its own users table and send that exact text in generate

Useful optional fields:

- `portfolio_last_synced_at`
- `chrome_extension_connected`
- `preferred_platform`
- standard app fields such as:
  - `email`
  - `phoneNumber`
  - `username`

## What Full Stack Sends To AI Dev For Pinecone Upsert

- `user_id`
- `projects`
- optional `scraped_profile_text`

That is enough for the AI dev backend to upsert project vectors into Pinecone using `user_id` as the namespace/filter.

## What Full Stack Sends To AI Dev For Proposal Generation

- `user_id`
- `user_profile`
- `template`
- `job_details`

Template payload rule:
- `template.template_text` is always required in generate
- if the template is built-in, Full Stack gets that text from `GET /api/v1/templates`
- if the template is custom or AI-generated, Full Stack gets that text from its own stored user record
- if `template.template_type = provided`, then `template.template_id` should be one of the built-in ids returned by `GET /api/v1/templates`

## What AI Dev Backend Stores In `Users-Proposals`

- `thread_id`
- `user_id`
- `user_profile_snapshot`
- `template_snapshot`
- `job_details`
- `proposals`
- `selected_proposal_id`
- `messages`
- `summary`
- `last_retriever_tool_message`
- `latest_response_type`

## What AI Dev Backend Stores In `Proposal-Tasks`

- `task_id`
- `thread_id`
- `status`
- `result`
- `error_message`

## Notes For Full Stack

- Full Stack should treat `user_profile` and `template` sent to generate as a snapshot of the current user state.
- AI dev backend stores that snapshot with the thread so optimize does not need to ask Full Stack for the users table again.
- built-in template ids and full text should be fetched from `GET /api/v1/templates`
- Proposal ids are always:
  - `alt_1`
  - `alt_2`
  - `alt_3`
- If retriever does not find accepted evidence, the AI may still generate proposals using only confirmed user profile and job context.
- AI dev backend is designed to avoid claiming unsupported skills or project details.
