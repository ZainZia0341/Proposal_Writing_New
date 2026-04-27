from __future__ import annotations

import json
from typing import Any, Literal
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command

from .bid_examples import format_bid_example_markdown
from .config import settings
from .llm import get_model_used, invoke_json, invoke_text, reset_llm_request_state
from .logging_utils import get_logger
from .repositories import get_proposals_repository
from .schemas import (
    BidExampleDraftRecord,
    BidExampleDraftResponse,
    BidExampleInput,
    ConversationMessage,
    FullStackUserProfile,
    GenerateProposalResponse,
    MessageRole,
    OptimizeProposalResponse,
    ProposalOption,
    ProposalThreadRecord,
    ResponseType,
    RetrieverToolMessage,
    StoredBidExample,
    TaskStatus,
)
from .vector_store import get_project_store

logger = get_logger(__name__)
BID_EXAMPLE_REFUSAL_TEXT = "I can only generate an example bid i can not help you with that"


class ProposalState(MessagesState):
    mode: str
    task_id: str
    user_id: str
    thread_id: str
    bid_examples_markdown: list[str]
    user_profile: dict[str, Any]
    job_details: dict[str, Any]
    thread_record: dict[str, Any] | None
    selected_proposal_id: str | None
    feedback_msg: str | None
    pinned_context: str
    summary: str | None
    retrieval_attempt: int
    vector_query: str | None
    retrieved_projects: list[dict[str, Any]]
    retrieval_used: bool
    retrieval_accepted: bool
    fallback_used: bool
    last_retriever_tool_message: dict[str, Any] | None
    response_type: str | None
    direct_answer: str | None
    direct_answer_kind: str | None
    updated_proposal: str | None
    proposals: list[dict[str, Any]]
    route_decision: str | None
    route_reason: str | None
    model_used: str | None


def _debug_print(title: str, payload: Any | None = None) -> None:
    print(f"\n============= {title} =============", flush=True)
    if payload is not None:
        if isinstance(payload, str):
            print(payload, flush=True)
        else:
            try:
                print(json.dumps(payload, indent=2, ensure_ascii=False, default=str), flush=True)
            except TypeError:
                print(str(payload), flush=True)
    print("========================================\n", flush=True)


def _debug_node_start(node_name: str, state: ProposalState) -> None:
    _debug_print(
        f"starting node: {node_name}",
        {
            "mode": state.get("mode"),
            "thread_id": state.get("thread_id"),
            "task_id": state.get("task_id"),
            "retrieval_attempt": state.get("retrieval_attempt"),
            "selected_proposal_id": state.get("selected_proposal_id"),
            "messages_count": len(state.get("messages", [])),
            "retrieval_used": state.get("retrieval_used"),
            "retrieval_accepted": state.get("retrieval_accepted"),
            "fallback_used": state.get("fallback_used"),
            "route_decision": state.get("route_decision"),
            "bid_examples_count": len(state.get("bid_examples_markdown", [])),
            "model_used": state.get("model_used"),
        },
    )


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _human_message(content: str) -> HumanMessage:
    return HumanMessage(content=content, id=f"human_{uuid4().hex}")


def _ai_message(content: str) -> AIMessage:
    return AIMessage(content=content, id=f"ai_{uuid4().hex}")


def _schema_to_langchain(message: ConversationMessage, index: int) -> BaseMessage:
    message_id = f"{message.role.value}_{index}_{message.created_at}"
    if message.role == MessageRole.USER:
        return HumanMessage(content=message.content, id=message_id)
    if message.role == MessageRole.ASSISTANT:
        return AIMessage(content=message.content, id=message_id)
    if message.role == MessageRole.TOOL:
        return ToolMessage(content=message.content, tool_call_id=message_id, id=message_id)
    return SystemMessage(content=message.content, id=message_id)


def _langchain_to_schema(message: BaseMessage) -> ConversationMessage | None:
    content = _message_text(message.content)
    if isinstance(message, HumanMessage):
        return ConversationMessage(role=MessageRole.USER, content=content)
    if isinstance(message, AIMessage):
        return ConversationMessage(role=MessageRole.ASSISTANT, content=content)
    if isinstance(message, ToolMessage):
        return ConversationMessage(role=MessageRole.TOOL, content=content)
    if isinstance(message, SystemMessage):
        return ConversationMessage(role=MessageRole.SYSTEM, content=content)
    return None


def _messages_to_schema(messages: list[BaseMessage]) -> list[ConversationMessage]:
    converted: list[ConversationMessage] = []
    for message in messages:
        schema_message = _langchain_to_schema(message)
        if schema_message is not None:
            converted.append(schema_message)
    return converted


def _format_project(project: dict[str, Any]) -> str:
    return (
        f"Project ID: {project['project_id']}\n"
        f"Title: {project['title']}\n"
        f"Role: {project['role']}\n"
        f"Description: {project['description']}\n"
        f"Tech Stack: {', '.join(project.get('tech_stack', []))}"
    )


def _trim_for_prompt(messages: list[BaseMessage]) -> list[BaseMessage]:
    if not messages:
        return []
    return trim_messages(
        messages,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=1800,
        start_on="human",
        include_system=True,
        allow_partial=False,
    )


def _messages_to_text(messages: list[BaseMessage]) -> str:
    trimmed = _trim_for_prompt(messages)
    lines = []
    for message in trimmed:
        role = message.type.upper()
        lines.append(f"{role}: {_message_text(message.content)}")
    return "\n".join(lines)


def _build_selected_proposal(state: ProposalState) -> dict[str, Any] | None:
    thread_record = state.get("thread_record") or {}
    selected_id = state.get("selected_proposal_id") or thread_record.get("selected_proposal_id")
    if not selected_id:
        return None
    return next(
        (proposal for proposal in thread_record.get("proposals", []) if proposal["id"] == selected_id),
        None,
    )


def _build_pinned_context(state: ProposalState, summary: str | None = None) -> str:
    user = state.get("user_profile") or {}
    job = state["job_details"]
    selected_proposal = _build_selected_proposal(state)
    lines = [
        f"User Name: {user.get('full_name', 'Not provided')}",
        f"Designation: {user.get('designation', 'Not provided')}",
        f"Expertise: {', '.join(user.get('expertise_areas', []))}",
        f"Languages: {', '.join(user.get('experience_languages', []))}",
        f"Tone Preference: {user.get('tone_preference') or 'Not provided'}",
        f"Job Title: {job['title']}",
        f"Job Description: {job['description']}",
        f"Budget: {job.get('budget') or 'Not specified'}",
        f"Skills: {', '.join(job.get('required_skills', []))}",
        f"Client Info: {job.get('client_info') or 'Not provided'}",
    ]
    current_summary = summary if summary is not None else state.get("summary")
    if current_summary:
        lines.append(f"Conversation Summary: {current_summary}")
    if selected_proposal:
        lines.append(f"Selected Proposal ID: {selected_proposal['id']}")
        lines.append(f"Base Proposal Text: {selected_proposal['text']}")
    return "\n".join(lines)


def _bid_examples_to_text(state: ProposalState) -> str:
    bid_examples = state.get("bid_examples_markdown", [])
    if not bid_examples:
        return "No stored previous bid examples were provided for this user."
    rendered = []
    for index, example in enumerate(bid_examples, start=1):
        rendered.append(f"### Previous Bid Example {index}\n{example}")
    return "\n\n---\n\n".join(rendered)


def _summarize_messages(messages: list[BaseMessage], existing_summary: str | None = None) -> str:
    conversation_text = _messages_to_text(messages)
    if existing_summary:
        conversation_text = f"Existing summary:\n{existing_summary}\n\nAdditional messages:\n{conversation_text}"
    fallback = conversation_text[:1200]
    return invoke_text(
        system_prompt=(
            "Summarize proposal-chat history. Keep confirmed facts, user preferences, proposal changes, "
            "and any relevant retriever outcomes. Be concise."
        ),
        user_prompt=conversation_text,
        fallback=fallback,
    )


def _route_feedback_with_llm(state: ProposalState) -> dict[str, Any]:
    selected_proposal = _build_selected_proposal(state)
    fallback = {
        "route": "retrieve_then_revise",
        "answer_kind": "generic",
        "reason": "Fallback route used because structured LLM routing was unavailable.",
    }
    return invoke_json(
        system_prompt=(
            "You route proposal optimization requests. Return JSON with keys route, answer_kind, and reason. "
            "Allowed routes: direct_answer, revise_only, retrieve_then_revise. "
            "Choose direct_answer only when the answer can be produced entirely from stored job details or selected proposal "
            "without changing the proposal text. "
            "Choose revise_only when the proposal should be changed using only existing pinned context. "
            "Choose retrieve_then_revise when extra project history is needed from the retriever. "
            "Allowed answer_kind values: budget, title, client_info, job_description, skills, selected_proposal, generic."
        ),
        user_prompt=json.dumps(
            {
                "feedback_msg": state.get("feedback_msg"),
                "job_details": state["job_details"],
                "selected_proposal": selected_proposal,
                "summary": state.get("summary"),
            },
            indent=2,
        ),
        fallback=fallback,
    )


def _plan_vector_query_with_llm(state: ProposalState) -> str:
    fallback = {
        "query": " | ".join(
            part
            for part in [
                state["job_details"]["title"],
                state["job_details"]["description"],
                ", ".join(state["job_details"].get("required_skills", [])),
                state.get("feedback_msg") or "",
            ]
            if part
        )
    }
    payload = invoke_json(
        system_prompt=(
            "You create concise vector search queries for retrieving relevant portfolio projects. "
            "Return JSON with one key named query."
        ),
        user_prompt=json.dumps(
            {
                "job_details": state["job_details"],
                "feedback_msg": state.get("feedback_msg"),
                "pinned_context": state["pinned_context"],
            },
            indent=2,
        ),
        fallback=fallback,
    )
    return str(payload.get("query", fallback["query"])).strip()


def _verify_retrieval_with_llm(state: ProposalState) -> dict[str, Any]:
    fallback = {
        "accepted": False,
        "rationale": "Fallback verifier could not confirm relevance.",
    }
    return invoke_json(
        system_prompt=(
            "You verify whether retrieved projects are relevant enough to use for proposal writing. "
            "Return JSON with accepted (boolean) and rationale (string). "
            "Accept only if the projects clearly support the job description or user feedback request. "
            "Reject projects that look related only because they share broad keywords from the job description. "
            "Reject projects if they do not give trustworthy evidence for the user's real worked-on skills, role, or domain experience. "
            "Be strict. It is better to reject weak matches than to allow unsupported claims into a proposal."
        ),
        user_prompt=json.dumps(
            {
                "job_details": state["job_details"],
                "feedback_msg": state.get("feedback_msg"),
                "retrieved_projects": state.get("retrieved_projects", []),
            },
            indent=2,
        ),
        fallback=fallback,
    )


def _fallback_generation(state: ProposalState) -> list[ProposalOption]:
    user = state["user_profile"]
    job = state["job_details"]
    supported_skills = user.get("experience_languages", [])
    requested_skills = job.get("required_skills", [])
    skills = ", ".join(supported_skills or requested_skills)
    project_lines = []
    for project in state.get("retrieved_projects", [])[:2]:
        project_lines.append(
            f"{project['title']} - {project['description']} ({', '.join(project.get('tech_stack', []))})"
        )
    experience_section = f"Relevant experience:\n{chr(10).join(project_lines)}\n\n" if project_lines else ""
    skills_sentence = f" with hands-on experience in {skills}" if supported_skills else ""
    requested_skills_sentence = (
        f" Your listed stack includes {', '.join(requested_skills)}, so I would keep the implementation aligned with those requirements."
        if requested_skills
        else ""
    )
    variants = [
        ProposalOption(
            id="alt_1",
            label="Balanced",
            text=(
                f"Hi, this sounds like a strong fit.\n\n"
                f"I'm {user['full_name']}, a {user['designation']}{skills_sentence}. "
                f"Your job around {job['title']} is the kind of work where a clear technical plan and careful delivery matter.{requested_skills_sentence}\n\n"
                f"{experience_section}"
                f"I can help you deliver this with a practical and reliable implementation. Would you like me to outline the execution plan?\n\n"
                f"Best,\n{user['full_name']}"
            ),
        ),
        ProposalOption(
            id="alt_2",
            label="Consultative",
            text=(
                f"Hi, this aligns closely with the type of structured technical work I focus on.\n\n"
                f"The challenge in {job['title']} is not only shipping features, but making sure the solution is maintainable and aligned with the requirements. "
                f"I would start by clarifying the core workflow, integration points, and success criteria before implementation.{requested_skills_sentence}\n\n"
                f"{experience_section}"
                f"If helpful, I can break this into milestones and recommend a clean technical approach.\n\n"
                f"Best,\n{user['full_name']}"
            ),
        ),
        ProposalOption(
            id="alt_3",
            label="Fast Mover",
            text=(
                f"I can definitely help here.\n\n"
                f"My background as a {user['designation']} helps me bridge product goals with clean implementation on a project like {job['title']}.{requested_skills_sentence}\n\n"
                f"{experience_section}"
                f"If you want, I can start with the highest-impact deliverable first and keep the scope tight.\n\n"
                f"Best,\n{user['full_name']}"
            ),
        ),
    ]
    return variants


def _normalize_proposal_ids(proposals: list[ProposalOption]) -> list[ProposalOption]:
    normalized: list[ProposalOption] = []
    for index, proposal in enumerate(proposals[:3], start=1):
        normalized.append(proposal.model_copy(update={"id": f"alt_{index}"}))
    return normalized


def _generate_proposals_with_llm(state: ProposalState) -> list[ProposalOption]:
    fallback = {"alternatives": [proposal.model_dump(mode="json") for proposal in _fallback_generation(state)]}
    projects_text = "\n\n".join(_format_project(project) for project in state.get("retrieved_projects", []))
    bid_examples_text = _bid_examples_to_text(state)
    payload = invoke_json(
        system_prompt=(
            "You write three distinct freelance proposal alternatives. "
            "Return JSON with key alternatives, an array of objects containing id, label, and text. "
            "The ids must be alt_1, alt_2, and alt_3. "
            "You will receive previous bid examples written by the user. Study all of them together and infer the user's natural hook, tone, CTA style, pacing, and structure. "
            "Choose the hook and style that best fit the current job. "
            "Use those examples only as writing-style references, never as factual evidence for the current proposal. "
            "Stay strictly grounded in the provided user profile, accepted retrieved projects, and job details. "
            "Do not mention any skill, tool, framework, certification, domain experience, or project detail unless it is explicitly supported by the provided context. "
            "Do not let job-description keywords push you into claiming experience the user does not actually have. "
            "Never copy old client names, job facts, budgets, or project claims from the previous bid examples into the new proposal. "
            "You may adapt the user's style naturally, but you must remain within the factual boundaries of the provided context. "
            "Never invent evidence, never overclaim, and never add unsupported project names."
        ),
        user_prompt=(
            f"Pinned context:\n{state['pinned_context']}\n\n"
            f"Previous bid examples:\n{bid_examples_text}\n\n"
            f"Recent messages:\n{_messages_to_text(state['messages'])}\n\n"
            f"Accepted retrieved projects:\n{projects_text or 'No accepted retrieved projects. Use only confirmed user profile and job context. Do not mention any project details.'}"
        ),
        fallback=fallback,
    )
    proposals = [ProposalOption.model_validate(item) for item in payload.get("alternatives", fallback["alternatives"])[:3]]
    return _normalize_proposal_ids(proposals)


def _generate_fallback_proposals_with_llm(state: ProposalState) -> list[ProposalOption]:
    fallback = {"alternatives": [proposal.model_dump(mode="json") for proposal in _fallback_generation(state)]}
    bid_examples_text = _bid_examples_to_text(state)
    payload = invoke_json(
        system_prompt=(
            "You write three distinct freelance proposal alternatives when portfolio retrieval did not return "
            "trusted supporting projects. Return JSON with key alternatives, an array of objects containing "
            "id, label, and text. The ids must be alt_1, alt_2, and alt_3. "
            "You will receive previous bid examples written by the user. Study all of them together and infer the user's likely hook, tone, CTA style, pacing, and structure for the current job. "
            "Use those examples only as style references. "
            "Use only the provided user profile, job details, and recent conversation as factual grounding. "
            "Do not mention any project name, project achievement, or project detail because retrieval did not produce accepted evidence. "
            "Do not add empty experience headings, portfolio placeholders, or phrases like 'Relevant portfolio examples available on request'. "
            "If there are no accepted retrieved projects, omit portfolio and relevant-experience sections entirely. "
            "Do not mention any skill, tool, framework, certification, or domain experience unless it is explicitly supported by the provided user profile. "
            "Do not let job-description keywords influence you into adding unsupported skills. "
            "Never copy old client names, job facts, budgets, or project claims from the previous bid examples. "
            "You may write naturally and creatively, but you must stay strictly inside the factual boundaries of the provided context."
        ),
        user_prompt=(
            f"Pinned context:\n{state['pinned_context']}\n\n"
            f"Previous bid examples:\n{bid_examples_text}\n\n"
            f"Recent messages:\n{_messages_to_text(state['messages'])}\n\n"
            "Retriever status:\n"
            "No accepted retrieved projects were available after retry attempts. "
            "Generate strong proposals from the user's real skills, designation, previous writing style, and the current job details only. "
            "If a detail is not explicitly supported, leave it out."
        ),
        fallback=fallback,
    )
    proposals = [ProposalOption.model_validate(item) for item in payload.get("alternatives", fallback["alternatives"])[:3]]
    return _normalize_proposal_ids(proposals)


def _revise_proposal_with_llm(state: ProposalState, base_text: str) -> str:
    fallback = (
        f"{base_text}\n\n"
        f"Update requested: {state.get('feedback_msg')}\n"
        "This draft was updated using the existing proposal context."
    )
    projects_text = "\n\n".join(_format_project(project) for project in state.get("retrieved_projects", []))
    bid_examples_text = _bid_examples_to_text(state)
    return invoke_text(
        system_prompt=(
            "You revise an existing freelance proposal. Preserve the strongest parts of the original, "
            "apply the feedback exactly, and avoid inventing facts. "
            "You will receive previous bid examples written by the user. Study all of them together and infer the user's natural hook, tone, CTA style, pacing, and structure. "
            "Use those examples only as writing-style references if they help you make the revision sound more like the user. "
            "Do not add any skill, tool, framework, certification, industry background, or project detail unless it is explicitly supported by the provided context. "
            "Do not let the job description or feedback wording push you into claiming experience the user has not actually shown. "
            "If there are no accepted retrieved projects, do not add project-specific claims. "
            "Never copy old client names, job facts, budgets, or project claims from the previous bid examples. "
            "You may improve the writing while staying strictly within factual boundaries."
        ),
        user_prompt=(
            f"Pinned context:\n{state['pinned_context']}\n\n"
            f"Previous bid examples:\n{bid_examples_text}\n\n"
            f"Recent messages:\n{_messages_to_text(state['messages'])}\n\n"
            f"Selected proposal:\n{base_text}\n\n"
            f"Accepted retrieved projects:\n{projects_text or 'No accepted retrieved projects. Do not mention project-specific details.'}"
        ),
        fallback=fallback,
    )


def _build_direct_answer_text(state: ProposalState) -> str:
    job = state["job_details"]
    selected_proposal = _build_selected_proposal(state)
    answer_kind = state.get("direct_answer_kind") or "generic"
    if answer_kind == "budget":
        return f"The stored budget for this job is {job.get('budget') or 'not specified'}."
    if answer_kind == "title":
        return f"The stored job title is '{job['title']}'."
    if answer_kind == "client_info":
        return f"The stored client information is {job.get('client_info') or 'not provided'}."
    if answer_kind == "job_description":
        return f"The stored job description is: {job['description']}"
    if answer_kind == "skills":
        return "The stored required skills are " + ", ".join(job.get("required_skills", [])) + "."
    if answer_kind == "selected_proposal" and selected_proposal:
        return f"The currently selected proposal is {selected_proposal['id']}:\n\n{selected_proposal['text']}"
    return (
        "Here are the stored job details:\n"
        f"{json.dumps(job, indent=2)}"
    )


def _proposal_message_text(proposals: list[ProposalOption]) -> str:
    return json.dumps([proposal.model_dump(mode="json") for proposal in proposals], indent=2)


def _resolve_model_used(state: ProposalState) -> str | None:
    return get_model_used() or state.get("model_used")


def _fallback_bid_example(user_profile: dict[str, Any], feedback_msg: str | None = None) -> StoredBidExample:
    skills = user_profile.get("experience_languages") or user_profile.get("expertise_areas") or ["relevant technologies"]
    skill_text = ", ".join(skills)
    name = user_profile.get("full_name", "there")
    designation = user_profile.get("designation", "Freelance Developer")
    title = f"{designation} needed for a practical implementation project"
    description = (
        f"We need an experienced {designation} to help design and implement a reliable solution using {skill_text}. "
        "The work includes understanding requirements, proposing a clean approach, and delivering maintainable results."
    )
    if feedback_msg:
        description += f" The sample should reflect this preference: {feedback_msg}"
    proposal_text = (
        "Hi, this sounds like a strong fit.\n\n"
        f"I'm {name}, a {designation} with hands-on experience in {skill_text}. "
        "I can help turn your requirements into a clean, practical implementation while keeping the workflow easy to maintain.\n\n"
        "I would start by clarifying the main success criteria, then break the work into focused milestones so you can review progress quickly.\n\n"
        "Would you like me to share a short implementation plan for this?\n\n"
        f"Best,\n{name}"
    )
    bid = BidExampleInput.model_validate(
        {
            "job_details": {
                "title": title,
                "description": description,
                "budget": "To be discussed",
                "required_skills": skills,
                "client_info": "Sample client",
            },
            "proposal_text": proposal_text,
        }
    )
    return StoredBidExample(
        job_details=bid.job_details,
        proposal_text=bid.proposal_text,
        markdown=format_bid_example_markdown(bid),
    )


def _stored_bid_example_from_payload(payload: dict[str, Any]) -> StoredBidExample:
    bid = BidExampleInput.model_validate(payload)
    return StoredBidExample(
        job_details=bid.job_details,
        proposal_text=bid.proposal_text,
        markdown=format_bid_example_markdown(bid),
    )


def _generate_bid_example_with_llm(
    *,
    user_profile: dict[str, Any],
    feedback_msg: str | None,
    existing_example_bid: dict[str, Any] | None = None,
    recent_messages: list[BaseMessage] | None = None,
) -> tuple[StoredBidExample | None, str | None]:
    fallback_bid = _fallback_bid_example(user_profile, feedback_msg)
    fallback = {
        "is_bid_example_request": True,
        "direct_answer": None,
        "example_bid": fallback_bid.model_dump(mode="json"),
    }
    payload = invoke_json(
        system_prompt=(
            "You are an assistant that only creates or edits editable example freelance bids. "
            "Return JSON with keys is_bid_example_request, direct_answer, and example_bid. "
            "A bid example means a sample job description/details plus the proposal text the freelancer could send. "
            "Use the user's profile, experience languages, expertise, tone preference, and feedback to create or revise the example. "
            "If the user's request is not about generating or editing an example bid, set is_bid_example_request=false, "
            f"set direct_answer exactly to: {BID_EXAMPLE_REFUSAL_TEXT}, and set example_bid=null. "
            "Do not use keyword rules; judge the user's intent from the full request. "
            "When returning example_bid, it must contain job_details and proposal_text. "
            "Do not invent unsupported credentials, companies, client names, or project claims."
        ),
        user_prompt=json.dumps(
            {
                "user_profile": user_profile,
                "feedback_msg": feedback_msg,
                "existing_example_bid": existing_example_bid,
                "recent_messages": _messages_to_text(recent_messages or []),
            },
            indent=2,
        ),
        fallback=fallback,
    )
    if not bool(payload.get("is_bid_example_request", True)):
        return None, BID_EXAMPLE_REFUSAL_TEXT

    example_payload = payload.get("example_bid") or fallback["example_bid"]
    return _stored_bid_example_from_payload(example_payload), None


def run_bid_example_flow(task_id: str, payload: dict[str, Any]) -> BidExampleDraftResponse:
    reset_llm_request_state()
    proposals_repo = get_proposals_repository()
    thread_id = payload["thread_id"]
    user_id = payload["user_id"]
    existing = proposals_repo.get_bid_example_draft(user_id, thread_id)

    if existing:
        user_profile = existing.user_profile_snapshot.model_dump(mode="json")
        messages = [_schema_to_langchain(message, index) for index, message in enumerate(existing.messages)]
        summary = existing.summary
        existing_example_bid = existing.example_bid.model_dump(mode="json") if existing.example_bid else None
    else:
        user_profile = payload.get("user_profile") or {}
        if not user_profile:
            raise ValueError("user_profile is required when creating a new bid example draft.")
        messages = []
        summary = None
        existing_example_bid = None

    feedback_msg = payload.get("feedback_msg")
    user_message = feedback_msg or "Generate an editable example bid from this user profile."
    messages.append(_human_message(user_message))

    example_bid, direct_answer = _generate_bid_example_with_llm(
        user_profile=user_profile,
        feedback_msg=feedback_msg,
        existing_example_bid=existing_example_bid,
        recent_messages=messages,
    )

    response_text = direct_answer or (example_bid.markdown if example_bid else BID_EXAMPLE_REFUSAL_TEXT)
    messages.append(_ai_message(response_text))
    if len(messages) > settings.summary_trigger_messages:
        older_messages = messages[:-settings.recent_messages_to_keep]
        if older_messages:
            summary = _summarize_messages(older_messages, summary)
            messages = messages[-settings.recent_messages_to_keep :]

    final_example_bid = example_bid or (existing.example_bid if existing else None)
    if example_bid or existing:
        record = BidExampleDraftRecord(
            thread_id=thread_id,
            user_id=user_id,
            user_profile_snapshot=FullStackUserProfile.model_validate(user_profile),
            example_bid=final_example_bid,
            messages=_messages_to_schema(messages),
            summary=summary,
            status=TaskStatus.COMPLETED,
        )
        proposals_repo.upsert_bid_example_draft(record)

    logger.info(
        "Completed bid example draft flow",
        extra={"thread_id": thread_id, "user_id": user_id, "refused": direct_answer is not None},
    )
    return BidExampleDraftResponse(
        thread_id=thread_id,
        task_id=task_id,
        status=TaskStatus.COMPLETED,
        example_bid=final_example_bid,
        direct_answer=direct_answer,
        summary=summary,
        model_used=get_model_used(),
    )


def initialize_context(
    state: ProposalState,
) -> Command[Literal["summarize_history", "route_feedback", "plan_query"]]:
    _debug_node_start("initialize_context", state)
    proposals_repo = get_proposals_repository()
    thread_record = None
    if state.get("thread_id"):
        existing = proposals_repo.get(state["user_id"], state["thread_id"])
        if existing:
            thread_record = existing.model_dump(mode="json")

    if state["mode"] == "generate":
        user_profile = state.get("user_profile") or {}
        if not user_profile:
            raise ValueError("Generate flow requires full stack user_profile in the request payload.")
        bid_style_record = proposals_repo.get_bid_style(state["user_id"])
    else:
        if thread_record is None:
            raise ValueError(f"Thread '{state['thread_id']}' was not found.")
        user_profile = thread_record.get("user_profile_snapshot") or {}
        if not user_profile:
            raise ValueError("Thread is missing stored full stack user_profile context. Regenerate the thread with user_profile.")
        bid_style_record = proposals_repo.get_bid_style(thread_record.get("user_id") or state["user_id"])

    updates: dict[str, Any] = {
        "user_profile": user_profile,
        "thread_record": thread_record,
        "bid_examples_markdown": [bid.markdown for bid in bid_style_record.bids] if bid_style_record else [],
        "retrieval_attempt": 0,
        "retrieval_used": False,
        "retrieval_accepted": False,
        "fallback_used": False,
        "last_retriever_tool_message": None,
        "response_type": None,
        "direct_answer": None,
        "direct_answer_kind": None,
        "updated_proposal": None,
        "proposals": [],
        "route_decision": None,
        "route_reason": None,
        "model_used": state.get("model_used"),
    }
    updates["pinned_context"] = _build_pinned_context({**state, **updates})

    should_summarize = len(state["messages"]) > settings.summary_trigger_messages
    next_node: Literal["summarize_history", "route_feedback", "plan_query"]
    if should_summarize:
        next_node = "summarize_history"
    elif state["mode"] == "optimize":
        next_node = "route_feedback"
    else:
        next_node = "plan_query"

    logger.info(
        "Initialized graph context",
        extra={
            "thread_id": state["thread_id"],
            "mode": state["mode"],
            "next_node": next_node,
            "bid_examples_count": len(updates["bid_examples_markdown"]),
        },
    )
    return Command(update=updates, goto=next_node)


def summarize_history(
    state: ProposalState,
) -> Command[Literal["route_feedback", "plan_query"]]:
    _debug_node_start("summarize_history", state)
    older_messages = state["messages"][:-settings.recent_messages_to_keep]
    if not older_messages:
        next_node: Literal["route_feedback", "plan_query"] = "route_feedback" if state["mode"] == "optimize" else "plan_query"
        return Command(goto=next_node)

    summary = _summarize_messages(older_messages, state.get("summary"))
    removals = [RemoveMessage(id=message.id) for message in older_messages if getattr(message, "id", None)]
    next_node = "route_feedback" if state["mode"] == "optimize" else "plan_query"
    updates: dict[str, Any] = {
        "summary": summary,
        "messages": removals,
        "model_used": _resolve_model_used(state),
    }
    updates["pinned_context"] = _build_pinned_context({**state, **updates}, summary=summary)
    logger.info(
        "Summarized conversation history",
        extra={"thread_id": state["thread_id"], "removed_messages": len(removals), "next_node": next_node},
    )
    return Command(update=updates, goto=next_node)


def route_feedback(
    state: ProposalState,
) -> Command[Literal["answer_direct_from_context", "revise_selected_proposal", "plan_query"]]:
    _debug_node_start("route_feedback", state)
    decision = _route_feedback_with_llm(state)
    _debug_print("route decision output", decision)
    route = str(decision.get("route", "retrieve_then_revise"))
    answer_kind = str(decision.get("answer_kind", "generic"))
    reason = str(decision.get("reason", ""))
    updates = {
        "route_decision": route,
        "route_reason": reason,
        "direct_answer_kind": answer_kind,
        "model_used": _resolve_model_used(state),
    }
    if route == "direct_answer":
        updates["response_type"] = ResponseType.DIRECT_ANSWER.value
        next_node: Literal["answer_direct_from_context", "revise_selected_proposal", "plan_query"] = "answer_direct_from_context"
    elif route == "revise_only":
        updates["response_type"] = ResponseType.PROPOSAL_UPDATE.value
        next_node = "revise_selected_proposal"
    else:
        updates["response_type"] = ResponseType.PROPOSAL_UPDATE.value
        next_node = "plan_query"
    logger.info(
        "LLM routed optimization feedback",
        extra={"thread_id": state["thread_id"], "route": route, "reason": reason},
    )
    return Command(update=updates, goto=next_node)


def answer_direct_from_context(state: ProposalState) -> dict[str, Any]:
    _debug_node_start("answer_direct_from_context", state)
    answer = _build_direct_answer_text(state)
    logger.info("Generated direct answer from stored context", extra={"thread_id": state["thread_id"]})
    return {
        "direct_answer": answer,
        "messages": [_ai_message(answer)],
        "response_type": ResponseType.DIRECT_ANSWER.value,
    }


def plan_query(state: ProposalState) -> Command[Literal["retrieve_projects"]]:
    _debug_node_start("plan_query", state)
    query = _plan_vector_query_with_llm(state)
    attempt = int(state.get("retrieval_attempt", 0)) + 1
    _debug_print("planned vector query output", {"attempt": attempt, "query": query})
    logger.info(
        "Planned vector query",
        extra={"thread_id": state["thread_id"], "attempt": attempt, "query": query},
    )
    return Command(
        update={
            "vector_query": query,
            "retrieval_attempt": attempt,
            "model_used": _resolve_model_used(state),
        },
        goto="retrieve_projects",
    )


def retrieve_projects(state: ProposalState) -> dict[str, Any]:
    _debug_node_start("retrieve_projects", state)
    projects = get_project_store().search_projects(
        user_id=state["user_id"],
        query=state["vector_query"] or "",
        top_k=settings.retrieval_top_k,
    )
    _debug_print(
        "retriever output",
        {
            "attempt": state.get("retrieval_attempt"),
            "vector_query": state.get("vector_query"),
            "retrieved_projects": [project.model_dump(mode="json") for project in projects],
        },
    )
    logger.info(
        "Retrieved candidate projects",
        extra={"thread_id": state["thread_id"], "attempt": state["retrieval_attempt"], "count": len(projects)},
    )
    return {
        "retrieved_projects": [project.model_dump(mode="json") for project in projects],
        "retrieval_used": True,
    }


def verify_retrieval(
    state: ProposalState,
) -> Command[Literal["plan_query", "generate_proposals", "generate_fallback_proposals", "revise_selected_proposal"]]:
    _debug_node_start("verify_retrieval", state)
    verification = _verify_retrieval_with_llm(state)
    accepted = bool(verification.get("accepted", False))
    rationale = str(verification.get("rationale", ""))
    tool_message = RetrieverToolMessage(
        query=state["vector_query"] or "",
        matched_project_ids=[project["project_id"] for project in state.get("retrieved_projects", [])],
        matched_project_titles=[project["title"] for project in state.get("retrieved_projects", [])],
        accepted=accepted,
        rationale=rationale,
        attempt=state["retrieval_attempt"],
    )
    updates: dict[str, Any] = {
        "retrieval_accepted": accepted,
        "last_retriever_tool_message": tool_message.model_dump(mode="json"),
        "model_used": _resolve_model_used(state),
    }

    if accepted:
        next_node: Literal["plan_query", "generate_proposals", "generate_fallback_proposals", "revise_selected_proposal"]
        next_node = "revise_selected_proposal" if state["mode"] == "optimize" else "generate_proposals"
    elif state["retrieval_attempt"] < settings.retrieval_max_retries:
        next_node = "plan_query"
    else:
        updates["fallback_used"] = True
        updates["retrieved_projects"] = []
        next_node = "revise_selected_proposal" if state["mode"] == "optimize" else "generate_fallback_proposals"

    logger.info(
        "Verified retrieval results",
        extra={
            "thread_id": state["thread_id"],
            "attempt": state["retrieval_attempt"],
            "accepted": accepted,
            "next_node": next_node,
        },
    )
    _debug_print(
        "retrieval verification output",
        {
            "verification": verification,
            "accepted": accepted,
            "next_node": next_node,
            "tool_message": tool_message.model_dump(mode="json"),
        },
    )
    return Command(update=updates, goto=next_node)


def generate_proposals(state: ProposalState) -> dict[str, Any]:
    _debug_node_start("generate_proposals", state)
    proposals = _generate_proposals_with_llm(state)
    _debug_print(
        "proposal generation output",
        [proposal.model_dump(mode="json") for proposal in proposals],
    )
    logger.info("Generated proposal alternatives", extra={"thread_id": state["thread_id"], "count": len(proposals)})
    return {
        "proposals": [proposal.model_dump(mode="json") for proposal in proposals],
        "messages": [_ai_message(_proposal_message_text(proposals))],
        "response_type": ResponseType.PROPOSALS.value,
        "model_used": _resolve_model_used(state),
    }


def generate_fallback_proposals(state: ProposalState) -> dict[str, Any]:
    _debug_node_start("generate_fallback_proposals", state)
    proposals = _generate_fallback_proposals_with_llm(state)
    _debug_print(
        "fallback proposal generation output",
        [proposal.model_dump(mode="json") for proposal in proposals],
    )
    logger.info(
        "Generated fallback proposal alternatives with LLM context",
        extra={"thread_id": state["thread_id"], "count": len(proposals)},
    )
    return {
        "proposals": [proposal.model_dump(mode="json") for proposal in proposals],
        "messages": [_ai_message(_proposal_message_text(proposals))],
        "response_type": ResponseType.PROPOSALS.value,
        "fallback_used": True,
        "model_used": _resolve_model_used(state),
    }


def revise_selected_proposal(state: ProposalState) -> dict[str, Any]:
    _debug_node_start("revise_selected_proposal", state)
    selected_proposal = _build_selected_proposal(state)
    if selected_proposal is None:
        raise ValueError(f"Proposal '{state.get('selected_proposal_id')}' was not found in thread '{state['thread_id']}'.")
    updated_text = _revise_proposal_with_llm(state, selected_proposal["text"])
    _debug_print(
        "proposal revision output",
        {
            "selected_proposal_id": selected_proposal["id"],
            "updated_text": updated_text,
        },
    )
    logger.info("Revised selected proposal", extra={"thread_id": state["thread_id"], "proposal_id": selected_proposal["id"]})
    return {
        "updated_proposal": updated_text,
        "messages": [_ai_message(updated_text)],
        "response_type": ResponseType.PROPOSAL_UPDATE.value,
        "model_used": _resolve_model_used(state),
    }


def finalize_memory_window(state: ProposalState) -> Command[Literal["persist_result"]]:
    _debug_node_start("finalize_memory_window", state)
    if len(state["messages"]) <= settings.summary_trigger_messages:
        return Command(goto="persist_result")

    older_messages = state["messages"][:-settings.recent_messages_to_keep]
    if not older_messages:
        return Command(goto="persist_result")

    summary = _summarize_messages(older_messages, state.get("summary"))
    removals = [RemoveMessage(id=message.id) for message in older_messages if getattr(message, "id", None)]
    updates: dict[str, Any] = {
        "summary": summary,
        "messages": removals,
        "model_used": _resolve_model_used(state),
    }
    updates["pinned_context"] = _build_pinned_context({**state, **updates}, summary=summary)
    logger.info(
        "Applied final memory window before persistence",
        extra={"thread_id": state["thread_id"], "removed_messages": len(removals)},
    )
    return Command(update=updates, goto="persist_result")


def persist_result(state: ProposalState) -> dict[str, Any]:
    _debug_node_start("persist_result", state)
    proposals_repo = get_proposals_repository()
    existing_thread = proposals_repo.get(state["user_id"], state["thread_id"])
    stored_messages = _messages_to_schema(state["messages"])

    if state["response_type"] == ResponseType.PROPOSALS.value:
        record = ProposalThreadRecord(
            user_id=state["user_id"],
            thread_id=state["thread_id"],
            job_details=state["job_details"],
            user_profile_snapshot=FullStackUserProfile.model_validate(state["user_profile"]),
            template_snapshot=None,
            template_id=None,
            template_text=None,
            proposals=[ProposalOption.model_validate(item) for item in state.get("proposals", [])],
            selected_proposal_id=None,
            latest_response_type=ResponseType.PROPOSALS,
            messages=stored_messages,
            summary=state.get("summary"),
            last_retriever_tool_message=(
                RetrieverToolMessage.model_validate(state["last_retriever_tool_message"])
                if state.get("retrieval_accepted") and state.get("last_retriever_tool_message")
                else None
            ),
            status=TaskStatus.COMPLETED,
        )
    else:
        if existing_thread is None:
            raise ValueError(f"Thread '{state['thread_id']}' was not found for persistence.")

        updated_proposals = existing_thread.proposals
        if state["response_type"] == ResponseType.PROPOSAL_UPDATE.value:
            updated_proposals = []
            for proposal in existing_thread.proposals:
                if proposal.id == state.get("selected_proposal_id"):
                    updated_proposals.append(
                        ProposalOption(
                            id=proposal.id,
                            label=proposal.label,
                            text=state.get("updated_proposal") or proposal.text,
                        )
                    )
                else:
                    updated_proposals.append(proposal)

        record = existing_thread.model_copy(
            update={
                "user_profile_snapshot": existing_thread.user_profile_snapshot,
                "proposals": updated_proposals,
                "selected_proposal_id": state.get("selected_proposal_id") or existing_thread.selected_proposal_id,
                "latest_response_type": ResponseType(state["response_type"]),
                "messages": stored_messages,
                "summary": state.get("summary"),
                "last_retriever_tool_message": (
                    RetrieverToolMessage.model_validate(state["last_retriever_tool_message"])
                    if state.get("retrieval_accepted") and state.get("last_retriever_tool_message")
                    else existing_thread.last_retriever_tool_message
                ),
                "status": TaskStatus.COMPLETED,
            }
        )

    stored = proposals_repo.upsert(record)
    logger.info(
        "Persisted proposal thread state",
        extra={"thread_id": state["thread_id"], "response_type": state["response_type"]},
    )
    _debug_print(
        "persisted thread snapshot",
        {
            "thread_id": stored.thread_id,
            "latest_response_type": stored.latest_response_type.value,
            "summary": stored.summary,
            "selected_proposal_id": stored.selected_proposal_id,
            "last_retriever_tool_message": (
                stored.last_retriever_tool_message.model_dump(mode="json")
                if stored.last_retriever_tool_message
                else None
            ),
        },
    )
    return {
        "summary": stored.summary,
        "thread_record": stored.model_dump(mode="json"),
    }


builder = StateGraph(ProposalState)
builder.add_node("initialize_context", initialize_context)
builder.add_node("summarize_history", summarize_history)
builder.add_node("route_feedback", route_feedback)
builder.add_node("answer_direct_from_context", answer_direct_from_context)
builder.add_node("plan_query", plan_query)
builder.add_node("retrieve_projects", retrieve_projects)
builder.add_node("verify_retrieval", verify_retrieval)
builder.add_node("generate_proposals", generate_proposals)
builder.add_node("generate_fallback_proposals", generate_fallback_proposals)
builder.add_node("revise_selected_proposal", revise_selected_proposal)
builder.add_node("finalize_memory_window", finalize_memory_window)
builder.add_node("persist_result", persist_result)

builder.add_edge(START, "initialize_context")
builder.add_edge("answer_direct_from_context", "finalize_memory_window")
builder.add_edge("retrieve_projects", "verify_retrieval")
builder.add_edge("generate_proposals", "finalize_memory_window")
builder.add_edge("generate_fallback_proposals", "finalize_memory_window")
builder.add_edge("revise_selected_proposal", "finalize_memory_window")
builder.add_edge("persist_result", END)

proposal_graph = builder.compile()


def run_generate_flow(task_id: str, payload: dict[str, Any]) -> GenerateProposalResponse:
    reset_llm_request_state()
    existing_thread = get_proposals_repository().get(payload["user_id"], payload["thread_id"])

    messages = []
    summary = None
    if existing_thread:
        messages = [_schema_to_langchain(message, index) for index, message in enumerate(existing_thread.messages)]
        summary = existing_thread.summary

    messages.append(
        _human_message(
            "Generate three proposal alternatives for this job.\n\n"
            f"{json.dumps(payload['job_details'], indent=2)}"
        )
    )
    result = proposal_graph.invoke(
        {
            "mode": "generate",
            "task_id": task_id,
            "user_id": payload["user_id"],
            "thread_id": payload["thread_id"],
            "user_profile": payload["user_profile"],
            "job_details": payload["job_details"],
            "messages": messages,
            "summary": summary,
        }
    )
    return GenerateProposalResponse(
        thread_id=result["thread_id"],
        task_id=task_id,
        status=TaskStatus.COMPLETED,
        proposals=[ProposalOption.model_validate(item) for item in result.get("proposals", [])],
        retrieval_used=result.get("retrieval_used", False),
        fallback_used=result.get("fallback_used", False),
        retrieved_project_ids=[project["project_id"] for project in result.get("retrieved_projects", [])],
        summary=result.get("summary"),
        model_used=result.get("model_used"),
    )


def run_optimize_flow(task_id: str, payload: dict[str, Any]) -> OptimizeProposalResponse:
    reset_llm_request_state()
    thread = get_proposals_repository().get(payload["user_id"], payload["thread_id"])
    if thread is None:
        raise ValueError(f"Thread '{payload['thread_id']}' was not found.")

    messages = [_schema_to_langchain(message, index) for index, message in enumerate(thread.messages)]
    messages.append(_human_message(payload["feedback_msg"]))

    result = proposal_graph.invoke(
        {
            "mode": "optimize",
            "task_id": task_id,
            "user_id": payload["user_id"],
            "thread_id": thread.thread_id,
            "job_details": thread.job_details.model_dump(mode="json"),
            "selected_proposal_id": payload["selected_proposal_id"],
            "feedback_msg": payload["feedback_msg"],
            "messages": messages,
            "summary": thread.summary,
        }
    )
    return OptimizeProposalResponse(
        thread_id=result["thread_id"],
        task_id=task_id,
        status=TaskStatus.COMPLETED,
        response_type=ResponseType(result["response_type"]),
        updated_proposal=result.get("updated_proposal"),
        direct_answer=result.get("direct_answer"),
        retrieval_used=result.get("retrieval_used", False),
        fallback_used=result.get("fallback_used", False),
        selected_proposal_id=payload["selected_proposal_id"],
        summary=result.get("summary"),
        model_used=result.get("model_used"),
    )
