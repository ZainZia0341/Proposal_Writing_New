from __future__ import annotations

import json
import re
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.config import settings
from app.llm import invoke_json, invoke_text, llm_available
from app.repositories import get_proposals_repository, get_users_repository
from app.schemas import (
    ConversationMessage,
    GenerateProposalResponse,
    MessageRole,
    OptimizeProposalResponse,
    ProposalOption,
    ProposalThreadRecord,
    ResponseType,
    RetrieverToolMessage,
    TaskStatus,
)
from app.seed_data import DEFAULT_TEMPLATES
from app.vector_store import get_project_store


class ProposalState(TypedDict, total=False):
    mode: str
    task_id: str
    user_id: str
    thread_id: str
    template_id: str | None
    selected_proposal_id: str | None
    feedback_msg: str | None
    job_details: dict[str, Any]
    user_profile: dict[str, Any]
    template_text: str
    thread_record: dict[str, Any] | None
    pinned_context: str
    retrieval_attempt: int
    vector_query: str
    retrieved_projects: list[dict[str, Any]]
    retrieval_used: bool
    retrieval_accepted: bool
    fallback_used: bool
    should_retry: bool
    direct_answer: str | None
    response_type: str
    proposals: list[dict[str, Any]]
    updated_proposal: str | None
    summary: str | None
    last_retriever_tool_message: dict[str, Any] | None


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z0-9\.\+#]+", text.lower()) if len(token) > 1}


def _format_project(project: dict[str, Any]) -> str:
    tech_stack = ", ".join(project.get("tech_stack", []))
    return (
        f"Project ID: {project['project_id']}\n"
        f"Title: {project['title']}\n"
        f"Role: {project['role']}\n"
        f"Description: {project['description']}\n"
        f"Tech Stack: {tech_stack}"
    )


def _truncate_summary(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    if len(words) <= 120:
        return text
    return " ".join(words[:120]).strip()


def _summarize_messages(
    messages: list[ConversationMessage] | list[dict[str, Any]], existing_summary: str | None = None
) -> str:
    normalized_messages = [
        message if isinstance(message, ConversationMessage) else ConversationMessage.model_validate(message)
        for message in messages
    ]
    summary_input = "\n".join(f"{message.role.value}: {message.content}" for message in normalized_messages)
    if existing_summary:
        summary_input = f"Existing summary:\n{existing_summary}\n\nAdditional messages:\n{summary_input}"

    fallback = _truncate_summary(summary_input)
    if not llm_available():
        return fallback

    return invoke_text(
        system_prompt=(
            "Summarize proposal-chat history. Keep confirmed facts, user preferences, and proposal changes. Be concise."
        ),
        user_prompt=summary_input,
        fallback=fallback,
    )


def _apply_summary_window(
    messages: list[ConversationMessage], existing_summary: str | None = None
) -> tuple[list[ConversationMessage], str | None]:
    if len(messages) <= settings.summary_trigger_messages:
        return messages, existing_summary

    older_messages = messages[:-settings.recent_messages_to_keep]
    recent_messages = messages[-settings.recent_messages_to_keep :]
    summary = _summarize_messages(older_messages, existing_summary=existing_summary)
    return recent_messages, summary


def _build_summary(record: dict[str, Any] | None) -> str | None:
    if not record:
        return None
    messages = record.get("messages", [])
    if len(messages) <= settings.summary_trigger_messages:
        return record.get("summary")
    older_messages = messages[:-settings.recent_messages_to_keep]
    return _summarize_messages(older_messages, existing_summary=record.get("summary"))


def _build_pinned_context(state: ProposalState) -> str:
    user = state["user_profile"]
    job = state["job_details"]
    lines = [
        f"User Name: {user['full_name']}",
        f"Designation: {user['designation']}",
        f"Expertise: {', '.join(user.get('expertise_areas', []))}",
        f"Languages: {', '.join(user.get('experience_languages', []))}",
        f"Template ID: {state['template_id'] or user['template_id']}",
        f"Template Text: {state['template_text']}",
        f"Job Title: {job['title']}",
        f"Job Description: {job['description']}",
        f"Budget: {job.get('budget') or 'Not specified'}",
        f"Skills: {', '.join(job.get('required_skills', []))}",
        f"Client Info: {job.get('client_info') or 'Not provided'}",
    ]
    if state.get("summary"):
        lines.append(f"Conversation Summary: {state['summary']}")
    if state.get("thread_record"):
        selected_id = state.get("selected_proposal_id") or state["thread_record"].get("selected_proposal_id")
        if selected_id:
            lines.append(f"Selected Proposal ID: {selected_id}")
            selected_proposal = next(
                (
                    proposal
                    for proposal in state["thread_record"].get("proposals", [])
                    if proposal["id"] == selected_id
                ),
                None,
            )
            if selected_proposal:
                lines.append(f"Base Proposal Text: {selected_proposal['text']}")
    return "\n".join(lines)


def _heuristic_query(state: ProposalState) -> str:
    parts = [
        state["job_details"]["title"],
        state["job_details"]["description"],
        ", ".join(state["job_details"].get("required_skills", [])),
    ]
    if state["mode"] == "optimize" and state.get("feedback_msg"):
        parts.append(state["feedback_msg"] or "")
    return " | ".join(part for part in parts if part)


def _llm_query(state: ProposalState) -> str:
    fallback = {"query": _heuristic_query(state)}
    payload = invoke_json(
        system_prompt=(
            "You create concise retrieval queries for proposal generation. "
            "Return JSON with a single key named query."
        ),
        user_prompt=(
            "Create a project-search query from this proposal context:\n\n"
            f"{json.dumps({'job_details': state['job_details'], 'feedback_msg': state.get('feedback_msg')}, indent=2)}"
        ),
        fallback=fallback,
    )
    return str(payload.get("query", fallback["query"])).strip()


def _needs_direct_answer(feedback_msg: str) -> bool:
    direct_patterns = [
        "what was budget",
        "what is the budget",
        "what was title",
        "what is the title",
        "client info",
        "job title",
        "job description",
        "skills required",
    ]
    lowered = feedback_msg.lower()
    if any(pattern in lowered for pattern in direct_patterns):
        return True
    action_words = ["change", "rewrite", "update", "modify", "improve", "shorter", "longer", "replace"]
    return not any(word in lowered for word in action_words) and lowered.endswith("?")


def _requires_retrieval(state: ProposalState) -> bool:
    feedback = (state.get("feedback_msg") or "").lower()
    trigger_keywords = [
        "project",
        "experience",
        "aws",
        "similar",
        "portfolio",
        "justify",
        "example",
        "past work",
    ]
    if state["mode"] == "generate":
        return True
    return any(keyword in feedback for keyword in trigger_keywords)


def _verify_projects(state: ProposalState) -> tuple[bool, str]:
    projects = state.get("retrieved_projects", [])
    if not projects:
        return False, "No projects were retrieved for this user."

    query_tokens = _tokenize(
        " ".join(
            [
                state["job_details"]["title"],
                state["job_details"]["description"],
                " ".join(state["job_details"].get("required_skills", [])),
                state.get("feedback_msg") or "",
            ]
        )
    )

    overlaps = []
    for project in projects:
        project_tokens = _tokenize(
            " ".join(
                [
                    project["title"],
                    project["description"],
                    " ".join(project.get("tech_stack", [])),
                    project["role"],
                ]
            )
        )
        overlaps.append(len(query_tokens.intersection(project_tokens)))

    best_overlap = max(overlaps, default=0)
    if best_overlap >= 2:
        return True, f"Retrieved projects matched the job context with overlap score {best_overlap}."

    if llm_available():
        payload = invoke_json(
            system_prompt=(
                "You verify whether retrieved portfolio projects are relevant enough for proposal writing. "
                "Return JSON with accepted (boolean) and rationale (string)."
            ),
            user_prompt=(
                "Job context:\n"
                f"{json.dumps(state['job_details'], indent=2)}\n\n"
                "Retrieved projects:\n"
                f"{json.dumps(projects, indent=2)}"
            ),
            fallback={"accepted": False, "rationale": "Token overlap was too low for confident retrieval."},
        )
        return bool(payload.get("accepted", False)), str(payload.get("rationale", ""))

    return False, "Retrieved projects were too weakly related to the job description."


def _fallback_generation(state: ProposalState) -> list[ProposalOption]:
    user = state["user_profile"]
    job = state["job_details"]
    skills = ", ".join(job.get("required_skills", [])) or ", ".join(user.get("experience_languages", []))
    project_lines = []
    for project in state.get("retrieved_projects", [])[:2]:
        project_lines.append(
            f"{project['title']} - {project['description']} ({', '.join(project.get('tech_stack', []))})"
        )
    portfolio_section = "\n".join(project_lines) if project_lines else "Relevant portfolio examples available on request."

    variants = [
        (
            "alt_1",
            "Balanced",
            f"Hi, this sounds like a strong fit.\n\n"
            f"I'm {user['full_name']}, a {user['designation']} with hands-on experience in {skills}. "
            f"Your job around {job['title']} is closely aligned with the kind of systems I build.\n\n"
            f"Relevant experience:\n{portfolio_section}\n\n"
            f"I can help you deliver this with a practical and reliable implementation. Would you like me to outline the execution plan?\n\n"
            f"Best,\n{user['full_name']}",
        ),
        (
            "alt_2",
            "Consultative",
            f"Hi, this aligns closely with the type of work I've been doing recently.\n\n"
            f"The challenge in {job['title']} is not only shipping features, but making sure the solution is maintainable and aligned with {skills}. "
            f"I've handled similar delivery problems across AI and backend projects.\n\n"
            f"Relevant experience:\n{portfolio_section}\n\n"
            f"If helpful, I can break this into milestones and recommend a clean technical approach.\n\n"
            f"Best,\n{user['full_name']}",
        ),
        (
            "alt_3",
            "Fast Mover",
            f"I can definitely help here.\n\n"
            f"I've built similar systems using {skills} and can move quickly on a project like {job['title']}. "
            f"My background as a {user['designation']} helps me bridge product goals with clean implementation.\n\n"
            f"Relevant experience:\n{portfolio_section}\n\n"
            f"If you want, I can start with the highest-impact deliverable first and keep the scope tight.\n\n"
            f"Best,\n{user['full_name']}",
        ),
    ]
    return [ProposalOption(id=item[0], label=item[1], text=item[2]) for item in variants]


def _llm_generation(state: ProposalState) -> list[ProposalOption]:
    fallback = {
        "alternatives": [proposal.model_dump(mode="json") for proposal in _fallback_generation(state)]
    }
    projects_text = "\n\n".join(_format_project(project) for project in state.get("retrieved_projects", []))
    payload = invoke_json(
        system_prompt=(
            "You write three distinct high-converting freelance proposals. "
            "Return JSON with key alternatives, an array of objects containing id, label, and text. "
            "The ids must be alt_1, alt_2, and alt_3."
        ),
        user_prompt=(
            f"Pinned context:\n{state['pinned_context']}\n\n"
            f"Retrieved projects:\n{projects_text or 'No strong retrieved projects. Use core skills only.'}\n\n"
            "Write three proposal alternatives. Each should be distinct, concise, and specific."
        ),
        fallback=fallback,
    )
    alternatives = payload.get("alternatives", fallback["alternatives"])
    return [ProposalOption.model_validate(item) for item in alternatives[:3]]


def _llm_revision(state: ProposalState, base_text: str) -> str:
    fallback = (
        f"{base_text}\n\n"
        f"Update requested: {state['feedback_msg']}\n"
        "This draft was updated to reflect the latest feedback while preserving the original intent."
    )
    projects_text = "\n\n".join(_format_project(project) for project in state.get("retrieved_projects", []))
    return invoke_text(
        system_prompt=(
            "You revise an existing freelance proposal. "
            "Preserve the strongest parts of the original, apply the feedback exactly, and avoid inventing facts."
        ),
        user_prompt=(
            f"Pinned context:\n{state['pinned_context']}\n\n"
            f"Selected proposal:\n{base_text}\n\n"
            f"Feedback:\n{state['feedback_msg']}\n\n"
            f"Retrieved projects:\n{projects_text or 'No additional project context retrieved.'}"
        ),
        fallback=fallback,
    )


def initialize_state(state: ProposalState) -> ProposalState:
    user = get_users_repository().get(state["user_id"])
    if user is None:
        raise ValueError(f"User '{state['user_id']}' was not found.")

    thread_record = None
    if state.get("thread_id"):
        existing = get_proposals_repository().get(state["thread_id"])
        if existing:
            thread_record = existing.model_dump(mode="json")

    template_id = state.get("template_id") or user.template_id
    template_text = user.selected_template_text
    if template_id in DEFAULT_TEMPLATES:
        template_text = DEFAULT_TEMPLATES[template_id].body
    if thread_record and thread_record.get("template_text"):
        template_text = thread_record["template_text"]
        template_id = thread_record.get("template_id", template_id)

    state["user_profile"] = user.model_dump(mode="json")
    state["template_id"] = template_id
    state["template_text"] = template_text
    state["thread_record"] = thread_record
    state["retrieval_attempt"] = 0
    state["retrieval_used"] = False
    state["fallback_used"] = False
    state["response_type"] = ResponseType.PROPOSALS.value
    state["summary"] = _build_summary(thread_record)
    state["pinned_context"] = _build_pinned_context(state)
    return state


def route_after_initialize(state: ProposalState) -> str:
    if state["mode"] == "optimize":
        return "classify_feedback"
    return "generate_query"


def classify_feedback(state: ProposalState) -> ProposalState:
    feedback_msg = state.get("feedback_msg") or ""
    state["direct_answer"] = None
    if _needs_direct_answer(feedback_msg):
        job = state["job_details"]
        lowered = feedback_msg.lower()
        if "budget" in lowered:
            answer = f"The stored budget for this job is {job.get('budget') or 'not specified'}."
        elif "title" in lowered:
            answer = f"The stored job title is '{job['title']}'."
        elif "client" in lowered:
            answer = f"The stored client information is {job.get('client_info') or 'not provided'}."
        elif "skills" in lowered:
            answer = "The stored required skills are " + ", ".join(job.get("required_skills", [])) + "."
        else:
            answer = f"Here are the stored job details:\n{json.dumps(job, indent=2)}"
        state["direct_answer"] = answer
        state["response_type"] = ResponseType.DIRECT_ANSWER.value
    return state


def route_after_classification(state: ProposalState) -> str:
    if state.get("direct_answer"):
        return "persist_direct_answer"
    if _requires_retrieval(state):
        return "generate_query"
    return "revise_proposal"


def generate_query(state: ProposalState) -> ProposalState:
    state["retrieval_attempt"] = int(state.get("retrieval_attempt", 0)) + 1
    state["vector_query"] = _llm_query(state) if llm_available() else _heuristic_query(state)
    return state


def retrieve_projects(state: ProposalState) -> ProposalState:
    projects = get_project_store().search_projects(
        user_id=state["user_id"],
        query=state["vector_query"],
        top_k=settings.retrieval_top_k,
    )
    state["retrieved_projects"] = [project.model_dump(mode="json") for project in projects]
    state["retrieval_used"] = True
    return state


def verify_retrieval(state: ProposalState) -> ProposalState:
    accepted, rationale = _verify_projects(state)
    matched_projects = state.get("retrieved_projects", [])
    tool_message = RetrieverToolMessage(
        query=state["vector_query"],
        matched_project_ids=[project["project_id"] for project in matched_projects],
        matched_project_titles=[project["title"] for project in matched_projects],
        accepted=accepted,
        rationale=rationale,
        attempt=state["retrieval_attempt"],
    )
    state["retrieval_accepted"] = accepted
    state["last_retriever_tool_message"] = tool_message.model_dump(mode="json")
    state["should_retry"] = not accepted and state["retrieval_attempt"] < settings.retrieval_max_retries
    if not accepted and not state["should_retry"]:
        state["fallback_used"] = True
    return state


def route_after_retrieval(state: ProposalState) -> str:
    if state.get("should_retry"):
        return "generate_query"
    if state["mode"] == "optimize":
        return "revise_proposal"
    return "generate_proposals"


def generate_proposals(state: ProposalState) -> ProposalState:
    proposals = _llm_generation(state) if llm_available() else _fallback_generation(state)
    state["proposals"] = [proposal.model_dump(mode="json") for proposal in proposals]
    state["response_type"] = ResponseType.PROPOSALS.value
    return state


def revise_proposal(state: ProposalState) -> ProposalState:
    thread = state.get("thread_record")
    if not thread:
        raise ValueError(f"Thread '{state['thread_id']}' was not found for optimization.")

    proposals = thread.get("proposals", [])
    selected_id = state["selected_proposal_id"]
    selected = next((proposal for proposal in proposals if proposal["id"] == selected_id), None)
    if selected is None:
        raise ValueError(f"Proposal '{selected_id}' was not found in thread '{state['thread_id']}'.")

    updated_text = _llm_revision(state, selected["text"]) if llm_available() else (
        f"{selected['text']}\n\nUpdate requested: {state['feedback_msg']}"
    )
    state["updated_proposal"] = updated_text
    state["response_type"] = ResponseType.PROPOSAL_UPDATE.value
    return state


def persist_direct_answer(state: ProposalState) -> ProposalState:
    thread = get_proposals_repository().get(state["thread_id"])
    if thread is None:
        raise ValueError(f"Thread '{state['thread_id']}' was not found for optimization.")

    updated_messages = [
        *thread.messages,
        ConversationMessage(role=MessageRole.USER, content=state["feedback_msg"] or ""),
        ConversationMessage(role=MessageRole.ASSISTANT, content=state["direct_answer"] or ""),
    ]
    trimmed_messages, summary = _apply_summary_window(updated_messages, existing_summary=thread.summary)
    updated_record = thread.model_copy(
        update={
            "messages": trimmed_messages,
            "summary": summary,
            "latest_response_type": ResponseType.DIRECT_ANSWER,
            "status": TaskStatus.COMPLETED,
        }
    )
    stored = get_proposals_repository().upsert(updated_record)
    state["summary"] = stored.summary
    return state


def persist_result(state: ProposalState) -> ProposalState:
    proposals_repo = get_proposals_repository()
    existing_thread = proposals_repo.get(state["thread_id"])

    if state["mode"] == "generate":
        thread_messages = [
            ConversationMessage(
                role=MessageRole.USER,
                content=(
                    f"Generate proposals for job '{state['job_details']['title']}' "
                    f"with skills {', '.join(state['job_details'].get('required_skills', []))}."
                ),
            ),
            ConversationMessage(
                role=MessageRole.ASSISTANT,
                content=json.dumps(state["proposals"], indent=2),
            ),
        ]
        trimmed_messages, summary = _apply_summary_window(thread_messages, existing_summary=state.get("summary"))
        record = ProposalThreadRecord(
            user_id=state["user_id"],
            thread_id=state["thread_id"],
            job_details=state["job_details"],
            template_id=state["template_id"],
            template_text=state["template_text"],
            proposals=[ProposalOption.model_validate(item) for item in state["proposals"]],
            selected_proposal_id=None,
            latest_response_type=ResponseType.PROPOSALS,
            messages=trimmed_messages,
            summary=summary,
            last_retriever_tool_message=(
                RetrieverToolMessage.model_validate(state["last_retriever_tool_message"])
                if state.get("last_retriever_tool_message") and state.get("retrieval_accepted")
                else None
            ),
            status=TaskStatus.COMPLETED,
        )
    else:
        if existing_thread is None:
            raise ValueError(f"Thread '{state['thread_id']}' was not found for optimization.")
        updated_messages = [
            *existing_thread.messages,
            ConversationMessage(role=MessageRole.USER, content=state["feedback_msg"] or ""),
            ConversationMessage(role=MessageRole.ASSISTANT, content=state["updated_proposal"] or ""),
        ]
        trimmed_messages, summary = _apply_summary_window(updated_messages, existing_summary=existing_thread.summary)
        updated_proposals = []
        for proposal in existing_thread.proposals:
            if proposal.id == state["selected_proposal_id"]:
                updated_proposals.append(
                    ProposalOption(id=proposal.id, label=proposal.label, text=state["updated_proposal"] or proposal.text)
                )
            else:
                updated_proposals.append(proposal)
        record = existing_thread.model_copy(
            update={
                "proposals": updated_proposals,
                "selected_proposal_id": state["selected_proposal_id"],
                "latest_response_type": ResponseType.PROPOSAL_UPDATE,
                "messages": trimmed_messages,
                "summary": summary,
                "last_retriever_tool_message": (
                    RetrieverToolMessage.model_validate(state["last_retriever_tool_message"])
                    if state.get("last_retriever_tool_message") and state.get("retrieval_accepted")
                    else existing_thread.last_retriever_tool_message
                ),
                "status": TaskStatus.COMPLETED,
            }
        )

    stored = proposals_repo.upsert(record)
    state["summary"] = stored.summary
    return state


builder = StateGraph(ProposalState)
builder.add_node("initialize_state", initialize_state)
builder.add_node("classify_feedback", classify_feedback)
builder.add_node("generate_query", generate_query)
builder.add_node("retrieve_projects", retrieve_projects)
builder.add_node("verify_retrieval", verify_retrieval)
builder.add_node("generate_proposals", generate_proposals)
builder.add_node("revise_proposal", revise_proposal)
builder.add_node("persist_direct_answer", persist_direct_answer)
builder.add_node("persist_result", persist_result)

builder.add_edge(START, "initialize_state")
builder.add_conditional_edges("initialize_state", route_after_initialize)
builder.add_conditional_edges("classify_feedback", route_after_classification)
builder.add_edge("generate_query", "retrieve_projects")
builder.add_edge("retrieve_projects", "verify_retrieval")
builder.add_conditional_edges("verify_retrieval", route_after_retrieval)
builder.add_edge("generate_proposals", "persist_result")
builder.add_edge("revise_proposal", "persist_result")
builder.add_edge("persist_direct_answer", END)
builder.add_edge("persist_result", END)

proposal_graph = builder.compile()


def run_generate_flow(task_id: str, payload: dict[str, Any]) -> GenerateProposalResponse:
    result = proposal_graph.invoke(
        {
            "mode": "generate",
            "task_id": task_id,
            "user_id": payload["user_id"],
            "thread_id": payload["thread_id"],
            "template_id": payload.get("template_id"),
            "job_details": payload["job_details"],
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
    )


def run_optimize_flow(task_id: str, payload: dict[str, Any]) -> OptimizeProposalResponse:
    thread = get_proposals_repository().get(payload["thread_id"])
    if thread is None:
        raise ValueError(f"Thread '{payload['thread_id']}' was not found.")

    result = proposal_graph.invoke(
        {
            "mode": "optimize",
            "task_id": task_id,
            "user_id": thread.user_id,
            "thread_id": thread.thread_id,
            "job_details": thread.job_details.model_dump(mode="json"),
            "selected_proposal_id": payload["selected_proposal_id"],
            "feedback_msg": payload["feedback_msg"],
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
    )
