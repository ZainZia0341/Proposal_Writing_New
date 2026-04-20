from __future__ import annotations

import json
from typing import Any, Literal
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command

from app.config import settings
from app.llm import invoke_json, invoke_text
from app.logging_utils import get_logger
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

logger = get_logger(__name__)


class ProposalState(MessagesState):
    mode: str
    task_id: str
    user_id: str
    thread_id: str
    template_id: str | None
    template_text: str
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
    user = state["user_profile"]
    job = state["job_details"]
    selected_proposal = _build_selected_proposal(state)
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
    current_summary = summary if summary is not None else state.get("summary")
    if current_summary:
        lines.append(f"Conversation Summary: {current_summary}")
    if selected_proposal:
        lines.append(f"Selected Proposal ID: {selected_proposal['id']}")
        lines.append(f"Base Proposal Text: {selected_proposal['text']}")
    return "\n".join(lines)


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
    skills = ", ".join(job.get("required_skills", [])) or ", ".join(user.get("experience_languages", []))
    project_lines = []
    for project in state.get("retrieved_projects", [])[:2]:
        project_lines.append(
            f"{project['title']} - {project['description']} ({', '.join(project.get('tech_stack', []))})"
        )
    portfolio_section = "\n".join(project_lines) if project_lines else "Relevant portfolio examples available on request."
    variants = [
        ProposalOption(
            id="alt_1",
            label="Balanced",
            text=(
                f"Hi, this sounds like a strong fit.\n\n"
                f"I'm {user['full_name']}, a {user['designation']} with hands-on experience in {skills}. "
                f"Your job around {job['title']} is closely aligned with the kind of systems I build.\n\n"
                f"Relevant experience:\n{portfolio_section}\n\n"
                f"I can help you deliver this with a practical and reliable implementation. Would you like me to outline the execution plan?\n\n"
                f"Best,\n{user['full_name']}"
            ),
        ),
        ProposalOption(
            id="alt_2",
            label="Consultative",
            text=(
                f"Hi, this aligns closely with the type of work I've been doing recently.\n\n"
                f"The challenge in {job['title']} is not only shipping features, but making sure the solution is maintainable and aligned with {skills}. "
                f"I've handled similar delivery problems across AI and backend projects.\n\n"
                f"Relevant experience:\n{portfolio_section}\n\n"
                f"If helpful, I can break this into milestones and recommend a clean technical approach.\n\n"
                f"Best,\n{user['full_name']}"
            ),
        ),
        ProposalOption(
            id="alt_3",
            label="Fast Mover",
            text=(
                f"I can definitely help here.\n\n"
                f"I've built similar systems using {skills} and can move quickly on a project like {job['title']}. "
                f"My background as a {user['designation']} helps me bridge product goals with clean implementation.\n\n"
                f"Relevant experience:\n{portfolio_section}\n\n"
                f"If you want, I can start with the highest-impact deliverable first and keep the scope tight.\n\n"
                f"Best,\n{user['full_name']}"
            ),
        ),
    ]
    return variants


def _generate_proposals_with_llm(state: ProposalState) -> list[ProposalOption]:
    fallback = {"alternatives": [proposal.model_dump(mode="json") for proposal in _fallback_generation(state)]}
    projects_text = "\n\n".join(_format_project(project) for project in state.get("retrieved_projects", []))
    payload = invoke_json(
        system_prompt=(
            "You write three distinct freelance proposal alternatives. "
            "Return JSON with key alternatives, an array of objects containing id, label, and text. "
            "The ids must be alt_1, alt_2, and alt_3. "
            "Stay strictly grounded in the provided user profile, accepted retrieved projects, and job details. "
            "Do not mention any skill, tool, framework, certification, domain experience, or project detail unless it is explicitly supported by the provided context. "
            "Do not let job-description keywords push you into claiming experience the user does not actually have. "
            "The saved template is style guidance, not a rigid script. You may improve, adapt, or creatively expand the writing style, but you must remain within the factual boundaries of the provided context. "
            "Never invent evidence, never overclaim, and never add unsupported project names."
        ),
        user_prompt=(
            f"Pinned context:\n{state['pinned_context']}\n\n"
            f"Recent messages:\n{_messages_to_text(state['messages'])}\n\n"
            f"Accepted retrieved projects:\n{projects_text or 'No accepted retrieved projects. Use only confirmed user profile and job context. Do not mention any project details.'}"
        ),
        fallback=fallback,
    )
    return [ProposalOption.model_validate(item) for item in payload.get("alternatives", fallback["alternatives"])[:3]]


def _generate_fallback_proposals_with_llm(state: ProposalState) -> list[ProposalOption]:
    fallback = {"alternatives": [proposal.model_dump(mode="json") for proposal in _fallback_generation(state)]}
    payload = invoke_json(
        system_prompt=(
            "You write three distinct freelance proposal alternatives when portfolio retrieval did not return "
            "trusted supporting projects. Return JSON with key alternatives, an array of objects containing "
            "id, label, and text. The ids must be alt_1, alt_2, and alt_3. Use only the provided user profile, "
            "template style, job details, and recent conversation. "
            "Do not mention any project name, project achievement, or project detail because retrieval did not produce accepted evidence. "
            "Do not mention any skill, tool, framework, certification, or domain experience unless it is explicitly supported by the provided user profile. "
            "Do not let job-description keywords influence you into adding unsupported skills. "
            "The saved template is style guidance, not a rigid script. You may write more naturally and creatively than the template, but you must stay strictly inside the factual boundaries of the provided context."
        ),
        user_prompt=(
            f"Pinned context:\n{state['pinned_context']}\n\n"
            f"Recent messages:\n{_messages_to_text(state['messages'])}\n\n"
            "Retriever status:\n"
            "No accepted retrieved projects were available after retry attempts. "
            "Generate strong proposals from the user's real skills, designation, template, and the current job details only. "
            "If a detail is not explicitly supported, leave it out."
        ),
        fallback=fallback,
    )
    return [ProposalOption.model_validate(item) for item in payload.get("alternatives", fallback["alternatives"])[:3]]


def _revise_proposal_with_llm(state: ProposalState, base_text: str) -> str:
    fallback = (
        f"{base_text}\n\n"
        f"Update requested: {state.get('feedback_msg')}\n"
        "This draft was updated using the existing proposal context."
    )
    projects_text = "\n\n".join(_format_project(project) for project in state.get("retrieved_projects", []))
    return invoke_text(
        system_prompt=(
            "You revise an existing freelance proposal. Preserve the strongest parts of the original, "
            "apply the feedback exactly, and avoid inventing facts. "
            "Do not add any skill, tool, framework, certification, industry background, or project detail unless it is explicitly supported by the provided context. "
            "Do not let the job description or feedback wording push you into claiming experience the user has not actually shown. "
            "If there are no accepted retrieved projects, do not add project-specific claims. "
            "The template is guidance for tone and structure, not a rigid format. You may improve the writing while staying strictly within factual boundaries."
        ),
        user_prompt=(
            f"Pinned context:\n{state['pinned_context']}\n\n"
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


def initialize_context(
    state: ProposalState,
) -> Command[Literal["summarize_history", "route_feedback", "plan_query"]]:
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

    updates: dict[str, Any] = {
        "user_profile": user.model_dump(mode="json"),
        "thread_record": thread_record,
        "template_id": template_id,
        "template_text": template_text,
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
        extra={"thread_id": state["thread_id"], "mode": state["mode"], "next_node": next_node},
    )
    return Command(update=updates, goto=next_node)


def summarize_history(
    state: ProposalState,
) -> Command[Literal["route_feedback", "plan_query"]]:
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
    decision = _route_feedback_with_llm(state)
    route = str(decision.get("route", "retrieve_then_revise"))
    answer_kind = str(decision.get("answer_kind", "generic"))
    reason = str(decision.get("reason", ""))
    updates = {
        "route_decision": route,
        "route_reason": reason,
        "direct_answer_kind": answer_kind,
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
    answer = _build_direct_answer_text(state)
    logger.info("Generated direct answer from stored context", extra={"thread_id": state["thread_id"]})
    return {
        "direct_answer": answer,
        "messages": [_ai_message(answer)],
        "response_type": ResponseType.DIRECT_ANSWER.value,
    }


def plan_query(state: ProposalState) -> Command[Literal["retrieve_projects"]]:
    query = _plan_vector_query_with_llm(state)
    attempt = int(state.get("retrieval_attempt", 0)) + 1
    logger.info(
        "Planned vector query",
        extra={"thread_id": state["thread_id"], "attempt": attempt, "query": query},
    )
    return Command(
        update={
            "vector_query": query,
            "retrieval_attempt": attempt,
        },
        goto="retrieve_projects",
    )


def retrieve_projects(state: ProposalState) -> dict[str, Any]:
    projects = get_project_store().search_projects(
        user_id=state["user_id"],
        query=state["vector_query"] or "",
        top_k=settings.retrieval_top_k,
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
    return Command(update=updates, goto=next_node)


def generate_proposals(state: ProposalState) -> dict[str, Any]:
    proposals = _generate_proposals_with_llm(state)
    logger.info("Generated proposal alternatives", extra={"thread_id": state["thread_id"], "count": len(proposals)})
    return {
        "proposals": [proposal.model_dump(mode="json") for proposal in proposals],
        "messages": [_ai_message(_proposal_message_text(proposals))],
        "response_type": ResponseType.PROPOSALS.value,
    }


def generate_fallback_proposals(state: ProposalState) -> dict[str, Any]:
    proposals = _generate_fallback_proposals_with_llm(state)
    logger.info(
        "Generated fallback proposal alternatives with LLM context",
        extra={"thread_id": state["thread_id"], "count": len(proposals)},
    )
    return {
        "proposals": [proposal.model_dump(mode="json") for proposal in proposals],
        "messages": [_ai_message(_proposal_message_text(proposals))],
        "response_type": ResponseType.PROPOSALS.value,
        "fallback_used": True,
    }


def revise_selected_proposal(state: ProposalState) -> dict[str, Any]:
    selected_proposal = _build_selected_proposal(state)
    if selected_proposal is None:
        raise ValueError(f"Proposal '{state.get('selected_proposal_id')}' was not found in thread '{state['thread_id']}'.")
    updated_text = _revise_proposal_with_llm(state, selected_proposal["text"])
    logger.info("Revised selected proposal", extra={"thread_id": state["thread_id"], "proposal_id": selected_proposal["id"]})
    return {
        "updated_proposal": updated_text,
        "messages": [_ai_message(updated_text)],
        "response_type": ResponseType.PROPOSAL_UPDATE.value,
    }


def finalize_memory_window(state: ProposalState) -> Command[Literal["persist_result"]]:
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
    }
    updates["pinned_context"] = _build_pinned_context({**state, **updates}, summary=summary)
    logger.info(
        "Applied final memory window before persistence",
        extra={"thread_id": state["thread_id"], "removed_messages": len(removals)},
    )
    return Command(update=updates, goto="persist_result")


def persist_result(state: ProposalState) -> dict[str, Any]:
    proposals_repo = get_proposals_repository()
    existing_thread = proposals_repo.get(state["thread_id"])
    stored_messages = _messages_to_schema(state["messages"])

    if state["response_type"] == ResponseType.PROPOSALS.value:
        record = ProposalThreadRecord(
            user_id=state["user_id"],
            thread_id=state["thread_id"],
            job_details=state["job_details"],
            template_id=state["template_id"] or "",
            template_text=state["template_text"],
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
    existing_thread = get_proposals_repository().get(payload["thread_id"])
    if existing_thread and existing_thread.user_id != payload["user_id"]:
        raise ValueError(f"Thread '{payload['thread_id']}' does not belong to user '{payload['user_id']}'.")

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
            "template_id": payload.get("template_id"),
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
    )


def run_optimize_flow(task_id: str, payload: dict[str, Any]) -> OptimizeProposalResponse:
    thread = get_proposals_repository().get(payload["thread_id"])
    if thread is None:
        raise ValueError(f"Thread '{payload['thread_id']}' was not found.")

    messages = [_schema_to_langchain(message, index) for index, message in enumerate(thread.messages)]
    messages.append(_human_message(payload["feedback_msg"]))

    result = proposal_graph.invoke(
        {
            "mode": "optimize",
            "task_id": task_id,
            "user_id": thread.user_id,
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
    )
