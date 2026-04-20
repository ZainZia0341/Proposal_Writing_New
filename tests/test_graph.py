from __future__ import annotations

from app.graph import run_generate_flow, run_optimize_flow
from app.repositories import get_proposals_repository
from app.schemas import ConversationMessage, JobDetails, MessageRole, ProposalOption, ProposalThreadRecord, ResponseType, TaskStatus


def _seed_thread(thread_id: str = "thread_001") -> ProposalThreadRecord:
    record = ProposalThreadRecord(
        user_id="zain_zia_001",
        thread_id=thread_id,
        job_details=JobDetails(
            title="Senior AI Developer for Children's Educational App",
            description="Build personalized learning stories with image consistency.",
            budget="$3,000",
            skills_required=["Node.js", "Generative AI", "API Integration"],
            client_info="EdTech Startup in London",
        ),
        template_id="custom-template-1",
        template_text="Custom template text",
        proposals=[
            ProposalOption(id="alt_1", label="Balanced", text="Balanced proposal"),
            ProposalOption(id="alt_2", label="Consultative", text="Consultative proposal"),
            ProposalOption(id="alt_3", label="Fast Mover", text="Fast mover proposal"),
        ],
        selected_proposal_id="alt_2",
        latest_response_type=ResponseType.PROPOSALS,
        messages=[
            ConversationMessage(role=MessageRole.USER, content="Generate proposal"),
            ConversationMessage(role=MessageRole.ASSISTANT, content="Generated proposals"),
        ],
        status=TaskStatus.COMPLETED,
    )
    return get_proposals_repository().upsert(record)


def test_graph_routes_direct_answer(monkeypatch):
    import app.graph as graph

    _seed_thread()
    monkeypatch.setattr(
        graph,
        "_route_feedback_with_llm",
        lambda state: {"route": "direct_answer", "answer_kind": "budget", "reason": "Stored answer"},
    )
    response = run_optimize_flow(
        "task_direct",
        {
            "thread_id": "thread_001",
            "selected_proposal_id": "alt_2",
            "feedback_msg": "What is the budget?",
        },
    )
    assert response.response_type == ResponseType.DIRECT_ANSWER
    assert response.updated_proposal is None
    assert "$3,000" in (response.direct_answer or "")


def test_graph_routes_revise_only(monkeypatch):
    import app.graph as graph

    _seed_thread("thread_revise")
    monkeypatch.setattr(
        graph,
        "_route_feedback_with_llm",
        lambda state: {"route": "revise_only", "answer_kind": "generic", "reason": "Existing context is enough"},
    )
    monkeypatch.setattr(graph, "_revise_proposal_with_llm", lambda state, base_text: "Revised without retriever")
    response = run_optimize_flow(
        "task_revise",
        {
            "thread_id": "thread_revise",
            "selected_proposal_id": "alt_2",
            "feedback_msg": "Make it shorter.",
        },
    )
    assert response.response_type == ResponseType.PROPOSAL_UPDATE
    assert response.updated_proposal == "Revised without retriever"
    assert response.retrieval_used is False


def test_graph_routes_retrieve_then_revise(monkeypatch):
    import app.graph as graph

    _seed_thread("thread_retrieve")
    monkeypatch.setattr(
        graph,
        "_route_feedback_with_llm",
        lambda state: {"route": "retrieve_then_revise", "answer_kind": "generic", "reason": "Need project history"},
    )
    monkeypatch.setattr(graph, "_plan_vector_query_with_llm", lambda state: "aws experience query")
    monkeypatch.setattr(graph, "_verify_retrieval_with_llm", lambda state: {"accepted": True, "rationale": "Relevant projects found"})
    monkeypatch.setattr(graph, "_revise_proposal_with_llm", lambda state, base_text: "Revised with retriever")
    response = run_optimize_flow(
        "task_retrieve",
        {
            "thread_id": "thread_retrieve",
            "selected_proposal_id": "alt_2",
            "feedback_msg": "Justify the budget using my AWS experience.",
        },
    )
    assert response.response_type == ResponseType.PROPOSAL_UPDATE
    assert response.updated_proposal == "Revised with retriever"
    assert response.retrieval_used is True


def test_graph_retries_then_accepts_retrieval(monkeypatch):
    import app.graph as graph

    monkeypatch.setattr(graph, "_plan_vector_query_with_llm", lambda state: f"query-{state.get('retrieval_attempt', 0) + 1}")
    monkeypatch.setattr(
        graph,
        "_verify_retrieval_with_llm",
        lambda state: {
            "accepted": state.get("vector_query") == "query-2",
            "rationale": "Accepted on second attempt" if state.get("vector_query") == "query-2" else "Retry",
        },
    )
    monkeypatch.setattr(
        graph,
        "_generate_proposals_with_llm",
        lambda state: [
            graph.ProposalOption(id="alt_1", label="Balanced", text="P1"),
            graph.ProposalOption(id="alt_2", label="Consultative", text="P2"),
            graph.ProposalOption(id="alt_3", label="Fast Mover", text="P3"),
        ],
    )
    response = run_generate_flow(
        "task_retry",
        {
            "user_id": "zain_zia_001",
            "thread_id": "thread_retry",
            "template_id": "custom-template-1",
            "job_details": JobDetails(
                title="AI Developer",
                description="Need a strong RAG engineer",
                budget="$2,000",
                skills_required=["Python", "Pinecone"],
                client_info="Startup",
            ).model_dump(mode="json"),
        },
    )
    assert response.fallback_used is False
    stored = get_proposals_repository().get("thread_retry")
    assert stored is not None
    assert stored.last_retriever_tool_message is not None
    assert stored.last_retriever_tool_message.attempt == 2


def test_graph_uses_fallback_after_three_failed_retrieval_attempts(monkeypatch):
    import app.graph as graph

    monkeypatch.setattr(graph, "_plan_vector_query_with_llm", lambda state: "always-fail-query")
    monkeypatch.setattr(graph, "_verify_retrieval_with_llm", lambda state: {"accepted": False, "rationale": "No relevant docs"})
    monkeypatch.setattr(
        graph,
        "_generate_fallback_proposals_with_llm",
        lambda state: [
            graph.ProposalOption(id="alt_1", label="Balanced", text="Fallback LLM proposal 1"),
            graph.ProposalOption(id="alt_2", label="Consultative", text="Fallback LLM proposal 2"),
            graph.ProposalOption(id="alt_3", label="Fast Mover", text="Fallback LLM proposal 3"),
        ],
    )
    response = run_generate_flow(
        "task_fallback",
        {
            "user_id": "zain_zia_001",
            "thread_id": "thread_fallback",
            "template_id": "custom-template-1",
            "job_details": JobDetails(
                title="Backend AI Engineer",
                description="Need a proposal without relevant history",
                budget="$1,500",
                skills_required=["Python", "FastAPI"],
                client_info="Solo founder",
            ).model_dump(mode="json"),
        },
    )
    assert response.fallback_used is True
    assert len(response.proposals) == 3
    assert [proposal.text for proposal in response.proposals] == [
        "Fallback LLM proposal 1",
        "Fallback LLM proposal 2",
        "Fallback LLM proposal 3",
    ]


def test_graph_summary_triggers_after_threshold(monkeypatch):
    import app.graph as graph

    messages = []
    for index in range(12):
        role = MessageRole.USER if index % 2 == 0 else MessageRole.ASSISTANT
        messages.append(ConversationMessage(role=role, content=f"message-{index}"))

    record = ProposalThreadRecord(
        user_id="zain_zia_001",
        thread_id="thread_summary",
        job_details=JobDetails(
            title="Summary Job",
            description="Summary description",
            budget="$999",
            skills_required=["Node.js"],
            client_info="Client",
        ),
        template_id="custom-template-1",
        template_text="Template",
        proposals=[
            ProposalOption(id="alt_1", label="Balanced", text="Balanced proposal"),
            ProposalOption(id="alt_2", label="Consultative", text="Consultative proposal"),
            ProposalOption(id="alt_3", label="Fast Mover", text="Fast mover proposal"),
        ],
        selected_proposal_id="alt_1",
        latest_response_type=ResponseType.PROPOSALS,
        messages=messages,
        status=TaskStatus.COMPLETED,
    )
    get_proposals_repository().upsert(record)

    monkeypatch.setattr(graph, "_summarize_messages", lambda messages, existing_summary=None: "rolled up summary")
    monkeypatch.setattr(
        graph,
        "_route_feedback_with_llm",
        lambda state: {"route": "direct_answer", "answer_kind": "budget", "reason": "Stored answer"},
    )
    response = run_optimize_flow(
        "task_summary",
        {
            "thread_id": "thread_summary",
            "selected_proposal_id": "alt_1",
            "feedback_msg": "What was the budget?",
        },
    )
    assert response.response_type == ResponseType.DIRECT_ANSWER
    stored = get_proposals_repository().get("thread_summary")
    assert stored is not None
    assert stored.summary == "rolled up summary"
    assert len(stored.messages) <= 10
