from __future__ import annotations

from app.bid_examples import build_bid_style_record, format_bid_example_markdown
from app.graph import run_generate_flow, run_optimize_flow
from app.repositories import get_proposals_repository
from app.schemas import (
    BidExampleInput,
    ConversationMessage,
    FullStackUserProfile,
    JobDetails,
    MessageRole,
    ProposalOption,
    ProposalThreadRecord,
    ResponseType,
    TaskStatus,
    TemplateSnapshot,
    TemplateType,
)


USER_SNAPSHOT = FullStackUserProfile(
    full_name="Zain Zia",
    designation="Generative AI Developer",
    expertise_areas=["LLMs", "RAG systems", "Vector Databases"],
    experience_languages=["Node.js", "Python", "React.js"],
    experience_years=5,
    tone_preference="upwork",
)

TEMPLATE_SNAPSHOT = TemplateSnapshot(
    template_id="custom-template-1",
    template_type=TemplateType.CUSTOM,
    template_text="Hi [Client Name], I saw your post for [Job Title]...",
)

BID_EXAMPLE = BidExampleInput(
    job_details=JobDetails(
        title="AI chatbot for education",
        description="Need RAG and LLM workflow support.",
        budget="$2,500",
        skills_required=["Python", "LangChain", "Pinecone"],
        client_info="EdTech startup",
    ),
    proposal_text="Hi, this aligns closely with my recent AI work and I can help quickly.",
)


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
        user_profile_snapshot=USER_SNAPSHOT,
        template_snapshot=TEMPLATE_SNAPSHOT,
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


def _seed_bid_style(user_id: str = "zain_zia_001"):
    return get_proposals_repository().upsert_bid_style(build_bid_style_record(user_id, [BID_EXAMPLE]))


def test_bid_example_markdown_formatting_is_stable():
    markdown = format_bid_example_markdown(BID_EXAMPLE)
    assert "## Job Title" in markdown
    assert "AI chatbot for education" in markdown
    assert "### Skills" in markdown
    assert "Python, LangChain, Pinecone" in markdown
    assert "## Sent Proposal" in markdown
    assert BID_EXAMPLE.proposal_text in markdown


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

    _seed_bid_style()
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
            "user_profile": USER_SNAPSHOT.model_dump(mode="json"),
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


def test_graph_propagates_model_used_into_generate_response(monkeypatch):
    import app.graph as graph

    _seed_bid_style()
    monkeypatch.setattr(graph, "get_model_used", lambda: "groq:openai/gpt-oss-120b")
    monkeypatch.setattr(graph, "_plan_vector_query_with_llm", lambda state: "education ai story platform")
    monkeypatch.setattr(graph, "_verify_retrieval_with_llm", lambda state: {"accepted": True, "rationale": "Matched job context"})
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
        "task_model_used",
        {
            "user_id": "zain_zia_001",
            "thread_id": "thread_model_used",
            "user_profile": USER_SNAPSHOT.model_dump(mode="json"),
            "job_details": JobDetails(
                title="AI Developer",
                description="Need a strong RAG engineer",
                budget="$2,000",
                skills_required=["Python", "Pinecone"],
                client_info="Startup",
            ).model_dump(mode="json"),
        },
    )
    assert response.model_used == "groq:openai/gpt-oss-120b"


def test_generate_flow_passes_hook_to_generation_prompt_and_persists(monkeypatch):
    import app.graph as graph

    hook = "Lead with a short audit of their onboarding workflow."
    captured: dict[str, str | None] = {}

    _seed_bid_style()
    monkeypatch.setattr(graph, "_plan_vector_query_with_llm", lambda state: "onboarding workflow audit")
    monkeypatch.setattr(graph, "_verify_retrieval_with_llm", lambda state: {"accepted": True, "rationale": "Matched hook"})

    def fake_generate_proposals(state):
        captured["hook"] = state.get("hook")
        captured["pinned_context"] = state.get("pinned_context")
        return [
            graph.ProposalOption(id="alt_1", label="Balanced", text="P1"),
            graph.ProposalOption(id="alt_2", label="Consultative", text="P2"),
            graph.ProposalOption(id="alt_3", label="Fast Mover", text="P3"),
        ]

    monkeypatch.setattr(graph, "_generate_proposals_with_llm", fake_generate_proposals)

    response = run_generate_flow(
        "task_hook",
        {
            "user_id": "zain_zia_001",
            "thread_id": "thread_hook",
            "user_profile": USER_SNAPSHOT.model_dump(mode="json"),
            "job_details": JobDetails(
                title="AI Developer",
                description="Need a strong RAG engineer",
                budget="$2,000",
                skills_required=["Python", "Pinecone"],
                client_info="Startup",
            ).model_dump(mode="json"),
            "hook": hook,
        },
    )

    assert len(response.proposals) == 3
    assert captured["hook"] == hook
    assert f"User-Provided Proposal Hook: {hook}" in (captured["pinned_context"] or "")
    stored = get_proposals_repository().get("thread_hook")
    assert stored is not None
    assert stored.hook == hook


def test_graph_uses_fallback_after_three_failed_retrieval_attempts(monkeypatch):
    import app.graph as graph

    _seed_bid_style()
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
            "user_profile": USER_SNAPSHOT.model_dump(mode="json"),
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


def test_fallback_generation_omits_empty_relevant_experience_section():
    import app.graph as graph

    proposals = graph._fallback_generation(
        {
            "user_profile": USER_SNAPSHOT.model_dump(mode="json"),
            "job_details": JobDetails(
                title="Senior Node.js Developer for Fintech",
                description="Need secure API gateways and financial transactions.",
                budget="$5,000",
                skills_required=["Node.js", "AWS", "PostgreSQL"],
                client_info="Fintech client",
            ).model_dump(mode="json"),
            "retrieved_projects": [],
        }
    )

    assert len(proposals) == 3
    joined = "\n\n".join(proposal.text for proposal in proposals)
    assert "Relevant experience" not in joined
    assert "Relevant portfolio examples available on request" not in joined
    assert "portfolio examples" not in joined.lower()
    assert "Senior Node.js Developer for Fintech" in joined
    assert "Zain Zia" in joined


def test_fallback_prompt_bans_empty_portfolio_placeholders(monkeypatch):
    import app.graph as graph

    captured: dict[str, str] = {}
    hook = "Focus on reducing risk before writing new features."

    def fake_invoke_json(*, system_prompt, user_prompt, fallback):
        captured["system_prompt"] = system_prompt
        captured["user_prompt"] = user_prompt
        return fallback

    monkeypatch.setattr(graph, "invoke_json", fake_invoke_json)

    proposals = graph._generate_fallback_proposals_with_llm(
        {
            "user_profile": USER_SNAPSHOT.model_dump(mode="json"),
            "job_details": JobDetails(
                title="Backend AI Engineer",
                description="Need proposal help without accepted retrieved projects",
                budget="$1,500",
                skills_required=["Python", "FastAPI"],
                client_info="Solo founder",
            ).model_dump(mode="json"),
            "pinned_context": "Pinned job and user context",
            "hook": hook,
            "messages": [],
            "retrieved_projects": [],
            "bid_examples_markdown": [],
        }
    )

    assert len(proposals) == 3
    assert "Relevant portfolio examples available on request" in captured["system_prompt"]
    assert "omit portfolio and relevant-experience sections entirely" in captured["system_prompt"]
    assert "primary proposal angle" in captured["system_prompt"]
    assert hook in captured["user_prompt"]
    joined = "\n\n".join(proposal.text for proposal in proposals)
    assert "Relevant experience" not in joined
    assert "Relevant portfolio examples available on request" not in joined


def test_generate_prompt_includes_all_bid_examples_and_excludes_template_text(monkeypatch):
    import app.graph as graph

    captured: dict[str, str] = {}
    hook = "Open with a quick diagnosis of the AI workflow before proposing the build."

    def fake_invoke_json(*, system_prompt, user_prompt, fallback):
        captured["system_prompt"] = system_prompt
        captured["user_prompt"] = user_prompt
        return fallback

    monkeypatch.setattr(graph, "invoke_json", fake_invoke_json)

    graph._generate_proposals_with_llm(
        {
            "user_profile": USER_SNAPSHOT.model_dump(mode="json"),
            "job_details": JobDetails(
                title="AI Developer",
                description="Need AI proposal help",
                budget="$2,000",
                skills_required=["Python"],
                client_info="Startup",
            ).model_dump(mode="json"),
            "pinned_context": "Pinned job and user context",
            "hook": hook,
            "messages": [],
            "retrieved_projects": [],
            "bid_examples_markdown": [
                "## Job Title\nAI job one\n\n## Sent Proposal\nProposal one",
                "## Job Title\nBackend job two\n\n## Sent Proposal\nProposal two",
            ],
        }
    )

    assert "Previous Bid Example 1" in captured["user_prompt"]
    assert "Previous Bid Example 2" in captured["user_prompt"]
    assert "Proposal one" in captured["user_prompt"]
    assert "Proposal two" in captured["user_prompt"]
    assert "Frontend-provided hook" in captured["user_prompt"]
    assert hook in captured["user_prompt"]
    assert "primary proposal angle" in captured["system_prompt"]
    assert "template" not in captured["system_prompt"].lower()


def test_fallback_prompt_warns_against_copying_old_bid_facts(monkeypatch):
    import app.graph as graph

    captured: dict[str, str] = {}

    def fake_invoke_json(*, system_prompt, user_prompt, fallback):
        captured["system_prompt"] = system_prompt
        captured["user_prompt"] = user_prompt
        return fallback

    monkeypatch.setattr(graph, "invoke_json", fake_invoke_json)

    graph._generate_fallback_proposals_with_llm(
        {
            "user_profile": USER_SNAPSHOT.model_dump(mode="json"),
            "job_details": JobDetails(
                title="Full Stack Developer",
                description="Need proposal help",
                budget="$1,800",
                skills_required=["Node.js"],
                client_info="Agency",
            ).model_dump(mode="json"),
            "pinned_context": "Pinned job and user context",
            "messages": [],
            "retrieved_projects": [],
            "bid_examples_markdown": [
                "## Job Title\nOld client job\n\n## Sent Proposal\nOld proposal"
            ],
        }
    )

    assert "Previous Bid Example 1" in captured["user_prompt"]
    assert "Never copy old client names" in captured["system_prompt"]


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
        user_profile_snapshot=USER_SNAPSHOT,
        template_snapshot=TEMPLATE_SNAPSHOT,
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
