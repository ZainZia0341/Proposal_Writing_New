from __future__ import annotations

import json
from pathlib import Path

from app.repositories import get_proposals_repository
from app.schemas import BidExampleDraftRequest, BidSyncRequest, GenerateProposalRequest, OptimizeProposalRequest, PortfolioSyncRequest


def _mock_three_proposals():
    return [
        {"id": "alt_1", "label": "Balanced", "text": "Balanced proposal"},
        {"id": "alt_2", "label": "Consultative", "text": "Consultative proposal"},
        {"id": "alt_3", "label": "Fast Mover", "text": "Fast mover proposal"},
    ]


def _patch_successful_generation(monkeypatch):
    import app.graph as graph

    monkeypatch.setattr(graph, "_plan_vector_query_with_llm", lambda state: "education ai story platform")
    monkeypatch.setattr(graph, "_verify_retrieval_with_llm", lambda state: {"accepted": True, "rationale": "Matched job context"})
    monkeypatch.setattr(
        graph,
        "_generate_proposals_with_llm",
        lambda state: [graph.ProposalOption.model_validate(item) for item in _mock_three_proposals()],
    )


def test_templates_endpoint_returns_full_template_records(client):
    response = client.get("/api/v1/templates")
    assert response.status_code == 200
    body = response.json()
    assert len(body) >= 3
    consultative = next(template for template in body if template["template_id"] == "consultative_expert")
    assert consultative["label"] == "Consultative Expert"
    assert "body" in consultative
    assert "recent project" in consultative["body"]


def test_health_endpoint_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_signup_route_removed_from_ai_backend(client):
    response = client.post("/api/v1/signup", json={})
    assert response.status_code == 404


def test_generate_returns_three_proposals(client, load_payload, monkeypatch):
    _patch_successful_generation(monkeypatch)
    payload = load_payload("generate_proposal.json")
    response = client.post("/api/v1/proposals/generate", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["thread_id"]
    assert body["task_id"]
    assert len(body["proposals"]) == 3
    assert [proposal["id"] for proposal in body["proposals"]] == ["alt_1", "alt_2", "alt_3"]
    assert "model_used" in body
    assert body["model_used"] is None
    stored = get_proposals_repository().get(body["thread_id"])
    assert stored is not None
    assert stored.user_profile_snapshot is not None
    assert stored.template_snapshot is None
    assert stored.user_profile_snapshot.full_name == payload["user_profile"]["full_name"]


def test_portfolio_sync_upserts_projects_for_ai_dev(client, load_payload):
    payload = load_payload("portfolio_sync.json")
    response = client.post("/api/v1/portfolio/sync", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == payload["user_id"]
    assert body["stored_projects"] == len(payload["projects"])
    assert body["namespace"] == payload["user_id"]
    assert body["received_scraped_profile_text"] is True
    assert body["model_used"] is None


def test_bid_sync_stores_markdown_examples(client, load_payload):
    payload = load_payload("bids_sync.json")
    response = client.post("/api/v1/proposals/bids/sync", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == payload["user_id"]
    assert body["stored_bids"] == len(payload["bids"])
    assert body["max_examples"] == 5
    stored = get_proposals_repository().get_bid_style(payload["user_id"])
    assert stored is not None
    assert len(stored.bids) == len(payload["bids"])
    assert "## Job Title" in stored.bids[0].markdown
    assert "## Sent Proposal" in stored.bids[0].markdown


def test_bid_sync_empty_payload_clears_stored_examples(client, load_payload):
    payload = load_payload("bids_sync.json")
    first_response = client.post("/api/v1/proposals/bids/sync", json=payload)
    assert first_response.status_code == 200

    clear_response = client.post("/api/v1/proposals/bids/sync", json={"user_id": payload["user_id"], "bids": []})
    assert clear_response.status_code == 200
    assert clear_response.json()["stored_bids"] == 0
    assert get_proposals_repository().get_bid_style(payload["user_id"]) is None


def test_bid_sync_rejects_more_than_five_examples(client, load_payload):
    payload = load_payload("bids_sync.json")
    example = payload["bids"][0]
    response = client.post(
        "/api/v1/proposals/bids/sync",
        json={"user_id": payload["user_id"], "bids": [example, example, example, example, example, example]},
    )
    assert response.status_code == 422


def test_portfolio_sync_returns_cleaner_error_when_vector_upsert_fails(client, load_payload, monkeypatch):
    import app.services as services

    class BrokenStore:
        def upsert_projects(self, user_id, projects):
            raise RuntimeError("simulated pinecone failure")

    monkeypatch.setattr(services, "get_project_store", lambda: BrokenStore())
    response = client.post("/api/v1/portfolio/sync", json=load_payload("portfolio_sync.json"))
    assert response.status_code == 500
    assert "Portfolio sync failed during Pinecone upsert or embedding generation" in response.json()["detail"]


def test_pinecone_upsert_uses_lambda_safe_sync_mode():
    from app.schemas import ProjectRecord
    from app.vector_store import PineconeProjectVectorStore

    captured: dict[str, object] = {}

    class FakeVectorStore:
        def add_documents(self, **kwargs):
            captured.update(kwargs)

    store = PineconeProjectVectorStore.__new__(PineconeProjectVectorStore)
    store._vector_store = FakeVectorStore()
    store._known_namespaces = set()
    store._namespace_exists = lambda user_id: False

    stored_count = store.upsert_projects(
        "lambda_user_001",
        [
            ProjectRecord(
                project_id="p1",
                title="StoryBloom",
                description="Built an AI storybook generator.",
                tech_stack=["Node.js", "OpenAI"],
                role="Lead Generative AI Developer",
            )
        ],
    )

    assert stored_count == 1
    assert captured["namespace"] == "lambda_user_001"
    assert captured["async_req"] is False


def test_generation_endpoint_creates_task_with_non_null_thread_id(client, load_payload, monkeypatch):
    import app.main as main_module
    from app.schemas import TaskStatus

    captured: dict[str, object] = {}

    def fake_build_generation_task(thread_id: str):
        captured["thread_id"] = thread_id

        class _Task:
            task_id = "task_test_123"

        return _Task()

    monkeypatch.setattr(main_module, "build_generation_task", fake_build_generation_task)
    monkeypatch.setattr(main_module, "mark_task_started", lambda task_id, thread_id=None: None)
    monkeypatch.setattr(
        main_module,
        "run_generate_flow",
        lambda task_id, payload: main_module.GenerateProposalResponse(
            thread_id=payload["thread_id"],
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            proposals=[],
            retrieval_used=False,
            fallback_used=False,
            retrieved_project_ids=[],
            summary=None,
            model_used=None,
        ),
    )
    monkeypatch.setattr(main_module, "finalize_generation_result", lambda task_id, response: response)

    response = client.post("/api/v1/proposals/generate", json=load_payload("generate_proposal.json"))
    assert response.status_code == 200
    assert isinstance(captured["thread_id"], str)
    assert captured["thread_id"]


def test_generate_tolerates_legacy_template_field(client, load_payload, monkeypatch):
    _patch_successful_generation(monkeypatch)
    payload = load_payload("generate_proposal.json")
    payload["template"] = {
        "template_id": "consultative_expert",
        "template_type": "provided",
        "template_text": "Legacy template payload from old frontend",
    }
    response = client.post("/api/v1/proposals/generate", json=payload)
    assert response.status_code == 200
    assert len(response.json()["proposals"]) == 3


def test_optimize_direct_answer_path_returns_budget_without_rewriting(client, load_payload, monkeypatch):
    import app.graph as graph

    _patch_successful_generation(monkeypatch)
    generate_response = client.post("/api/v1/proposals/generate", json=load_payload("generate_proposal.json"))
    thread_id = generate_response.json()["thread_id"]
    stored_before = get_proposals_repository().get(thread_id)
    original_alt_1 = next(proposal.text for proposal in stored_before.proposals if proposal.id == "alt_1")

    monkeypatch.setattr(
        graph,
        "_route_feedback_with_llm",
        lambda state: {"route": "direct_answer", "answer_kind": "budget", "reason": "Budget question"},
    )

    payload = load_payload("optimize_proposal_direct_answer.json")
    payload["thread_id"] = thread_id
    response = client.post("/api/v1/proposals/optimize", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["response_type"] == "direct_answer"
    assert "$3,000" in body["direct_answer"]
    assert body["model_used"] is None

    stored_after = get_proposals_repository().get(thread_id)
    updated_alt_1 = next(proposal.text for proposal in stored_after.proposals if proposal.id == "alt_1")
    assert updated_alt_1 == original_alt_1


def test_optimize_revision_updates_only_selected_proposal(client, load_payload, monkeypatch):
    import app.graph as graph

    _patch_successful_generation(monkeypatch)
    generate_response = client.post("/api/v1/proposals/generate", json=load_payload("generate_proposal.json"))
    thread_id = generate_response.json()["thread_id"]

    monkeypatch.setattr(
        graph,
        "_route_feedback_with_llm",
        lambda state: {"route": "revise_only", "answer_kind": "generic", "reason": "Simple rewrite"},
    )
    monkeypatch.setattr(graph, "_revise_proposal_with_llm", lambda state, base_text: "Updated selected proposal")

    payload = load_payload("optimize_proposal_revise.json")
    payload["thread_id"] = thread_id
    response = client.post("/api/v1/proposals/optimize", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["response_type"] == "proposal_update"
    assert body["updated_proposal"] == "Updated selected proposal"
    assert body["model_used"] is None

    stored = get_proposals_repository().get(thread_id)
    alt_1 = next(proposal.text for proposal in stored.proposals if proposal.id == "alt_1")
    alt_2 = next(proposal.text for proposal in stored.proposals if proposal.id == "alt_2")
    assert alt_1 == "Balanced proposal"
    assert alt_2 == "Updated selected proposal"


def test_task_status_endpoint_returns_completed_result(client, load_payload, monkeypatch):
    _patch_successful_generation(monkeypatch)
    generate_response = client.post("/api/v1/proposals/generate", json=load_payload("generate_proposal.json"))
    task_id = generate_response.json()["task_id"]

    response = client.get(f"/api/v1/tasks/{task_id}")
    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert "model_used" in response.json()
    assert response.json()["model_used"] is None


def test_legacy_generate_route_maps_to_generate_endpoint(client, load_payload, monkeypatch):
    _patch_successful_generation(monkeypatch)
    response = client.post("/generate_proposal", json=load_payload("generate_proposal.json"))
    assert response.status_code == 200
    assert len(response.json()["proposals"]) == 3


def test_legacy_chat_route_maps_to_optimize_endpoint(client, load_payload, monkeypatch):
    import app.graph as graph

    _patch_successful_generation(monkeypatch)
    generate_response = client.post("/api/v1/proposals/generate", json=load_payload("generate_proposal.json"))
    thread_id = generate_response.json()["thread_id"]

    monkeypatch.setattr(
        graph,
        "_route_feedback_with_llm",
        lambda state: {"route": "direct_answer", "answer_kind": "budget", "reason": "Budget question"},
    )

    payload = load_payload("optimize_proposal_direct_answer.json")
    payload["thread_id"] = thread_id
    response = client.post("/chat_proposal", json=payload)
    assert response.status_code == 200
    assert response.json()["response_type"] == "direct_answer"


def test_dual_mode_local_fallback_works_without_cloud(client, load_payload, monkeypatch):
    import app.graph as graph

    monkeypatch.setattr(graph, "_plan_vector_query_with_llm", lambda state: "fallback query")
    monkeypatch.setattr(graph, "_verify_retrieval_with_llm", lambda state: {"accepted": False, "rationale": "No match"})

    sync_response = client.post("/api/v1/portfolio/sync", json=load_payload("portfolio_sync.json"))
    assert sync_response.status_code == 200

    generate_response = client.post("/api/v1/proposals/generate", json=load_payload("generate_proposal.json"))
    assert generate_response.status_code == 200
    assert len(generate_response.json()["proposals"]) == 3


def test_payload_files_match_request_models(load_payload):
    assert PortfolioSyncRequest.model_validate(load_payload("portfolio_sync.json"))
    assert PortfolioSyncRequest.model_validate(load_payload("portfolio_structured_sync.json"))
    assert BidSyncRequest.model_validate(load_payload("bids_sync.json"))
    assert BidExampleDraftRequest.model_validate(load_payload("generate_bid_example.json"))
    assert BidExampleDraftRequest.model_validate(load_payload("update_bid_example.json"))
    assert BidExampleDraftRequest.model_validate(load_payload("bid_example_unrelated_request.json"))
    assert GenerateProposalRequest.model_validate(load_payload("generate_proposal.json"))
    aliased = GenerateProposalRequest.model_validate(load_payload("generate_proposal_job_description_alias.json"))
    assert aliased.job_details.description == "We need an expert to build secure API gateways and handle financial transactions."
    assert OptimizeProposalRequest.model_validate(load_payload("optimize_proposal_direct_answer.json"))


def test_payload_index_references_existing_files():
    index = json.loads(Path("test_payloads.json").read_text(encoding="utf-8"))
    for relative_path in index["payload_files"]:
        assert Path(relative_path).exists()
