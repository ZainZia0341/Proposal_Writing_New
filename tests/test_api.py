from __future__ import annotations

import json
from pathlib import Path

from app.repositories import get_proposals_repository
from app.schemas import GenerateProposalRequest, OptimizeProposalRequest, UserSignupRequest


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


def test_signup_selected_template_persists_profile_and_projects(client, load_payload, monkeypatch):
    import app.services as services

    monkeypatch.setattr(services, "llm_available", lambda: False)
    payload = load_payload("signup_selected_template.json")
    response = client.post("/api/v1/signup", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["stored_projects"] == 2
    assert body["template"]["template_id"] == "consultative_expert"
    assert body["namespace"] == payload["user_id"]
    assert body["user"]["portfolio_projects"][0]["project_id"] == "p1"


def test_signup_custom_template_stores_custom_template_id(client, load_payload, monkeypatch):
    import app.services as services

    monkeypatch.setattr(services, "llm_available", lambda: False)
    payload = load_payload("signup_custom_template.json")
    response = client.post("/api/v1/signup", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["user"]["template_id"] == "custom-template-1"
    assert body["user"]["custom_template_text"] is not None


def test_signup_requires_frontend_user_id(client, load_payload):
    payload = load_payload("signup_custom_template.json")
    payload.pop("user_id", None)

    response = client.post("/api/v1/signup", json=payload)
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any(error["type"] == "signup_user_id_required" for error in errors)
    assert any("user_id is required for signup" in error["msg"] for error in errors)


def test_signup_rejects_multiple_template_sources(client, load_payload):
    payload = load_payload("signup_selected_template.json")
    payload["custom_template_text"] = "Custom style"
    payload["ai_template_context"] = "Make it more confident"

    response = client.post("/api/v1/signup", json=payload)
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any(error["type"] == "signup_template_source_conflict" for error in errors)
    assert any("Provide only one template source" in error["msg"] for error in errors)


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


def test_dual_mode_local_fallback_works_without_cloud(client, load_payload, monkeypatch):
    import app.graph as graph
    import app.services as services

    monkeypatch.setattr(services, "llm_available", lambda: False)
    monkeypatch.setattr(graph, "_plan_vector_query_with_llm", lambda state: "fallback query")
    monkeypatch.setattr(graph, "_verify_retrieval_with_llm", lambda state: {"accepted": False, "rationale": "No match"})

    signup_response = client.post("/api/v1/signup", json=load_payload("signup_selected_template.json"))
    assert signup_response.status_code == 200

    generate_response = client.post("/api/v1/proposals/generate", json=load_payload("generate_proposal.json"))
    assert generate_response.status_code == 200
    assert len(generate_response.json()["proposals"]) == 3


def test_payload_files_match_request_models(load_payload):
    assert UserSignupRequest.model_validate(load_payload("signup_selected_template.json"))
    assert UserSignupRequest.model_validate(load_payload("signup_custom_template.json"))
    assert GenerateProposalRequest.model_validate(load_payload("generate_proposal.json"))
    aliased = GenerateProposalRequest.model_validate(load_payload("generate_proposal_job_description_alias.json"))
    assert aliased.job_details.description == "We need an expert to build secure API gateways and handle financial transactions."
    assert OptimizeProposalRequest.model_validate(load_payload("optimize_proposal_direct_answer.json"))


def test_payload_index_references_existing_files():
    index = json.loads(Path("test_payloads.json").read_text(encoding="utf-8"))
    for relative_path in index["payload_files"]:
        assert Path(relative_path).exists()
