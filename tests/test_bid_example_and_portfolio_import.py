from __future__ import annotations

from app.bid_examples import build_bid_style_record, format_bid_example_markdown
from app.repositories import get_proposals_repository
from app.schemas import (
    BidExampleInput,
    PortfolioPdfParseResponse,
    ProjectRecord,
    StoredBidExample,
)


def _stored_bid(title: str = "Sample AI Automation Project", proposal_text: str = "Updated proposal") -> StoredBidExample:
    bid = BidExampleInput.model_validate(
        {
            "job_details": {
                "title": title,
                "description": "Need an AI developer to build a practical automation workflow.",
                "budget": "$2,000",
                "required_skills": ["Python", "FastAPI", "LLMs"],
                "client_info": "Sample SaaS client",
            },
            "proposal_text": proposal_text,
        }
    )
    return StoredBidExample(
        job_details=bid.job_details,
        proposal_text=bid.proposal_text,
        markdown=format_bid_example_markdown(bid),
    )


def test_bid_example_endpoint_creates_draft_without_touching_style_examples(client, load_payload, monkeypatch):
    import app.graph as graph

    style_record = build_bid_style_record("zain_zia_001", [_stored_bid("Existing Style Bid")])
    get_proposals_repository().upsert_bid_style(style_record)
    generated_bid = _stored_bid(proposal_text="Generated editable proposal")

    monkeypatch.setattr(
        graph,
        "_generate_bid_example_with_llm",
        lambda **kwargs: (generated_bid, None),
    )

    response = client.post("/api/v1/proposals/bids/example", json=load_payload("generate_bid_example.json"))
    assert response.status_code == 200
    body = response.json()
    assert body["thread_id"]
    assert body["task_id"]
    assert body["example_bid"]["proposal_text"] == "Generated editable proposal"
    assert body["direct_answer"] is None

    stored_draft = get_proposals_repository().get_bid_example_draft("zain_zia_001", body["thread_id"])
    assert stored_draft is not None
    assert stored_draft.example_bid.proposal_text == "Generated editable proposal"
    stored_style = get_proposals_repository().get_bid_style("zain_zia_001")
    assert stored_style is not None
    assert stored_style.bids[0].job_details.title == "Existing Style Bid"


def test_bid_example_endpoint_updates_existing_draft(client, load_payload, monkeypatch):
    import app.graph as graph

    first_bid = _stored_bid(proposal_text="Initial editable proposal")
    updated_bid = _stored_bid(proposal_text="Warmer updated proposal")
    monkeypatch.setattr(graph, "_generate_bid_example_with_llm", lambda **kwargs: (first_bid, None))

    create_response = client.post("/api/v1/proposals/bids/example", json=load_payload("generate_bid_example.json"))
    thread_id = create_response.json()["thread_id"]

    monkeypatch.setattr(graph, "_generate_bid_example_with_llm", lambda **kwargs: (updated_bid, None))
    payload = load_payload("update_bid_example.json")
    payload["thread_id"] = thread_id
    response = client.post("/api/v1/proposals/bids/example", json=payload)

    assert response.status_code == 200
    assert response.json()["example_bid"]["proposal_text"] == "Warmer updated proposal"
    stored_draft = get_proposals_repository().get_bid_example_draft("zain_zia_001", thread_id)
    assert stored_draft.example_bid.proposal_text == "Warmer updated proposal"


def test_bid_example_endpoint_refuses_unrelated_request(client, load_payload, monkeypatch):
    import app.graph as graph

    initial_bid = _stored_bid(proposal_text="Initial editable proposal")
    monkeypatch.setattr(graph, "_generate_bid_example_with_llm", lambda **kwargs: (initial_bid, None))
    create_response = client.post("/api/v1/proposals/bids/example", json=load_payload("generate_bid_example.json"))
    thread_id = create_response.json()["thread_id"]

    monkeypatch.setattr(graph, "_generate_bid_example_with_llm", lambda **kwargs: (None, graph.BID_EXAMPLE_REFUSAL_TEXT))
    payload = load_payload("bid_example_unrelated_request.json")
    payload["thread_id"] = thread_id
    response = client.post("/api/v1/proposals/bids/example", json=payload)

    assert response.status_code == 200
    assert response.json()["direct_answer"] == "I can only generate an example bid i can not help you with that"
    assert response.json()["example_bid"]["proposal_text"] == "Initial editable proposal"
    stored_draft = get_proposals_repository().get_bid_example_draft("zain_zia_001", thread_id)
    assert stored_draft.example_bid.proposal_text == "Initial editable proposal"


def test_bid_example_endpoint_validates_missing_and_wrong_user_threads(client, load_payload, monkeypatch):
    import app.graph as graph

    payload = load_payload("update_bid_example.json")
    payload["thread_id"] = "missing-draft"
    missing_response = client.post("/api/v1/proposals/bids/example", json=payload)
    assert missing_response.status_code == 404

    monkeypatch.setattr(graph, "_generate_bid_example_with_llm", lambda **kwargs: (_stored_bid(), None))
    create_response = client.post("/api/v1/proposals/bids/example", json=load_payload("generate_bid_example.json"))
    wrong_user_payload = load_payload("update_bid_example.json")
    wrong_user_payload["thread_id"] = create_response.json()["thread_id"]
    wrong_user_payload["user_id"] = "different_user"
    wrong_user_response = client.post("/api/v1/proposals/bids/example", json=wrong_user_payload)
    assert wrong_user_response.status_code == 404


def test_portfolio_pdf_parse_endpoint_returns_mistral_preview(client, monkeypatch):
    import app.main as main_module

    parsed_project = ProjectRecord(
        project_id="pdf_p1",
        title="PDF Project",
        description="Structured from PDF",
        tech_stack=["Python"],
        role="AI Engineer",
    )
    monkeypatch.setattr(
        main_module,
        "parse_portfolio_pdf_for_ai_dev",
        lambda user_id, file_name, content: PortfolioPdfParseResponse(
            user_id=user_id,
            projects=[parsed_project],
            extracted_markdown="# PDF Project",
            model_used="mistral:mistral-ocr-latest",
        ),
    )

    response = client.post(
        "/api/v1/portfolio/pdf/parse",
        data={"user_id": "zain_zia_001"},
        files={"file": ("portfolio.pdf", b"%PDF fake content", "application/pdf")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == "zain_zia_001"
    assert body["projects"][0]["project_id"] == "pdf_p1"
    assert body["model_used"] == "mistral:mistral-ocr-latest"


def test_structured_portfolio_sync_reuses_vector_store(client, load_payload):
    response = client.post("/api/v1/portfolio/structured/sync", json=load_payload("portfolio_structured_sync.json"))
    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == "zain_zia_001"
    assert body["stored_projects"] == 1
    assert body["namespace"] == "zain_zia_001"


def test_portfolio_pdf_parse_falls_back_when_mistral_rate_limits(monkeypatch):
    import app.document_processing as document_processing

    class MistralRateLimitError(Exception):
        status_code = 429

    parsed_project = ProjectRecord(
        project_id="fallback_p1",
        title="Fallback Project",
        description="Parsed from local PDF text",
        tech_stack=["FastAPI"],
        role="Backend AI Engineer",
    )

    monkeypatch.setattr(
        document_processing,
        "extract_projects_with_mistral_ocr",
        lambda file_name, content: (_ for _ in ()).throw(MistralRateLimitError("rate limit")),
    )
    monkeypatch.setattr(document_processing, "extract_text_from_pdf_bytes", lambda content: "Fallback project text")
    monkeypatch.setattr(document_processing, "structure_projects_from_text_with_llm", lambda text: [parsed_project])

    response = document_processing.parse_portfolio_pdf(
        user_id="zain_zia_001",
        file_name="portfolio.pdf",
        content=b"%PDF fake content",
    )
    assert response.user_id == "zain_zia_001"
    assert response.projects[0].project_id == "fallback_p1"
    assert response.extracted_markdown == "Fallback project text"
    assert response.model_used == "local:pypdf"
