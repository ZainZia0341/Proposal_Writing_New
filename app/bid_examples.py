from __future__ import annotations

from typing import Iterable

from .schemas import BidExampleInput, StoredBidExample, UserBidStyleRecord


def bid_style_thread_id(user_id: str) -> str:
    return f"bids_profile#{user_id}"


def format_bid_example_markdown(example: BidExampleInput) -> str:
    job = example.job_details
    sections = [
        "## Job Title",
        job.title,
        "",
        "### Description",
        job.description,
    ]

    if job.budget:
        sections.extend(["", "### Budget", job.budget])
    if job.required_skills:
        sections.extend(["", "### Skills", ", ".join(job.required_skills)])
    if job.client_info:
        sections.extend(["", "### Client Info", job.client_info])

    sections.extend(["", "## Sent Proposal", example.proposal_text])
    return "\n".join(section.strip() if section else "" for section in sections).strip()


def build_stored_bid_examples(bids: Iterable[BidExampleInput]) -> list[StoredBidExample]:
    stored: list[StoredBidExample] = []
    for bid in bids:
        stored.append(
            StoredBidExample(
                job_details=bid.job_details,
                proposal_text=bid.proposal_text,
                markdown=format_bid_example_markdown(bid),
            )
        )
    return stored


def build_bid_style_record(user_id: str, bids: Iterable[BidExampleInput]) -> UserBidStyleRecord:
    return UserBidStyleRecord(
        thread_id=bid_style_thread_id(user_id),
        user_id=user_id,
        bids=build_stored_bid_examples(bids),
    )
