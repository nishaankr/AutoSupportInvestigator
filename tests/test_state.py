"""`merge_cases` and the `CaseResult` §5 validators."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from autosupport.graph.state import (
    CaseResult,
    Classification,
    Confidence,
    EscalationBlock,
    RetrievedCase,
    RunStats,
    VerificationOutcome,
    merge_cases,
)


def _case(case_id: str, similarity: float, score: float = 0.5) -> RetrievedCase:
    return RetrievedCase(
        case_id=case_id, source="dataset", subject="s", body_snippet="b", answer_snippet="a",
        score=score, similarity=similarity, retrieval_round=1, query_label="initial",
    )


def test_merge_cases_keeps_higher_similarity_copy():
    left = [_case("HF-1", similarity=0.70)]
    right = [_case("HF-1", similarity=0.85), _case("HF-2", similarity=0.60)]
    merged = merge_cases(left, right)
    by_id = {c.case_id: c for c in merged}
    assert by_id["HF-1"].similarity == 0.85
    assert set(by_id) == {"HF-1", "HF-2"}


def test_merge_cases_sorts_desc_and_caps_at_30():
    right = [_case(f"HF-{i}", similarity=i / 100) for i in range(40)]
    merged = merge_cases(None, right)
    assert len(merged) == 30
    assert [c.similarity for c in merged] == sorted((c.similarity for c in merged), reverse=True)


def _base_kwargs(**overrides):
    kwargs = dict(
        ticket_id="T-20260924-abcdef",
        customer_id="C-1",
        status="resolved",
        classification=Classification(
            queue="Technical Support", type="Incident", priority="high", tags=[], rationale="r",
            neighbor_agreement=0.8,
        ),
        evidence=[],
        analysis="a",
        resolution="r",
        escalation=EscalationBlock(required=False),
        acceptance="not_required",
        stats=RunStats(retrieval_rounds=1, tool_calls=0, verify_attempts=0, revisions=0),
        confidence=Confidence(value=0.5, support=0.5, agreement=1.0, relevance=0.5, penalty=0.0),
        verification=VerificationOutcome(passed=True, attempts=1),
    )
    kwargs.update(overrides)
    return kwargs


def test_case_result_valid_resolved_case():
    CaseResult(**_base_kwargs())  # doesn't raise


def test_status_escalated_requires_escalation_required():
    with pytest.raises(ValidationError):
        CaseResult(**_base_kwargs(status="escalated", escalation=EscalationBlock(required=False)))


def test_escalation_required_needs_all_fields_present():
    with pytest.raises(ValidationError):
        CaseResult(**_base_kwargs(status="escalated", escalation=EscalationBlock(required=True)))


def test_accepted_requires_resolved_status():
    with pytest.raises(ValidationError):
        CaseResult(**_base_kwargs(status="escalated", escalation=EscalationBlock(
            required=True, trigger="rule", rule="x", target_queue="q", reason="r", handoff_summary="h",
        ), acceptance="accepted"))
