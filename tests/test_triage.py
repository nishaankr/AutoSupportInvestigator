"""`triage._neighbor_agreement` (state-schema.md §2.4, F4 in the CP3 plan): the modal
queue's unweighted share of the initial retrieval round."""

from __future__ import annotations

from autosupport.graph.nodes.triage import _neighbor_agreement
from autosupport.graph.state import RetrievedCase


def _case(queue: str | None) -> RetrievedCase:
    return RetrievedCase(
        case_id="HF-1", source="dataset", subject="s", body_snippet="b", answer_snippet="a",
        queue=queue, score=0.5, similarity=0.8, retrieval_round=1, query_label="initial",
    )


def test_no_retrieved_cases_gives_zero():
    assert _neighbor_agreement([]) == 0.0


def test_unanimous_queue_gives_one():
    assert _neighbor_agreement([_case("Technical Support")] * 4) == 1.0


def test_modal_queue_share_is_unweighted():
    cases = [_case("Technical Support")] * 3 + [_case("Billing and Payments")]
    assert _neighbor_agreement(cases) == 0.75


def test_cases_with_no_queue_are_excluded_from_the_denominator():
    cases = [_case("Technical Support"), _case("Technical Support"), _case(None)]
    assert _neighbor_agreement(cases) == 2 / 3
