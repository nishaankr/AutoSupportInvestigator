"""`triage._neighbor_agreement`: the modal
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


def _labelled(cid, queue, type_, priority, sim, answer_class="resolution", tags=(), cluster_size=1):
    return RetrievedCase(case_id=cid, source="dataset", subject="s", body_snippet="b", answer_snippet="a",
                         queue=queue, type=type_, priority=priority, tags=list(tags), score=0.1, similarity=sim,
                         cluster_size=cluster_size, answer_class=answer_class, retrieval_round=1, query_label="initial")


def test_classification_is_a_weighted_vote_over_neighbour_labels():
    from autosupport.graph.nodes.triage import classify
    from autosupport.graph.state import TicketInput

    cases = [
        _labelled("HF-1", "IT Support", "Incident", "high", 0.90, tags=["VPN", "Network"], cluster_size=8),
        _labelled("HF-2", "Technical Support", "Request", "low", 0.85, tags=["VPN"]),
        _labelled("HF-3", "Technical Support", "Request", "low", 0.80),
    ]
    classification, skills = classify(TicketInput(subject="VPN drops", body="keeps disconnecting"), cases)
    # HF-1 stands for 8 historical tickets, so it outweighs two single cases (no LLM involved).
    assert (classification.queue, classification.type, classification.priority) == ("IT Support", "Incident", "high")
    assert classification.tags[0] == "VPN" and skills == []
    assert "Weighted vote" in classification.rationale


def test_customer_can_raise_priority_and_escalation_skill_follows_rules():
    from autosupport.graph.nodes.triage import classify
    from autosupport.graph.state import TicketInput

    cases = [_labelled("HF-1", "IT Support", "Incident", "low", 0.9, answer_class="escalation")]
    classification, skills = classify(TicketInput(subject="x", body="y", customer_priority="high"), cases)
    assert classification.priority == "high" and skills == ["escalation"]  # escalation-class neighbours
    _, skills = classify(TicketInput(subject="Possible data breach", body="y"), [_labelled("HF-2", "q", "t", "low", 0.9)])
    assert skills == ["escalation"]  # high-stakes keyword
