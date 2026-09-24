"""The code half of `assess_evidence` (graph/assessment.py) and `verify`'s rule pass
(graph/verification.py): verdicts, next_action, escalation rules, variant selection, G1-G3."""

from __future__ import annotations

import pytest

from autosupport.graph import assessment as A
from autosupport.graph.nodes.refine_retrieval import select_variants
from autosupport.graph.state import (
    ApproachCluster, Classification, CustomerMemory, DraftResponse, EscalationDraft, EvidenceEntry,
    RetrievedCase, ToolCallRecord,
)
from autosupport.graph.verification import grounding_issues

TAU = 0.76


def case(cid, sim=0.9, queue="Technical Support", ans="resolution", size=1):
    return RetrievedCase(case_id=cid, source="dataset", subject="s", body_snippet="b", answer_snippet="a",
                         queue=queue, score=0.5, similarity=sim, cluster_size=size, answer_class=ans,
                         retrieval_round=1, query_label="initial")


def ev(cid, stance="supports", ans="resolution"):
    return EvidenceEntry(case_id=cid, summary="s", stance=stance, source="dataset", subject="s",
                         answer_class=ans, cluster_size=1, similarity=0.9)


def cluster(*ids, label="x"):
    return ApproachCluster(label=label, case_ids=list(ids))


def verdict(relevant, clusters, evidence, missing=(), contradicts=False):
    return A.verdict_for(relevant=relevant, tau_rel=TAU, clusters=clusters, evidence=evidence,
                         missing_slots=list(missing), history_contradicts=contradicts)[0]


THREE = [case("HF-1"), case("HF-2"), case("HF-3")]
SUPPORT = [ev("HF-1"), ev("HF-2")]


def test_sufficient_when_three_relevant_dominant_and_supported():
    assert verdict(THREE, [cluster("HF-1", "HF-2", "HF-3")], SUPPORT) == "sufficient"


def test_insufficient_below_three_relevant_cases():
    assert verdict(THREE[:2], [cluster("HF-1", "HF-2")], SUPPORT) == "insufficient"


def test_insufficient_when_dominant_cluster_has_fewer_than_two_supporting_entries():
    assert verdict(THREE, [cluster("HF-1", "HF-2", "HF-3")], [ev("HF-1")]) == "insufficient"


def test_insufficient_when_dominant_cluster_is_clarification_led():
    rel = [case(f"HF-{i}", ans="clarification_request") for i in (1, 2, 3)]
    assert verdict(rel, [cluster("HF-1", "HF-2", "HF-3")], SUPPORT) == "insufficient"


def test_conflicting_when_two_multi_case_clusters_split_the_weight():
    rel = [case(f"HF-{i}") for i in range(1, 5)]
    assert verdict(rel, [cluster("HF-1", "HF-2", label="a"), cluster("HF-3", "HF-4", label="b")],
                   SUPPORT) == "conflicting"


def test_conflicting_when_neighbours_disagree_on_queue():
    rel = [case("HF-1", queue="a"), case("HF-2", queue="b"), case("HF-3", queue="c")]
    assert verdict(rel, [cluster("HF-1", "HF-2", "HF-3")], SUPPORT) == "conflicting"


def test_conflicting_when_customer_history_contradicts():
    assert verdict(THREE, [cluster("HF-1", "HF-2", "HF-3")], SUPPORT, contradicts=True) == "conflicting"


def test_missing_slot_downgrades_an_otherwise_sufficient_verdict():
    assert verdict(THREE, [cluster("HF-1", "HF-2", "HF-3")], SUPPORT, missing=["version"]) == "insufficient"


def test_dominant_share_is_weighted_by_cluster_size():
    rel = [case("HF-1", size=16), case("HF-2"), case("HF-3")]
    dom, share = A.dominant_cluster([cluster("HF-1"), cluster("HF-2", "HF-3")], rel)
    assert dom.case_ids == ["HF-1"] and share == pytest.approx(5 / 7)  # w(16)=5 vs 1+1


@pytest.mark.parametrize("verdict_,gap,slots,rnd,clar,expected", [
    ("sufficient", False, [], 1, 0, "resolve"),
    ("insufficient", True, [], 1, 0, "refine_retrieval"),
    ("insufficient", True, ["v"], 3, 0, "ask_user"),          # retrieval budget spent -> ask
    ("insufficient", False, ["v"], 1, 0, "ask_user"),
    ("insufficient", False, ["v"], 3, 2, "escalate"),          # clarification budget spent
    ("conflicting", False, [], 1, 0, "escalate"),              # nothing retrievable, nothing to ask
])
def test_next_action_order(verdict_, gap, slots, rnd, clar, expected):
    assert A.next_action_for(verdict=verdict_, rule_hit=None, gap_is_retrievable=gap, missing_slots=slots,
                             retrieval_round=rnd, clarification_count=clar,
                             max_retrieval_rounds=3, max_clarifications=2) == expected


def test_a_sufficient_verdict_with_a_rule_hit_escalates():
    assert A.next_action_for(verdict="sufficient", rule_hit="x", gap_is_retrievable=False, missing_slots=[],
                             retrieval_round=1, clarification_count=0, max_retrieval_rounds=3,
                             max_clarifications=2) == "escalate"


def _rule(priority="high", tags=(), queue="Technical Support", dom=None, rel=(), human=None, log=(), profile=None):
    c = Classification(queue=queue, type="Incident", priority=priority, tags=list(tags), rationale="r",
                       neighbor_agreement=1.0)
    return A.escalation_rule_hit(classification=c, dom=dom, relevant=list(rel), requires_human_action=human,
                                 tool_log=list(log), profile=profile)


def test_escalation_rules():
    assert _rule("critical", tags=["security"]) == "critical_high_stakes"
    assert _rule("high", tags=["security"]) is None  # priority alone gates it
    rel = [case("HF-1", ans="escalation"), case("HF-2", ans="escalation"), case("HF-3")]
    assert _rule(dom=cluster("HF-1", "HF-2", "HF-3"), rel=rel) == "dominant_cluster_escalated"
    assert _rule(human="needs a refund") == "action_beyond_agent"
    flag = ToolCallRecord(name="escalate_ticket", args={}, ok=True, duration_ms=1, round=1)
    assert _rule(log=[flag]) == "action_beyond_agent"
    assert _rule(profile=CustomerMemory(customer_id="C", flags=["vip"])) == "customer_flag:vip"
    assert _rule() is None


def _state(**kw):
    base = {"classification": Classification(queue="Technical Support", type="I", priority="high", tags=[],
                                             rationale="r", neighbor_agreement=1.0),
            "retrieved_cases": [case("HF-1", ans="resolution"), case("HF-2", ans="resolution")], "clarifications": []}
    return {**base, **kw}


def test_variants_v3_always_v2_when_few_resolution_cases_and_capped_at_three(monkeypatch):
    from autosupport.graph.nodes import refine_retrieval as r
    from autosupport.graph.state import ClarificationTurn

    monkeypatch.setattr(r, "_rewrite", lambda s, a: r.QueryRewrite(text="rw", keywords=["k"]))
    labels = lambda st: [v["label"] for v in select_variants(st, {}, "ticket")]  # noqa: E731
    assert labels(_state()) == ["hypothesis_rewrite"]  # 2 resolution cases held -> no V2
    assert labels(_state(retrieved_cases=[case("HF-1", ans="escalation")])) == ["resolution_only", "hypothesis_rewrite"]
    st = _state(retrieved_cases=[case("HF-1", ans="escalation", queue="other")],
                clarifications=[ClarificationTurn(question="q", answer="It is a QNAP TS-453D")])
    assert labels(st) == ["clarification_keywords", "resolution_only", "hypothesis_rewrite"]  # V4 cut by the cap
    v1 = select_variants(st, {}, "ticket")[0]
    assert "TS-453D" in v1["phrases"] or "QNAP" in v1["phrases"]


def test_grounding_rules_g1_g2_g3():
    supports_resolution, contradicts = ev("HF-1"), ev("HF-2", stance="contradicts")
    ok = DraftResponse(analysis="See [HF-1] and [HF-2].", resolution="Do X [HF-1].")
    assert grounding_issues(ok, [supports_resolution, contradicts], "resolve") == []

    g1 = DraftResponse(analysis="[HF-1][HF-9] [HF-2]", resolution="X [HF-1]")
    assert any(i.startswith("G1") and "HF-9" in i for i in grounding_issues(g1, [supports_resolution, contradicts], "resolve"))

    g2 = DraftResponse(analysis="[HF-3]", resolution="X [HF-3]")
    clar = ev("HF-3", ans="clarification_request")
    assert any(i.startswith("G2") for i in grounding_issues(g2, [clar], "resolve"))
    assert not any(i.startswith("G2") for i in grounding_issues(g2, [clar], "escalate"))  # G2 is resolve-only

    g3 = DraftResponse(analysis="Only [HF-1].", resolution="X [HF-1]")
    assert any(i.startswith("G3") and "HF-2" in i for i in grounding_issues(g3, [supports_resolution, contradicts], "resolve"))

    handoff = DraftResponse(analysis="a", resolution="r", escalation=EscalationDraft(
        target_queue="q", reason="r", handoff_summary="see [HF-7]"))
    assert any("HF-7" in i for i in grounding_issues(handoff, [], "escalate"))
