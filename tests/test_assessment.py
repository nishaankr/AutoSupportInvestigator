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


def test_noisy_queue_labels_no_longer_make_a_conflict():
    # D21: the dataset's queue labels disagree on the same problem; that isn't competing fixes.
    rel = [case("HF-1", queue="a"), case("HF-2", queue="b"), case("HF-3", queue="c")]
    assert verdict(rel, [cluster("HF-1", "HF-2", "HF-3")], SUPPORT) == "sufficient"


def test_non_fix_clusters_do_not_compete_but_two_fixes_do():
    rel = [case("HF-1"), case("HF-2"), case("HF-3"), case("HF-4", ans="clarification_request"),
           case("HF-5", ans="clarification_request"), case("HF-6", ans="escalation"), case("HF-7", ans="escalation")]
    asked = [cluster("HF-1", "HF-2", "HF-3"), cluster("HF-4", "HF-5", label="ask"), cluster("HF-6", "HF-7", label="esc")]
    assert verdict(rel, asked, SUPPORT) == "sufficient"  # one fix; the rest are non-answers
    rival = [case("HF-1"), case("HF-2"), case("HF-3"), case("HF-4")]
    assert verdict(rival, [cluster("HF-1", "HF-2"), cluster("HF-3", "HF-4", label="other fix")],
                   SUPPORT) == "conflicting"  # 50/50 between two fixes
    assert verdict(rel[3:], [cluster("HF-4", "HF-5"), cluster("HF-6", "HF-7")],
                   [ev("HF-4"), ev("HF-5")]) == "insufficient"  # no cluster proposes a fix


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

    labels = lambda st: [v["label"] for v in select_variants(st, {}, "ticket")]  # noqa: E731
    assert labels(_state()) == ["hypothesis_rewrite"]  # 2 resolution cases held -> no V2
    assert labels(_state(retrieved_cases=[case("HF-1", ans="escalation")])) == ["resolution_only", "hypothesis_rewrite"]
    st = _state(retrieved_cases=[case("HF-1", ans="escalation", queue="other")],
                clarifications=[ClarificationTurn(question="q", answer="It is a QNAP TS-453D")])
    assert labels(st) == ["clarification_keywords", "resolution_only", "hypothesis_rewrite"]  # V4 cut by the cap
    v1 = select_variants(st, {}, "ticket")[0]
    assert "TS-453D" in v1["phrases"] or "QNAP" in v1["phrases"]


def test_v3_query_is_built_from_the_hypothesis_without_a_model():
    from autosupport.graph.nodes.refine_retrieval import hypothesis_query
    from autosupport.graph.state import Hypothesis

    assert hypothesis_query({}, "ticket") == ("ticket", [])  # no hypothesis yet: the ticket itself
    st = {"hypothesis": Hypothesis(statement="SMB2 disabled after QNAP QTS 5.1 update", root_cause_category="config",
                                   supporting_case_ids=[]),
          "evidence": [ev("HF-1")]}
    text, keywords = hypothesis_query(st, "ticket")
    assert text.startswith("config: SMB2 disabled") and {"SMB2", "QNAP", "QTS"} <= set(keywords)


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


def test_a_matched_rule_escalates_before_refining_or_asking():
    # CP7 eval: rule-hit tickets were asking the customer instead of handing off (D18).
    assert A.next_action_for(verdict="insufficient", rule_hit="action_beyond_agent", gap_is_retrievable=True,
                             missing_slots=["error text"], retrieval_round=1, clarification_count=0,
                             max_retrieval_rounds=3, max_clarifications=2) == "escalate"


def test_investigate_context_shows_top_graph_cases_with_short_snippets():
    from autosupport.graph.nodes.investigate import _context_block
    from autosupport.graph.retrieval import PROMPT_CASES, PROMPT_SNIPPET_CHARS
    from autosupport.graph.state import Classification, RetrievedCase, TicketInput

    def case(i, label):
        return RetrievedCase(case_id=f"HF-{i}", source="dataset", subject="s", body_snippet="b" * 600,
                             answer_snippet="a" * 600, score=0.1, similarity=0.9 - i / 1000,
                             retrieval_round=1, query_label=label)
    state = {"ticket": TicketInput(subject="s", body="b"),
             "classification": Classification(queue="q", type="t", priority="low", tags=[], rationale="r",
                                              neighbor_agreement=1.0),
             "retrieved_cases": [case(0, "tool:search_similar_tickets")] + [case(i, "initial") for i in range(1, 20)]}
    block = _context_block(state)
    assert "[HF-0]" not in block  # tool hits live in their tool result, not the cached system block
    assert block.count("[HF-") == PROMPT_CASES
    assert "b" * (PROMPT_SNIPPET_CHARS + 1) not in block


def test_askable_slots_drop_secrets_and_cap_at_two():
    # D20: a live run asked the customer for their S3 access key and secret key.
    slots = ["Exact error message", "S3 credentials (access key/secret key)", "Bucket region", "Bucket policy"]
    assert A.askable_slots(slots) == ["Exact error message", "Bucket region"]
    assert A.askable_slots(["Your password", "API token"]) == []
