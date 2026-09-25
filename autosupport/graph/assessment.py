"""The rules behind `assess_evidence` (graph-design.md §4.2, §5): which cases are relevant, how
strongly they agree, the verdict, which escalation rule fires, and what happens next.

All pure functions. The investigator supplies only what code can't work out — how the cases
group into approaches, what fact is missing, whether a better search could help — and never
a verdict or a number. Every decision about the ticket's route is made here, in code."""

from __future__ import annotations

import re
from collections import Counter

from autosupport.graph.confidence import cluster_weight
from autosupport.graph.state import (
    ApproachCluster, Classification, CustomerMemory, EvidenceEntry, NextAction, RetrievedCase,
    ToolCallRecord, Verdict,
)

MIN_RELEVANT = 3
DOMINANT_SHARE_FLOOR = 0.6
MIN_SUPPORTING_IN_DOMINANT = 2
ESCALATED_SHARE_FLOOR = 0.6
CLARIFICATION_SHARE_FLOOR = 0.6
QUEUE_AGREEMENT_FLOOR = 0.5  # only triggers retrieval variant V4 now, not a verdict (D21)
FIX_CLUSTER_FLOOR = 0.5  # a cluster "proposes a fix" if at least half its weight is resolution-class

# `critical` priority alone isn't enough (graph-design.md §4.2): it has to point at one of
# these. Matched against the classification's queue, type and tags.
HIGH_STAKES = ("security", "breach", "data loss", "data-loss", "outage", "legal", "compliance", "gdpr")
_ESCALATION_FLAGS = ("vip", "repeat_unresolved")


def relevant_cases(retrieved: list[RetrievedCase], tau_rel: float) -> list[RetrievedCase]:
    return sorted((c for c in retrieved if c.similarity >= tau_rel), key=lambda c: c.similarity, reverse=True)


def top_queue_share(relevant: list[RetrievedCase]) -> float:
    """Unweighted modal-queue share over the *current* relevant cases — recomputed each
    assessment rather than reusing triage's frozen round-1 `neighbor_agreement`, which
    refinement could otherwise never move (D15 F3)."""
    counts = Counter(c.queue for c in relevant if c.queue)
    return counts.most_common(1)[0][1] / len(relevant) if counts and relevant else 0.0


def dominant_cluster(
    clusters: list[ApproachCluster], relevant: list[RetrievedCase]
) -> tuple[ApproachCluster | None, float]:
    """Largest cluster by `cluster_weight`, and its share of the relevant cases' total weight
    (D3: "15 cases against 3" reads differently from "15 against 14")."""
    weights = {c.case_id: cluster_weight(c.cluster_size) for c in relevant}
    total = sum(weights.values())
    if not clusters or total == 0:
        return None, 0.0
    scored = [(sum(weights.get(cid, 0.0) for cid in cl.case_ids), cl) for cl in clusters]
    weight, best = max(scored, key=lambda pair: pair[0])
    return best, weight / total


def _answer_class_share(ids: list[str], relevant: list[RetrievedCase], answer_class: str) -> float:
    by_id = {c.case_id: c for c in relevant}
    members = [by_id[i] for i in ids if i in by_id]
    total = sum(cluster_weight(c.cluster_size) for c in members)
    hit = sum(cluster_weight(c.cluster_size) for c in members if c.answer_class == answer_class)
    return hit / total if total else 0.0


def verdict_for(
    *, relevant: list[RetrievedCase], tau_rel: float, clusters: list[ApproachCluster],
    evidence: list[EvidenceEntry], missing_slots: list[str], history_contradicts: bool,
) -> tuple[Verdict, float, ApproachCluster | None, str]:
    """graph-design.md §5, checked insufficient -> conflicting -> sufficient. Returns
    (verdict, dominant_share, dominant_cluster, why). `why` is the code's own reason in plain
    words; it goes into the escalation handoff, so a person sees which test failed rather than
    the model's opinion of the evidence, which can disagree with the verdict (D22)."""
    dom, share = dominant_cluster(clusters, relevant)
    top = relevant[0].similarity if relevant else 0.0
    supporting = 0
    if dom:
        supporting = sum(1 for e in evidence if e.stance == "supports" and e.case_id in set(dom.case_ids))

    if len(relevant) < MIN_RELEVANT:
        return "insufficient", share, dom, f"only {len(relevant)} similar historical case(s) found; {MIN_RELEVANT} needed"
    if top < tau_rel:
        return "insufficient", share, dom, "no historical case is close enough to this ticket"
    if supporting < MIN_SUPPORTING_IN_DOMINANT:
        return "insufficient", share, dom, (
            f"only {supporting} case(s) support the most common approach; {MIN_SUPPORTING_IN_DOMINANT} needed")
    # A dominant cluster that is mostly clarification requests tells us what to ask, not how to
    # fix it (D15 F5), so it never counts as enough to resolve.
    if _answer_class_share(dom.case_ids, relevant, "clarification_request") >= CLARIFICATION_SHARE_FLOOR:
        return "insufficient", share, dom, "similar tickets were mostly answered with questions, not fixes"

    # A conflict is two *fixes* competing (D21). Clusters of "asked for more info" or
    # "escalated" answers aren't rival approaches, and the neighbours' queue labels are too
    # noisy to signal conflict: 3 of 5 resolution-class eval tickets had an all-resolution
    # dominant cluster yet were called conflicting by those two tests.
    fixes = [cl for cl in clusters if _answer_class_share(cl.case_ids, relevant, "resolution") >= FIX_CLUSTER_FLOOR]
    if not fixes:
        return "insufficient", share, dom, "none of the similar tickets was resolved with a fix"
    weights = {c.case_id: cluster_weight(c.cluster_size) for c in relevant}
    fix_weight = [sum(weights.get(i, 0.0) for i in cl.case_ids) for cl in fixes]
    if len(fixes) >= 2 and max(fix_weight) / sum(fix_weight) < DOMINANT_SHARE_FLOOR:
        return "conflicting", share, dom, "similar tickets were fixed in different ways, with no clear winner"
    if history_contradicts:
        return "conflicting", share, dom, "the customer's own history contradicts the usual fix"
    if missing_slots:
        return "insufficient", share, dom, "a fact only the customer has is still missing: " + "; ".join(missing_slots)
    return "sufficient", share, dom, "enough similar resolved tickets agree on one fix"


def escalation_rule_hit(
    *, classification: Classification, dom: ApproachCluster | None, relevant: list[RetrievedCase],
    requires_human_action: str | None, tool_log: list[ToolCallRecord], profile: CustomerMemory | None,
) -> str | None:
    """First matching rule name (graph-design.md §4.2), stored on the assessment and later
    surfaced as `EscalationBlock.rule`. A match escalates straight away, whatever the verdict
    (`next_action_for`, D18)."""
    haystack = " ".join([classification.queue, classification.type, *classification.tags]).lower()
    if classification.priority == "critical" and any(k in haystack for k in HIGH_STAKES):
        return "critical_high_stakes"
    if dom and _answer_class_share(dom.case_ids, relevant, "escalation") >= ESCALATED_SHARE_FLOOR:
        return "dominant_cluster_escalated"
    if requires_human_action or any(t.name == "escalate_ticket" and t.ok for t in tool_log):
        return "action_beyond_agent"
    if profile:
        for flag in _ESCALATION_FLAGS:
            if flag in profile.flags:
                return f"customer_flag:{flag}"
    return None


MAX_MISSING_SLOTS = 2
# Never ask a customer to send these in a ticket, whatever the model proposes (D20: measured,
# GPT-OSS 20B asked for an S3 access key and secret key).
_SECRET_SLOT = re.compile(r"password|passcode|secret|access[ _-]?key|api[ _-]?key|token|credential|private key", re.I)


def askable_slots(missing_slots: list[str]) -> list[str]:
    """The model's missing facts, minus anything secret, capped at `MAX_MISSING_SLOTS` —
    the prompt asks for both, code guarantees them."""
    return [s for s in missing_slots if not _SECRET_SLOT.search(s)][:MAX_MISSING_SLOTS]


# A ticket this short rarely says what product, what symptom or since when. When its evidence
# isn't enough and the investigator named no missing fact, ask once instead of escalating an
# under-specified ticket (D22: the demo's "my app stopped syncing" ticket was escalated
# unasked in one run and asked in another).
THIN_TICKET_WORDS = 25
THIN_TICKET_SLOTS = ["which product, app or feature is affected",
                     "what exactly happens (any error message) and since when"]


def thin_ticket_slots(body: str, verdict: Verdict, slots: list[str]) -> list[str]:
    if slots or verdict == "sufficient" or len(body.split()) >= THIN_TICKET_WORDS:
        return slots
    return THIN_TICKET_SLOTS


def next_action_for(
    *, verdict: Verdict, rule_hit: str | None, gap_is_retrievable: bool, missing_slots: list[str],
    retrieval_round: int, clarification_count: int, max_retrieval_rounds: int, max_clarifications: int,
) -> NextAction:
    """graph-design.md §4.2 order. A matched escalation rule wins outright: it names work a human
    must do, so asking the customer or searching again first only delays the handoff (D18;
    CP7 eval: rule-hit tickets were asking instead). Then retrieval before asking the user:
    re-retrieval is cheap and asks nothing of the customer. Every branch that can't progress
    lands on `escalate`, so the loop always terminates (§6)."""
    if rule_hit:
        return "escalate"
    if verdict == "sufficient":
        return "resolve"
    if gap_is_retrievable and retrieval_round < max_retrieval_rounds:
        return "refine_retrieval"
    if missing_slots and clarification_count < max_clarifications:
        return "ask_user"
    return "escalate"
