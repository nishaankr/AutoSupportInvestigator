"""Node 4: `triage` (graph-design.md) — pure Python, no model call (decisions.md D19).

The retrieved neighbours already carry the dataset's own queue/type/priority/tag labels, so
classification is a vote over them, weighted by similarity and by how many historical tickets
each canonical stands for (`cluster_size`, the same weight confidence uses). A model reading
the same neighbours added a call per ticket and nothing a weighted vote doesn't; the offline
`classification_accuracy` evaluator scores this vote against held-out labels.

`active_skills` (tools-and-skills.md §2) is a rule too: the `escalation` skill joins the
investigation when escalation-class answers dominate the neighbours or the ticket touches a
high-stakes area.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from autosupport.graph.assessment import HIGH_STAKES
from autosupport.graph.confidence import cluster_weight
from autosupport.graph.state import AgentState, Classification, RetrievedCase
from autosupport.store import cases as cases_repo
from autosupport.store import db as store_db

VOTE_NEIGHBOURS = 10
TAG_SHARE_FLOOR = 0.3
MAX_TAGS = 5
ESCALATION_SKILL_SHARE = 0.4
_PRIORITY_ORDER = ("low", "medium", "high", "critical")


def _weight(case: RetrievedCase) -> float:
    return case.similarity * cluster_weight(case.cluster_size)


def vote(cases: list[RetrievedCase], field: str) -> tuple[str | None, float]:
    """Weighted modal value of `field` and its share of the total weight."""
    totals: dict[str, float] = defaultdict(float)
    for c in cases:
        if getattr(c, field):
            totals[getattr(c, field)] += _weight(c)
    if not totals:
        return None, 0.0
    winner = max(totals, key=totals.get)
    return winner, totals[winner] / sum(totals.values())


def classify(ticket, cases: list[RetrievedCase]) -> tuple[Classification, list[str]]:
    neighbours = cases[:VOTE_NEIGHBOURS]
    queue, q_share = vote(neighbours, "queue")
    type_, t_share = vote(neighbours, "type")
    priority, p_share = vote(neighbours, "priority")
    # A customer can raise the priority they report, never have it lowered below the vote.
    if ticket.customer_priority and _PRIORITY_ORDER.index(ticket.customer_priority) > _PRIORITY_ORDER.index(priority or "low"):
        priority = ticket.customer_priority

    total = sum(_weight(c) for c in neighbours) or 1.0
    tag_weight: dict[str, float] = defaultdict(float)
    for c in neighbours:
        for tag in c.tags:
            tag_weight[tag] += _weight(c)
    tags = [t for t, w in sorted(tag_weight.items(), key=lambda kv: -kv[1]) if w / total >= TAG_SHARE_FLOOR][:MAX_TAGS]
    tags = list(dict.fromkeys([*ticket.customer_tags, *tags]))

    classification = Classification(
        queue=queue or "General Inquiry", type=type_ or "Request", priority=priority or "medium", tags=tags,
        rationale=(f"Weighted vote of the {len(neighbours)} nearest historical cases: queue {queue} ({q_share:.0%}), "
                   f"type {type_} ({t_share:.0%}), priority {priority} ({p_share:.0%})."),
        neighbor_agreement=_neighbor_agreement(cases),
    )

    escalation_share = sum(_weight(c) for c in neighbours if c.answer_class == "escalation") / total
    text = f"{ticket.subject} {ticket.body} {' '.join(tags)}".lower()
    skills = ["escalation"] if escalation_share >= ESCALATION_SKILL_SHARE or any(k in text for k in HIGH_STAKES) else []
    return classification, skills


def triage(state: AgentState) -> dict:
    classification, active_skills = classify(state["ticket"], state.get("retrieved_cases", []))
    conn = store_db.connect()
    try:
        cases_repo.set_classification(conn, state["ticket_id"], classification, status="investigating")
    finally:
        conn.close()
    return {"classification": classification, "active_skills": active_skills, "status": "investigating"}


def _neighbor_agreement(retrieved: list[RetrievedCase]) -> float:
    """Unweighted share of the initial retrieval round held by its modal queue
    (state-schema.md §2.4; F4 in the CP3 plan)."""
    if not retrieved:
        return 0.0
    counts = Counter(c.queue for c in retrieved if c.queue)
    if not counts:
        return 0.0
    return counts.most_common(1)[0][1] / len(retrieved)
