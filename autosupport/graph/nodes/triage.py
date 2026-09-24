"""Node 4: `triage` (graph-design.md, output-schema.md §2 "model" author for
`Classification`) — fast tier, structured output.

`neighbor_agreement` is a corpus fact (the modal queue's share among the initial retrieval),
so code computes it rather than asking the model for it (output-schema.md D2 "the model
can't inflate a number it never writes"; F3 in the CP3 plan) — `TriageOutput` below
deliberately omits it.

`active_skills` is written `[]` at CP3: skill selection needs the skill files CP4 adds
(F6 in the CP3 plan). Node 4's prompt is a module constant here and moves to
`skills/triage.md` at CP4, per `tools-and-skills.md`'s per-node skill-loading rule.
"""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, Field

from autosupport.graph.state import AgentState, Classification, Priority, RetrievedCase
from autosupport.llm import fast_llm
from autosupport.store import cases as cases_repo
from autosupport.store import db as store_db

_KNOWN_QUEUES = (
    "Billing and Payments, Customer Service, General Inquiry, Human Resources, IT Support, "
    "Product Support, Returns and Exchanges, Sales and Pre-Sales, Service Outages and "
    "Maintenance, Technical Support"
)
_KNOWN_TYPES = "Incident, Problem, Request, Change"

_SYSTEM_PROMPT = f"""You triage an incoming support ticket. Pick the queue and type that best
match the corpus this agent was built against, so retrieval and reporting stay comparable
across tickets. Known queues: {_KNOWN_QUEUES}. Known types: {_KNOWN_TYPES}. Use one of these
verbatim unless the ticket genuinely fits none of them.

You are given the ticket, the customer's known profile (if any) and the metadata of the
most similar historical cases retrieved so far. Use the neighbours' queue/type/priority as a
strong prior, but not a rule — the ticket's own content wins if it clearly disagrees.

Priority is one of: low, medium, high, critical. Tags are short lower-case slugs.
`rationale` is one or two sentences a human reviewer can check against the ticket."""


class TriageOutput(BaseModel):
    queue: str
    type: str
    priority: Priority
    tags: list[str] = Field(default_factory=list)
    rationale: str


def triage(state: AgentState) -> dict:
    ticket = state["ticket"]
    retrieved = state.get("retrieved_cases", [])
    profile = state.get("customer_profile")

    neighbour_block = "\n".join(
        f"- {c.case_id}: queue={c.queue}, type={c.type}, priority={c.priority}, similarity={c.similarity:.2f}"
        for c in retrieved[:10]
    ) or "(no retrieved neighbours)"
    profile_block = (
        f"facts={profile.facts}, flags={profile.flags}" if profile else "(no prior profile for this customer)"
    )

    user_prompt = (
        f"Subject: {ticket.subject}\n\nBody: {ticket.body}\n\n"
        f"Customer-supplied priority: {ticket.customer_priority or '(none)'}\n"
        f"Customer-supplied tags: {ticket.customer_tags or '(none)'}\n\n"
        f"Customer profile: {profile_block}\n\n"
        f"Retrieved neighbours:\n{neighbour_block}"
    )

    # method="json_schema": see resolve.py's module docstring for why the default
    # ("function_calling", forced tool choice) isn't reliable for this model.
    output: TriageOutput = fast_llm().with_structured_output(TriageOutput, method="json_schema").invoke(
        [("system", _SYSTEM_PROMPT), ("user", user_prompt)]
    )

    classification = Classification(
        queue=output.queue,
        type=output.type,
        priority=output.priority,
        tags=output.tags,
        rationale=output.rationale,
        neighbor_agreement=_neighbor_agreement(retrieved),
    )

    conn = store_db.connect()
    try:
        cases_repo.set_classification(conn, state["ticket_id"], classification, status="investigating")
    finally:
        conn.close()

    return {"classification": classification, "active_skills": [], "status": "investigating"}


def _neighbor_agreement(retrieved: list[RetrievedCase]) -> float:
    """Unweighted share of the initial retrieval round held by its modal queue
    (state-schema.md §2.4; F4 in the CP3 plan)."""
    if not retrieved:
        return 0.0
    counts = Counter(c.queue for c in retrieved if c.queue)
    if not counts:
        return 0.0
    return counts.most_common(1)[0][1] / len(retrieved)
