"""Node 4: `triage` (graph-design.md, output-schema.md §2 "model" author for
`Classification`) — fast tier, structured output.

`neighbor_agreement` is a corpus fact (the modal queue's share among the initial retrieval),
so code computes it rather than asking the model for it (output-schema.md D2 "the model
can't inflate a number it never writes"; F3 in the CP3 plan) — `TriageOutput` below
deliberately omits it.

`active_skills` is written by the model from CP4 on (F6 in the CP3 plan is resolved): the
prompt is `skills/triage.md`, not a module constant, so the "removing a skill file changes
behaviour" test (checkpoints.md CP4) applies to `triage` too, not just `investigate`.
"""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, Field

from autosupport.graph.state import AgentState, Classification, Priority, RetrievedCase
from autosupport.llm import fast_llm
from autosupport.skills import load_skill
from autosupport.store import cases as cases_repo
from autosupport.store import db as store_db

# The only optional skill triage can activate at CP4 (tools-and-skills.md §2) — validated
# against this set rather than trusted verbatim, since a model-invented name would otherwise
# crash `load_skill` inside `investigate` instead of failing here, at the boundary.
_KNOWN_OPTIONAL_SKILLS = {"escalation"}


class TriageOutput(BaseModel):
    queue: str
    type: str
    priority: Priority
    tags: list[str] = Field(default_factory=list)
    rationale: str
    active_skills: list[str] = Field(default_factory=list)


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

    # method="json_schema": see decisions.md D13 for why the default ("function_calling",
    # forced tool choice) isn't reliable for these two models.
    output: TriageOutput = fast_llm().with_structured_output(TriageOutput, method="json_schema").invoke(
        [("system", load_skill("triage")), ("user", user_prompt)]
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

    active_skills = [s for s in output.active_skills if s in _KNOWN_OPTIONAL_SKILLS]
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
