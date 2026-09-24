"""Node 11: `resolve` (graph-design.md) — fast tier (decisions.md D19), drafts a grounded
resolution. The one model call a resolved ticket makes after `investigate`.

`investigate` produces `evidence` (output-schema.md §3.2); `resolve` only drafts prose
against it (decisions.md D13 Q2). It is reached only when `assess_evidence` judged the
evidence sufficient — and again after a failed `verify` or a rejected resolution, in which
case the prompt carries the verifier's issues / the customer's feedback so the revision
actually differs from the draft it replaces (§7.2 acceptance loop, loop D).

`method="json_schema"`: see decisions.md D13 for why the default `with_structured_output`
method is unreliable for these two models.
"""

from __future__ import annotations

from pydantic import BaseModel

from autosupport.graph.evidence import evidence_context
from autosupport.graph.memory import profile_block
from autosupport.graph.state import AgentState, DraftResponse
from autosupport.llm import fast_llm, structured
from autosupport.skills import load_skill
from autosupport.store import db as store_db


class DraftOutput(BaseModel):
    analysis: str
    resolution: str


def resolve(state: AgentState) -> dict:
    ticket, classification = state["ticket"], state["classification"]
    conn = store_db.connect()
    try:
        evidence_block = evidence_context(state.get("evidence", []), state.get("retrieved_cases", []), conn)
    finally:
        conn.close()

    hypothesis, verification = state.get("hypothesis"), state.get("verification")
    extra = ""
    if hypothesis:
        extra += f"Investigator's hypothesis: {hypothesis.statement}\n\n"
    if state.get("clarifications"):
        extra += "The customer has told us:\n" + "\n".join(
            f"Q: {t.question}\nA: {t.answer}" for t in state["clarifications"]) + "\n\n"
    if state.get("user_acceptance") == "rejected" and state.get("user_feedback"):
        extra += f"The customer REJECTED the previous resolution: {state['user_feedback']}\nAddress this directly.\n\n"
    if verification and not verification.passed:
        extra += "A reviewer rejected the previous draft — fix these: " + "; ".join(
            [*verification.issues, *verification.unsupported_claims]) + "\n\n"

    # Long-term memory shapes the draft: honour stated preferences, don't re-offer a tried fix.
    memory = profile_block(state.get("customer_profile"), state.get("customer_history", []))
    user_prompt = (
        f"Subject: {ticket.subject}\n\nBody: {ticket.body}\n\n"
        f"Classification: queue={classification.queue}, type={classification.type}, "
        f"priority={classification.priority}\n\nCustomer memory:\n{memory}\n\n{extra}"
        f"Evidence gathered during investigation:\n\n{evidence_block}"
    )
    output: DraftOutput = structured(fast_llm(), DraftOutput).invoke(
        [("system", load_skill("customer_response")), ("user", user_prompt)]
    )
    draft = DraftResponse(analysis=output.analysis, resolution=output.resolution, escalation=None)
    return {"draft": draft, "decision": "resolve"}
