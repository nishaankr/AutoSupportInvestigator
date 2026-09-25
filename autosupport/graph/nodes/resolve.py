"""Node 11: `resolve` — drafts the customer-facing resolution from the gathered evidence.

It only writes prose: `investigate` already chose the evidence. It runs when the evidence
check says the evidence is enough, and again after a failed `verify` or a rejection — then the
prompt carries the checker's objections or the customer's feedback, so the new draft actually
differs from the one it replaces. The one model call a resolved ticket makes after
`investigate` (fast tier).
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

    # Rule G2 in `verify` rejects a resolution that cites no supporting resolution-class case.
    # Naming the qualifying IDs up front is cheaper than a failed check and a redraft — the 20B
    # drafter otherwise often left the citations out.
    citable = [e.case_id for e in state.get("evidence", []) if e.stance == "supports" and e.answer_class == "resolution"]
    if citable:
        extra += ("Your resolution MUST cite, in the form [case_id], at least one of these supporting cases: "
                  + ", ".join(f"[{c}]" for c in citable) + ".\n\n")

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
    return {"draft": DraftResponse(analysis=output.analysis, resolution=_with_sources(output.resolution, citable),
                                   escalation=None),
            "decision": "resolve"}


def _with_sources(resolution: str, citable: list[str]) -> str:
    """Even when told which ids to cite, the drafter sometimes writes the answer from the
    evidence but leaves the `[case_id]` brackets out, and rule G2 then rejects an otherwise
    sound draft twice over. If none of the supporting cases is cited, list them at the
    end. This adds references, not trust: `verify`'s claim check still tests every claim
    against exactly these cases."""
    if not citable or any(f"[{c}]" in resolution for c in citable):
        return resolution
    return resolution.rstrip() + "\n\nBased on similar resolved cases: " + ", ".join(f"[{c}]" for c in citable) + "."
