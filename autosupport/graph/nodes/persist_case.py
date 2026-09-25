"""Node 15: `persist_case` — assembles the final `CaseResult` and saves it to `cases`.

Every route here passes through `verify` first, so confidence and the check's outcome are
always available. An escalation's block combines the handoff `escalate` assembled with the
trigger it recorded and, for a rule, the rule's name. `unresolved_issues` is filled only when
an escalation is saved after the check ran out of retries, so a person sees what never got
fixed. Each evidence entry is labelled with the approach cluster it belongs to.
"""

from __future__ import annotations

from autosupport.graph.state import (
    AgentState, CaseResult, EscalationBlock, RunStats, VerificationOutcome,
)
from autosupport.store import cases as cases_repo
from autosupport.store import db as store_db


def persist_case(state: AgentState) -> dict:
    draft, verification = state["draft"], state["verification"]
    assessment = state.get("evidence_assessment")
    escalated = state["decision"] == "escalate"
    trigger = state.get("escalation_trigger")

    if escalated:
        esc = draft.escalation
        escalation = EscalationBlock(
            required=True, trigger=trigger,
            rule=assessment.escalation_rule_hit if trigger == "rule" and assessment else None,
            target_queue=esc.target_queue, reason=esc.reason, handoff_summary=esc.handoff_summary,
        )
    else:
        escalation = EscalationBlock(required=False)

    approach_of = {cid: cl.label for cl in (assessment.clusters if assessment else []) for cid in cl.case_ids}
    evidence = [e.model_copy(update={"approach": approach_of.get(e.case_id)}) for e in state.get("evidence", [])]
    acceptance = state.get("user_acceptance")

    result = CaseResult(
        ticket_id=state["ticket_id"],
        customer_id=state["customer_id"],
        status="escalated" if escalated else "resolved",
        classification=state["classification"],
        evidence=evidence,
        analysis=draft.analysis,
        resolution=draft.resolution,
        escalation=escalation,
        confidence=state["confidence"],
        verification=VerificationOutcome(
            passed=verification.passed,
            attempts=state.get("verify_attempts", 0),
            unresolved_issues=[*verification.issues, *verification.unsupported_claims]
            if escalated and not verification.passed else [],
        ),
        acceptance=acceptance if acceptance in ("accepted", "rejected") else "not_required",
        clarifications=state.get("clarifications", []),
        stats=RunStats(
            retrieval_rounds=state.get("retrieval_round", 0),
            tool_calls=len(state.get("tool_log", [])),
            verify_attempts=state.get("verify_attempts", 0),
            revisions=state.get("revision_count", 0),
        ),
        errors=state.get("errors", []),
    )

    conn = store_db.connect()
    try:
        cases_repo.save_final(conn, state["ticket_id"], result)
    finally:
        conn.close()
    return {"final_output": result, "status": result.status}
