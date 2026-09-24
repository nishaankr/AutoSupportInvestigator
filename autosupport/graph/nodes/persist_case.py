"""Node 15: `persist_case` (graph-design.md, output-schema.md) — builds the final
`CaseResult` and writes it to `cases`.

Every path here has passed through `verify`, so `confidence` and `verification` are always
present (output-schema.md §4.5). `status` follows `decision`; the `EscalationBlock` combines
the model's `EscalationDraft` with `escalation_trigger` (captured by `escalate`, D15 F4) and
the assessment's rule name. `unresolved_issues` is non-empty only for an escalation persisted
after `verify` retries ran out (graph-design.md §4.2). `approach` on each evidence entry is
filled from the final assessment's clusters (output-schema.md §3.2 step 3).
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
