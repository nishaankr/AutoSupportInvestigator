"""Node 15: `persist_case` (graph-design.md, output-schema.md) — builds the final
`CaseResult` and writes it to `cases`.

At CP4 every run still reaches here with `decision == "resolve"` (F8 in the CP3 plan: there's
no `escalate` node yet), `confidence`/`verification` unset (Q1: required again from CP5, once
`verify` is the gate every path passes through), and `acceptance == "not_required"` (F7:
`confirm_resolution` never runs). `stats.tool_calls` counts `tool_log`, which the
`investigate <-> tools` loop populates from CP4 on; `verify_attempts`/`revisions` stay 0
until CP5 adds those loops.
"""

from __future__ import annotations

from autosupport.graph.state import AgentState, CaseResult, EscalationBlock, RunStats
from autosupport.store import cases as cases_repo
from autosupport.store import db as store_db

_NO_ESCALATION = EscalationBlock(required=False)


def persist_case(state: AgentState) -> dict:
    draft = state["draft"]
    result = CaseResult(
        ticket_id=state["ticket_id"],
        customer_id=state["customer_id"],
        status="resolved",
        classification=state["classification"],
        evidence=state.get("evidence", []),
        analysis=draft.analysis,
        resolution=draft.resolution,
        escalation=_NO_ESCALATION,
        confidence=None,
        verification=None,
        acceptance="not_required",
        clarifications=state.get("clarifications", []),
        stats=RunStats(
            retrieval_rounds=state.get("retrieval_round", 0),
            tool_calls=len(state.get("tool_log", [])),
            verify_attempts=0,
            revisions=0,
        ),
        errors=state.get("errors", []),
    )

    conn = store_db.connect()
    try:
        cases_repo.save_final(conn, state["ticket_id"], result)
    finally:
        conn.close()

    return {"final_output": result, "status": "resolved"}
