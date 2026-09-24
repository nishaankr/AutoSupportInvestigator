"""Node 11: `resolve` (graph-design.md) — main tier, drafts a grounded resolution.

From CP4 on, `investigate` produces `evidence` (output-schema.md §3.2); `resolve` only
drafts prose against it, matching the full design (decisions.md D13 Q2 — CP3 was the one-off
exception, since it had no `investigate`). `resolve` still needs the actual historical text
behind each evidence entry (an `EvidenceEntry` carries `subject` but not the answer itself),
so it reads `retrieved_cases` for cases that came from retrieval and falls back to a direct
SQLite lookup (`_fallback_text`) for evidence sourced elsewhere (e.g. customer history).

CP4 has no `escalate` node yet (that's CP5's `assess_evidence`/`escalate` routing), so every
ticket still resolves through here — the skill is explicit that the model must say plainly
when the evidence doesn't support a fix, rather than inventing one.

`method="json_schema"`: see decisions.md D13 for why the default `with_structured_output`
method is unreliable for these two models.
"""

from __future__ import annotations

import sqlite3

from pydantic import BaseModel

from autosupport.graph.state import AgentState, DraftResponse, EvidenceEntry, RetrievedCase
from autosupport.llm import main_llm
from autosupport.skills import load_skill
from autosupport.store import db as store_db


class DraftOutput(BaseModel):
    analysis: str
    resolution: str


def resolve(state: AgentState) -> dict:
    ticket = state["ticket"]
    classification = state["classification"]
    evidence = state.get("evidence", [])
    retrieved = state.get("retrieved_cases", [])

    conn = store_db.connect()
    try:
        evidence_block = _evidence_block(evidence, retrieved, conn)
    finally:
        conn.close()

    hypothesis = state.get("hypothesis")
    hypothesis_line = f"Investigator's hypothesis: {hypothesis.statement}\n\n" if hypothesis else ""
    user_prompt = (
        f"Subject: {ticket.subject}\n\nBody: {ticket.body}\n\n"
        f"Classification: queue={classification.queue}, type={classification.type}, "
        f"priority={classification.priority}\n\n"
        f"{hypothesis_line}"
        f"Evidence gathered during investigation:\n\n{evidence_block}"
    )

    output: DraftOutput = main_llm().with_structured_output(DraftOutput, method="json_schema").invoke(
        [("system", load_skill("customer_response")), ("user", user_prompt)]
    )

    draft = DraftResponse(analysis=output.analysis, resolution=output.resolution, escalation=None)
    return {"draft": draft, "decision": "resolve"}


def _evidence_block(evidence: list[EvidenceEntry], retrieved: list[RetrievedCase], conn: sqlite3.Connection) -> str:
    if not evidence:
        return "(no evidence gathered)"
    by_id = {c.case_id: c for c in retrieved}
    blocks = []
    for e in evidence:
        rc = by_id.get(e.case_id)
        problem, answer = (rc.body_snippet, rc.answer_snippet) if rc else _fallback_text(conn, e.case_id)
        blocks.append(
            f"[{e.case_id}] stance={e.stance} answer_class={e.answer_class} cluster_size={e.cluster_size}\n"
            f"investigator's note: {e.summary}\nproblem: {problem}\nhistorical answer: {answer}"
        )
    return "\n\n".join(blocks)


def _fallback_text(conn: sqlite3.Connection, case_id: str) -> tuple[str, str]:
    """An evidence entry not backed by a `retrieved_cases` hit (e.g. sourced from customer
    history) still needs grounding text — one direct lookup rather than dropping it silently."""
    if case_id.startswith("HF-"):
        row = conn.execute("SELECT body_ix, answer_ix FROM dataset_tickets WHERE case_id = ?", (case_id,)).fetchone()
        return (row["body_ix"], row["answer_ix"]) if row else ("(unavailable)", "(unavailable)")

    row = conn.execute("SELECT body, final_output FROM cases WHERE ticket_id = ?", (case_id,)).fetchone()
    if row is None:
        return "(unavailable)", "(unavailable)"
    answer = "(no resolution on file — this case is open or was escalated)"
    if row["final_output"]:
        from autosupport.graph.state import CaseResult

        answer = CaseResult.model_validate_json(row["final_output"]).resolution
    return row["body"], answer
