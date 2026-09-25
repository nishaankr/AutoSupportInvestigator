"""Turns the evidence a model cites into facts code can trust.

The model only says *which* case and *what stance*. Everything else — the case's subject, its
answer class, how many tickets it stands for, where it came from — is looked up here, so a
model can't misreport it, and a case id that doesn't exist is dropped with an error.
"""

from __future__ import annotations

import sqlite3

from autosupport.graph.state import EvidenceEntry, EvidenceItem, RetrievedCase

_STANCE_ORDER = {"supports": 0, "contradicts": 1, "neutral": 2}
MAX_EVIDENCE_ENTRIES = 10


def enrich(
    items: list[EvidenceItem],
    retrieved_cases: list[RetrievedCase],
    conn: sqlite3.Connection,
) -> tuple[list[EvidenceEntry], list[str]]:
    """Looks each `case_id` up in SQLite, fills `similarity` from `retrieved_cases` when the
    case came from retrieval this round, drops unknown IDs (logged, not raised — one bad
    citation shouldn't fail the whole draft), then orders and caps per §3.3."""
    similarity_by_id = {c.case_id: c.similarity for c in retrieved_cases}
    entries: list[EvidenceEntry] = []
    errors: list[str] = []
    seen: set[str] = set()
    for item in items:
        # A model sometimes lists the same case twice; the final CaseResult rejects duplicate
        # ids, which crashed persist_case after the customer had accepted. First one wins.
        if item.case_id in seen:
            continue
        seen.add(item.case_id)
        looked_up = _lookup(conn, item.case_id)
        if looked_up is None:
            errors.append(f"evidence cites unknown case_id {item.case_id!r}; dropped")
            continue
        subject, source, answer_class, cluster_size = looked_up
        entries.append(
            EvidenceEntry(
                case_id=item.case_id,
                summary=item.summary,
                stance=item.stance,
                source=source,
                subject=subject,
                answer_class=answer_class,
                cluster_size=cluster_size,
                similarity=similarity_by_id.get(item.case_id),
            )
        )
    return _order(entries)[:MAX_EVIDENCE_ENTRIES], errors


def _lookup(conn: sqlite3.Connection, case_id: str) -> tuple[str, str, str | None, int] | None:
    """Returns (subject, source, answer_class, cluster_size), or None if `case_id` resolves
    to nothing in SQLite."""
    if case_id.startswith("HF-"):
        row = conn.execute(
            "SELECT subject, answer_class, cluster_size FROM dataset_tickets WHERE case_id = ?", (case_id,)
        ).fetchone()
        if row is None:
            return None
        return row["subject"], "dataset", row["answer_class"], row["cluster_size"] or 1

    row = conn.execute("SELECT subject, status, indexed_at FROM cases WHERE ticket_id = ?", (case_id,)).fetchone()
    if row is None:
        return None
    status = row["status"]
    if status == "resolved" and row["indexed_at"] is not None:
        source, answer_class = "agent_resolved", "resolution"
    else:
        source = "customer_history"
        answer_class = "escalation" if status == "escalated" else None
    return row["subject"], source, answer_class, 1


def evidence_context(evidence: list[EvidenceEntry], retrieved: list[RetrievedCase], conn: sqlite3.Connection) -> str:
    """The historical problem and answer behind each evidence entry, as prompt text for
    `resolve` (drafting) and `verify` (checking). Taken from the retrieved snippets when the
    case was retrieved, otherwise read from SQLite (e.g. a case from the customer's history)."""
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


def _order(entries: list[EvidenceEntry]) -> list[EvidenceEntry]:
    return sorted(
        entries,
        key=lambda e: (
            _STANCE_ORDER[e.stance],
            -e.cluster_size,
            e.similarity is None,
            -(e.similarity or 0.0),
        ),
    )
