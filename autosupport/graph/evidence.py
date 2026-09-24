"""Enrichment: model-authored `EvidenceItem` -> code-authored `EvidenceEntry`
(output-schema.md §3.2). At the full design this lives inside `investigate`; at CP3
`resolve` calls it directly (docs/project/decisions.md CP3 entry, Q2) — the function itself
doesn't change when `investigate` lands at CP4.
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
    for item in items:
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
    to nothing in SQLite (output-schema.md §3.1)."""
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
