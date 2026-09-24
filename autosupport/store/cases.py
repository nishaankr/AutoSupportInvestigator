"""Repository for the `cases` table (case-persistence.md)."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from autosupport.graph.state import CaseResult, CaseStatus, CaseSummary, Classification, TicketInput


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def insert_open(
    conn: sqlite3.Connection, ticket_id: str, customer_id: str, thread_id: str, ticket: TicketInput
) -> None:
    """The only `INSERT` into `cases`. Uses `ON CONFLICT` rather than a bare insert so a
    follow-up on an already-closed ticket (state-schema.md §5: "reopens the same thread")
    resets `status` without discarding `subject`/`body`/`created_at` — case-persistence.md
    §3 "Idempotency"."""
    now = _now()
    conn.execute(
        """
        INSERT INTO cases (ticket_id, customer_id, thread_id, status, subject, body,
                            submitted_at, created_at, updated_at)
        VALUES (?, ?, ?, 'open', ?, ?, ?, ?, ?)
        ON CONFLICT(ticket_id) DO UPDATE SET status='open', updated_at=excluded.updated_at
        """,
        (ticket_id, customer_id, thread_id, ticket.subject, ticket.body, ticket.submitted_at.isoformat(), now, now),
    )
    conn.commit()


def set_classification(conn: sqlite3.Connection, ticket_id: str, classification: Classification, status: CaseStatus) -> None:
    conn.execute(
        "UPDATE cases SET classification = ?, status = ?, updated_at = ? WHERE ticket_id = ?",
        (classification.model_dump_json(), status, _now(), ticket_id),
    )
    conn.commit()


def save_final(conn: sqlite3.Connection, ticket_id: str, result: CaseResult) -> None:
    conn.execute(
        "UPDATE cases SET final_output = ?, status = ?, updated_at = ? WHERE ticket_id = ?",
        (result.model_dump_json(), result.status, _now(), ticket_id),
    )
    conn.commit()


def get(conn: sqlite3.Connection, ticket_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM cases WHERE ticket_id = ?", (ticket_id,)).fetchone()


def history_for(conn: sqlite3.Connection, customer_id: str, exclude_ticket_id: str, limit: int = 5) -> list[CaseSummary]:
    """The customer's open cases plus their last `limit` terminal (resolved/escalated)
    cases (state-schema.md §2.3), excluding the ticket that was just opened by `intake`
    for this very run (memory-design.md §3)."""
    terminal = conn.execute(
        "SELECT ticket_id, status, subject, classification, final_output, updated_at FROM cases "
        "WHERE customer_id = ? AND ticket_id != ? AND status IN ('resolved', 'escalated') "
        "ORDER BY updated_at DESC LIMIT ?",
        (customer_id, exclude_ticket_id, limit),
    ).fetchall()
    open_cases = conn.execute(
        "SELECT ticket_id, status, subject, classification, final_output, updated_at FROM cases "
        "WHERE customer_id = ? AND ticket_id != ? AND status IN ('open', 'investigating', 'awaiting_user') "
        "ORDER BY updated_at DESC",
        (customer_id, exclude_ticket_id),
    ).fetchall()
    return [_to_summary(row) for row in (*open_cases, *terminal)]


def _to_summary(row: sqlite3.Row) -> CaseSummary:
    queue = None
    if row["classification"]:
        queue = Classification.model_validate_json(row["classification"]).queue
    resolution_snippet = None
    if row["final_output"] and row["status"] == "resolved":
        result = CaseResult.model_validate_json(row["final_output"])
        resolution_snippet = result.resolution[:200]
    return CaseSummary(
        case_id=row["ticket_id"],
        status=row["status"],
        subject=row["subject"],
        queue=queue,
        resolution_snippet=resolution_snippet,
        updated_at=row["updated_at"],
    )
