"""`compute_queue_stats` (tools-and-skills.md §1) — corpus-wide pattern context (how common
a queue/type combination is), not per-ticket evidence. Canonical rows only, matching the
retrievable set (rag-design.md §4) so counts reflect distinct scenarios, not template
duplicates."""

from __future__ import annotations

from langchain_core.tools import tool

from autosupport.store import db as store_db


def _grouped(conn, column: str, where: str, params: tuple) -> dict[str, int]:
    rows = conn.execute(
        f"SELECT {column}, COUNT(*) FROM dataset_tickets WHERE is_canonical = 1 AND {where} GROUP BY {column}",
        params,
    ).fetchall()
    return dict(rows)


@tool
def compute_queue_stats(queue: str | None = None) -> dict:
    """Corpus-wide counts: how many distinct historical cases fall in `queue` (or overall, if
    omitted), broken down by type, priority and answer_class. Use this when that context
    would change your hypothesis or your confidence in it — e.g. whether this kind of ticket
    is a common, well-trodden queue or a rare one."""
    conn = store_db.connect()
    try:
        where = "queue = ?" if queue else "1 = 1"
        params = (queue,) if queue else ()
        distinct_cases = conn.execute(
            f"SELECT COUNT(*) FROM dataset_tickets WHERE is_canonical = 1 AND {where}", params
        ).fetchone()[0]
        return {
            "queue": queue or "all",
            "distinct_cases": distinct_cases,
            "by_type": _grouped(conn, "type", where, params),
            "by_priority": _grouped(conn, "priority", where, params),
            "by_answer_class": _grouped(conn, "answer_class", where, params),
        }
    finally:
        conn.close()
