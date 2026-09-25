"""Repository for the `dataset_tickets` table and its FTS5 index."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pandas as pd

_TAG_COLUMNS = [f"tag_{i}" for i in range(1, 9)]
_INSERT_COLUMNS = [
    "case_id", "source", "hf_row", "subject", "body", "answer",
    "subject_ix", "body_ix", "answer_ix", "title", "title_is_derived",
    "queue", "type", "priority", "language", "version",
    *_TAG_COLUMNS,
    "answer_class", "answer_class_source", "below_content_threshold",
    "is_canonical", "canonical_of", "cluster_size", "ingested_at",
]


def insert_all(conn: sqlite3.Connection, records: pd.DataFrame) -> None:
    """`records` must carry every column in `_INSERT_COLUMNS` except `ingested_at`, which
    is stamped here. Replaces on conflict so `--rebuild` and re-running `--limit` are both
    idempotent."""
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        tuple(row.get(c, None) if c != "ingested_at" else now for c in _INSERT_COLUMNS)
        for row in records.to_dict("records")
    ]
    placeholders = ", ".join("?" for _ in _INSERT_COLUMNS)
    conn.executemany(
        f"INSERT OR REPLACE INTO dataset_tickets ({', '.join(_INSERT_COLUMNS)}) VALUES ({placeholders})",
        rows,
    )
    conn.commit()


def rebuild_fts(conn: sqlite3.Connection) -> None:
    """Repopulates the dataset half of `dataset_tickets_fts` from canonical rows only — the
    retrievable set, matching what gets embedded into Chroma, so RRF never double-counts a
    near-duplicate. Agent-resolved rows are left alone."""
    conn.execute("DELETE FROM dataset_tickets_fts WHERE source = 'dataset'")
    rows = conn.execute(
        "SELECT case_id, source, subject_ix, body_ix, answer_ix, queue, type, answer_class, "
        "tag_1, tag_2, tag_3, tag_4, tag_5, tag_6, tag_7, tag_8 "
        "FROM dataset_tickets WHERE is_canonical = 1"
    ).fetchall()
    conn.executemany(
        "INSERT INTO dataset_tickets_fts (subject, body, answer, tags, case_id, source, queue, type, answer_class) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                r["subject_ix"], r["body_ix"], r["answer_ix"],
                " ".join(r[t].lower() for t in _TAG_COLUMNS if r[t]),
                r["case_id"], r["source"], r["queue"], r["type"], r["answer_class"],
            )
            for r in rows
        ],
    )
    conn.commit()


def add_fts_row(
    conn: sqlite3.Connection, case_id: str, source: str, subject: str, body: str, answer: str,
    tags: list[str], queue: str, type_: str, answer_class: str,
) -> None:
    """One row into the shared FTS5 index — used for agent-resolved cases (`index_case`).
    Replaces any existing row for `case_id` so a forced re-index can't duplicate it."""
    conn.execute("DELETE FROM dataset_tickets_fts WHERE case_id = ?", (case_id,))
    conn.execute(
        "INSERT INTO dataset_tickets_fts (subject, body, answer, tags, case_id, source, queue, type, answer_class) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (subject, body, answer, " ".join(t.lower() for t in tags), case_id, source, queue, type_, answer_class),
    )
    conn.commit()


def mark_indexed(conn: sqlite3.Connection, case_ids: list[str]) -> None:
    """Stamps `indexed_at` on canonicals just upserted into Chroma."""
    now = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        "UPDATE dataset_tickets SET indexed_at = ? WHERE case_id = ?",
        [(now, c) for c in case_ids],
    )
    conn.commit()


def get_by_id(conn: sqlite3.Connection, case_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM dataset_tickets WHERE case_id = ?", (case_id,)).fetchone()


def answer_class_distribution(conn: sqlite3.Connection) -> dict[str, int]:
    return dict(conn.execute("SELECT answer_class, COUNT(*) FROM dataset_tickets GROUP BY answer_class").fetchall())


def answer_class_source_distribution(conn: sqlite3.Connection) -> dict[str, int]:
    return dict(
        conn.execute("SELECT answer_class_source, COUNT(*) FROM dataset_tickets GROUP BY answer_class_source").fetchall()
    )


def cluster_size_distribution(conn: sqlite3.Connection) -> dict[str, int]:
    """Bucketed distribution of `cluster_size` over canonical rows only."""
    sizes = [r[0] for r in conn.execute(
        "SELECT cluster_size FROM dataset_tickets WHERE is_canonical = 1 AND cluster_size IS NOT NULL"
    ).fetchall()]
    buckets = {"1": 0, "2-3": 0, "4-7": 0, "8-15": 0, "16+": 0}
    for size in sizes:
        key = "1" if size <= 1 else "2-3" if size <= 3 else "4-7" if size <= 7 else "8-15" if size <= 15 else "16+"
        buckets[key] += 1
    return buckets
