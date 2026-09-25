"""SQLite connection + schema for `data/autosupport.sqlite`.

Plain `sqlite3`, no ORM: five tables don't need one.
"""

from __future__ import annotations

import sqlite3

from autosupport.config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS dataset_tickets (
    case_id TEXT PRIMARY KEY,
    source TEXT NOT NULL DEFAULT 'dataset',
    hf_row INTEGER NOT NULL,
    subject TEXT NOT NULL,
    body TEXT NOT NULL,
    answer TEXT NOT NULL,
    subject_ix TEXT NOT NULL,
    body_ix TEXT NOT NULL,
    answer_ix TEXT NOT NULL,
    title TEXT NOT NULL,
    title_is_derived INTEGER NOT NULL,
    queue TEXT NOT NULL,
    type TEXT NOT NULL,
    priority TEXT NOT NULL,
    language TEXT NOT NULL,
    version INTEGER,
    tag_1 TEXT, tag_2 TEXT, tag_3 TEXT, tag_4 TEXT,
    tag_5 TEXT, tag_6 TEXT, tag_7 TEXT, tag_8 TEXT,
    answer_class TEXT NOT NULL,
    answer_class_source TEXT NOT NULL,
    below_content_threshold INTEGER NOT NULL DEFAULT 0,
    is_canonical INTEGER NOT NULL DEFAULT 0,
    canonical_of TEXT REFERENCES dataset_tickets(case_id),
    cluster_size INTEGER,
    indexed_at TEXT,
    ingested_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dataset_tickets_canonical_of ON dataset_tickets(canonical_of);
CREATE INDEX IF NOT EXISTS idx_dataset_tickets_is_canonical ON dataset_tickets(is_canonical);

CREATE VIRTUAL TABLE IF NOT EXISTS dataset_tickets_fts USING fts5(
    subject, body, answer, tags,
    case_id UNINDEXED, source UNINDEXED, queue UNINDEXED, type UNINDEXED, answer_class UNINDEXED,
    tokenize = 'unicode61 remove_diacritics 2'
);
CREATE VIRTUAL TABLE IF NOT EXISTS dataset_tickets_fts_vocab USING fts5vocab(dataset_tickets_fts, 'row');

CREATE TABLE IF NOT EXISTS cases (
    ticket_id       TEXT PRIMARY KEY,
    customer_id     TEXT NOT NULL,
    thread_id       TEXT NOT NULL UNIQUE,
    status          TEXT NOT NULL,
    subject         TEXT NOT NULL,
    body            TEXT NOT NULL,
    submitted_at    TEXT NOT NULL,
    classification  TEXT,
    pending_question TEXT,
    final_output    TEXT,
    indexed_at      TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cases_customer_updated ON cases(customer_id, updated_at);
CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status);

CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    profile     TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
"""


def connect(rebuild: bool = False) -> sqlite3.Connection:
    """Opens (creating if needed) `data/autosupport.sqlite` with the schema applied.
    `rebuild=True` drops `dataset_tickets` and its FTS5 index first — used by
    `autosupport ingest --rebuild`."""
    settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.sqlite_path)
    conn.row_factory = sqlite3.Row
    if rebuild:
        conn.executescript(
            "DROP TABLE IF EXISTS dataset_tickets_fts_vocab;"
            "DROP TABLE IF EXISTS dataset_tickets_fts;"
            "DROP TABLE IF EXISTS dataset_tickets;"
        )
    conn.executescript(_SCHEMA)
    return conn
