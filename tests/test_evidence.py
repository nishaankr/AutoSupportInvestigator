"""`graph/evidence.enrich` — dropping unknown IDs, then ordering and capping (output-schema.md
§3.2-§3.3)."""

from __future__ import annotations

import sqlite3

import pytest

from autosupport.graph.evidence import enrich
from autosupport.graph.state import EvidenceItem, RetrievedCase
from autosupport.store import db as store_db


@pytest.fixture
def conn(monkeypatch, tmp_path):
    from autosupport.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    c = store_db.connect()
    c.execute(
        "INSERT INTO dataset_tickets (case_id, source, hf_row, subject, body, answer, subject_ix, body_ix, "
        "answer_ix, title, title_is_derived, queue, type, priority, language, answer_class, "
        "answer_class_source, below_content_threshold, is_canonical, cluster_size, ingested_at) VALUES "
        "('HF-1', 'dataset', 1, 's1', 'b1', 'a1', 's1', 'b1', 'a1', 's1', 0, 'q', 't', 'high', 'en', "
        "'resolution', 'heuristic', 0, 1, 5, 'now')"
    )
    c.execute(
        "INSERT INTO cases (ticket_id, customer_id, thread_id, status, subject, body, submitted_at, "
        "created_at, updated_at) VALUES ('T-20260101-aaaaaa', 'C-1', 'C-1:T-20260101-aaaaaa', 'open', "
        "'sub', 'body', 'now', 'now', 'now')"
    )
    c.commit()
    yield c
    c.close()


def test_enrich_drops_unknown_case_ids(conn):
    items = [EvidenceItem(case_id="HF-999", summary="s", stance="supports")]
    entries, errors = enrich(items, [], conn)
    assert entries == []
    assert len(errors) == 1
    assert "HF-999" in errors[0]


def test_enrich_fills_dataset_fields_and_similarity_from_retrieved(conn):
    retrieved = [RetrievedCase(
        case_id="HF-1", source="dataset", subject="s1", body_snippet="b", answer_snippet="a",
        score=0.5, similarity=0.9, cluster_size=5, retrieval_round=1, query_label="initial",
    )]
    items = [EvidenceItem(case_id="HF-1", summary="s", stance="supports")]
    entries, errors = enrich(items, retrieved, conn)
    assert errors == []
    assert entries[0].source == "dataset"
    assert entries[0].cluster_size == 5
    assert entries[0].similarity == 0.9
    assert entries[0].answer_class == "resolution"


def test_enrich_customer_history_case_has_no_answer_class_when_open(conn):
    items = [EvidenceItem(case_id="T-20260101-aaaaaa", summary="s", stance="neutral")]
    entries, _ = enrich(items, [], conn)
    assert entries[0].source == "customer_history"
    assert entries[0].answer_class is None
    assert entries[0].cluster_size == 1


def test_enrich_orders_by_stance_then_cluster_size_then_similarity(conn):
    for i, (case_id, cluster_size) in enumerate([("HF-2", 3), ("HF-3", 8)], start=2):
        conn.execute(
            "INSERT INTO dataset_tickets (case_id, source, hf_row, subject, body, answer, subject_ix, body_ix, "
            "answer_ix, title, title_is_derived, queue, type, priority, language, answer_class, "
            "answer_class_source, below_content_threshold, is_canonical, cluster_size, ingested_at) VALUES "
            f"('{case_id}', 'dataset', {i}, 's', 'b', 'a', 's', 'b', 'a', 's', 0, 'q', 't', 'high', 'en', "
            f"'resolution', 'heuristic', 0, 1, {cluster_size}, 'now')"
        )
    conn.commit()
    items = [
        EvidenceItem(case_id="HF-1", summary="s", stance="contradicts"),
        EvidenceItem(case_id="HF-2", summary="s", stance="supports"),
        EvidenceItem(case_id="HF-3", summary="s", stance="supports"),
    ]
    entries, _ = enrich(items, [], conn)
    # both "supports" (HF-3 cluster_size=8, HF-2 cluster_size=3) sort before the one "contradicts" (HF-1)
    assert [e.case_id for e in entries] == ["HF-3", "HF-2", "HF-1"]
