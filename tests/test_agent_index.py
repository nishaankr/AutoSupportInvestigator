"""case-persistence.md §5: which agent cases grow the corpus, and that an indexed one comes
back out of retrieval as `source="agent_resolved"`. Embedding and Chroma are stubbed; the
SQLite/FTS5 side is real."""

from __future__ import annotations

import numpy as np
import pytest

from autosupport.graph.state import (
    CaseResult, Classification, Confidence, EscalationBlock, RunStats, TicketInput, VerificationOutcome,
)
from autosupport.ingest import agent_index
from autosupport.rag import lexical, queries
from autosupport.store import cases as cases_repo
from autosupport.store import dataset_tickets as dataset_repo
from autosupport.store import db as store_db

UPSERTS: list[tuple] = []
CLASSIFICATION = Classification(queue="Technical Support", type="Incident", priority="high",
                                tags=["Backup"], rationale="r", neighbor_agreement=1.0)


@pytest.fixture
def conn(tmp_path, monkeypatch):
    from autosupport.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(agent_index, "embed", lambda texts: np.zeros((len(texts), 4), dtype=np.float32))
    c = store_db.connect()
    for i in range(6):  # lexical excludes terms in >=20% of docs, so the index needs some bulk
        dataset_repo.add_fts_row(c, f"HF-{i}", "dataset", f"printer jam {i}", "paper stuck", "open tray",
                                 [], "Technical Support", "Incident", "resolution")
    UPSERTS[:] = []
    monkeypatch.setattr(agent_index.dense, "upsert", lambda *args: UPSERTS.append(args))
    yield c
    c.close()


def _case(conn, ticket_id: str, status: str = "resolved", acceptance: str = "accepted") -> None:
    cases_repo.insert_open(conn, ticket_id, "C-1", f"C-1:{ticket_id}",
                           TicketInput(subject="Hyper Backup to S3 fails", body="Synology DS920+ backup task errors"))
    cases_repo.set_classification(conn, ticket_id, CLASSIFICATION, "investigating")
    escalated = status == "escalated"
    cases_repo.save_final(conn, ticket_id, CaseResult(
        ticket_id=ticket_id, customer_id="C-1", status=status, classification=CLASSIFICATION, evidence=[],
        analysis="a", resolution="Re-link the S3 bucket credentials in Hyper Backup.",
        escalation=EscalationBlock(required=True, trigger="rule", target_queue="Technical Support",
                                   reason="r", handoff_summary="h") if escalated else EscalationBlock(required=False),
        confidence=Confidence(value=0.8, support=0.8, agreement=0.8, relevance=0.8, penalty=0.0),
        verification=VerificationOutcome(passed=True, attempts=1), acceptance=acceptance,
        stats=RunStats(retrieval_rounds=1, tool_calls=0, verify_attempts=1, revisions=0)))


def test_accepted_resolution_is_indexed_and_retrieved_as_agent_resolved(conn):
    _case(conn, "T-20260924-aaaaaa")
    assert agent_index.index_agent_case(conn, "T-20260924-aaaaaa")
    assert UPSERTS[0][3]["source"] == "agent_resolved" and "customer_id" not in UPSERTS[0][3]
    assert not agent_index.index_agent_case(conn, "T-20260924-aaaaaa")  # indexed_at guard: once only

    hits = lexical.search(conn, "hyper backup synology")
    assert [h.case_id for h in hits] == ["T-20260924-aaaaaa"]
    row = queries._fetch_rows(conn, ["T-20260924-aaaaaa"])["T-20260924-aaaaaa"]
    assert row["source"] == "agent_resolved" and row["answer_class"] == "resolution"
    assert "Re-link the S3 bucket" in row["answer_ix"]


@pytest.mark.parametrize("status,acceptance", [("escalated", "not_required"), ("resolved", "not_required"),
                                               ("resolved", "rejected")])
def test_unaccepted_outcomes_are_not_indexed(conn, status, acceptance):
    _case(conn, "T-20260924-bbbbbb", status=status, acceptance=acceptance)
    assert not agent_index.index_agent_case(conn, "T-20260924-bbbbbb")
    assert UPSERTS == [] and cases_repo.get(conn, "T-20260924-bbbbbb")["indexed_at"] is None


def test_dataset_fts_rebuild_keeps_agent_rows(conn):
    _case(conn, "T-20260924-cccccc")
    agent_index.index_agent_case(conn, "T-20260924-cccccc")
    dataset_repo.rebuild_fts(conn)
    ids = {r[0] for r in conn.execute("SELECT case_id FROM dataset_tickets_fts").fetchall()}
    assert ids == {"T-20260924-cccccc"}
