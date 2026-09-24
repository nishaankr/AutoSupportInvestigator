"""`persist_case` — `stats.tool_calls` must reflect `tool_log`, not a hardcoded 0
(regression: found via the CP4 live verification run, where 3 tool calls were logged but
persisted as 0)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from autosupport.graph.nodes.persist_case import persist_case
from autosupport.graph.state import (
    Classification, Confidence, DraftResponse, ToolCallRecord, VerificationResult,
)


@pytest.fixture
def sqlite_env(tmp_path, monkeypatch):
    from autosupport.config import settings
    from autosupport.store import cases as cases_repo
    from autosupport.store import db as store_db

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    conn = store_db.connect()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO cases (ticket_id, customer_id, thread_id, status, subject, body, submitted_at, "
        "created_at, updated_at) VALUES ('T-1', 'C-1', 'C-1:T-1', 'investigating', 's', 'b', ?, ?, ?)",
        (now, now, now),
    )
    conn.commit()
    conn.close()
    return tmp_path


def test_stats_tool_calls_counts_tool_log(sqlite_env):
    state = {
        "ticket_id": "T-1", "customer_id": "C-1",
        "classification": Classification(
            queue="q", type="t", priority="high", tags=[], rationale="r", neighbor_agreement=1.0
        ),
        "draft": DraftResponse(analysis="a", resolution="r", escalation=None),
        "evidence": [],
        "decision": "resolve",
        "verification": VerificationResult(passed=True),
        "confidence": Confidence(value=0.5, support=0.5, agreement=1.0, relevance=0.5, penalty=0.0),
        "verify_attempts": 1,
        "retrieval_round": 1,
        "tool_log": [
            ToolCallRecord(name="search_similar_tickets", args={}, ok=True, duration_ms=1, round=1),
            ToolCallRecord(name="get_customer_history", args={}, ok=True, duration_ms=1, round=1),
        ],
    }

    result = persist_case(state)

    assert result["final_output"].stats.tool_calls == 2
