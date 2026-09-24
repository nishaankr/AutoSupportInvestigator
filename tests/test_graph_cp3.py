"""CP3 vertical-slice test: intake -> (load_memory | retrieve_initial) -> triage -> resolve
-> persist_case -> END, on `InMemorySaver`, with both LLM tiers stubbed. Verifies the graph
runs to completion, the `cases` row lands correctly, and `final_output` is a schema-valid
`CaseResult` with a real cited case ID (checkpoints.md's CP3 "Done" criteria, minus the CLI)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from autosupport.graph.build import build_graph
from autosupport.graph.nodes import resolve as resolve_node
from autosupport.graph.nodes import triage as triage_node
from autosupport.graph.state import InputState, TicketInput
from autosupport.rag.queries import SearchResult
from autosupport.store import db as store_db


class _FakeStructured:
    def __init__(self, output):
        self._output = output

    def invoke(self, _messages):
        return self._output


class _FakeLLM:
    """Dispatches by schema class name, since `resolve` makes two structured calls on the
    same `main_llm()` instance (one per schema — see resolve.py's module docstring)."""

    def __init__(self, outputs_by_schema_name: dict):
        self._outputs = outputs_by_schema_name

    def with_structured_output(self, schema, **_kwargs):
        return _FakeStructured(self._outputs[schema.__name__])


@pytest.fixture
def sqlite_env(tmp_path, monkeypatch):
    from autosupport.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    conn = store_db.connect()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO dataset_tickets (case_id, source, hf_row, subject, body, answer, subject_ix, body_ix, "
        "answer_ix, title, title_is_derived, queue, type, priority, language, version, answer_class, "
        "answer_class_source, below_content_threshold, is_canonical, cluster_size, ingested_at) VALUES "
        "('HF-1', 'dataset', 1, 'NAS shares unreachable', 'Shares vanished after firmware update.', "
        "'Re-enable SMB2 in the NAS control panel.', 'NAS shares unreachable', "
        "'Shares vanished after firmware update.', 'Re-enable SMB2 in the NAS control panel.', "
        "'NAS shares unreachable', 0, 'Technical Support', 'Incident', 'high', 'en', 400, 'resolution', "
        "'heuristic', 0, 1, 3, ?)",
        (now,),
    )
    conn.commit()
    conn.close()
    return tmp_path


def _stub_search(_text, k=10, where=None, conn=None):
    return [
        SearchResult(
            case_id="HF-1", similarity=0.9, score=0.5, dense_rank=1, lexical_rank=1,
            subject="NAS shares unreachable", queue="Technical Support", type="Incident",
            priority="high", answer_class="resolution", cluster_size=3,
            body_snippet="Shares vanished after firmware update.",
            answer_snippet="Re-enable SMB2 in the NAS control panel.",
        )
    ]


def test_vertical_slice_produces_valid_case_result(sqlite_env, monkeypatch):
    from autosupport.graph.state import Priority  # noqa: F401  (import surfaced for readability)

    monkeypatch.setattr("autosupport.graph.nodes.retrieve_initial.rag_queries.search", _stub_search)
    monkeypatch.setattr(
        triage_node,
        "fast_llm",
        lambda: _FakeLLM({
            "TriageOutput": triage_node.TriageOutput(
                queue="Technical Support", type="Incident", priority="high",
                tags=["nas", "smb"], rationale="Matches the retrieved NAS/SMB neighbours.",
            )
        }),
    )
    monkeypatch.setattr(
        resolve_node,
        "main_llm",
        lambda: _FakeLLM({
            "ResolveOutput": resolve_node.ResolveOutput(
                evidence=[
                    resolve_node.EvidenceItem(
                        case_id="HF-1", summary="NAS shares unreachable -> re-enable SMB2", stance="supports"
                    )
                ],
                analysis="The firmware update disables SMB2; one historical case [HF-1] matches closely.",
                resolution="Re-enable SMB2 in the NAS control panel [HF-1].",
            ),
        }),
    )

    graph = build_graph().compile(checkpointer=InMemorySaver())
    input_state: InputState = {
        "ticket_id": "T-20260924-abcdef",
        "customer_id": "C-1",
        "ticket": TicketInput(subject="NAS shares gone", body="After the firmware update our NAS shares vanished."),
    }
    output = graph.invoke(input_state, {"configurable": {"thread_id": "C-1:T-20260924-abcdef"}})

    assert output["ticket_id"] == "T-20260924-abcdef"
    assert output["status"] == "resolved"
    result = output["final_output"]
    assert result is not None
    assert result.status == "resolved"
    assert result.acceptance == "not_required"
    assert result.confidence is None
    assert result.escalation.required is False
    assert [e.case_id for e in result.evidence] == ["HF-1"]
    assert "[HF-1]" in result.resolution

    conn = store_db.connect()
    row = conn.execute("SELECT status, final_output FROM cases WHERE ticket_id = ?", ("T-20260924-abcdef",)).fetchone()
    conn.close()
    assert row["status"] == "resolved"
    assert row["final_output"] is not None
