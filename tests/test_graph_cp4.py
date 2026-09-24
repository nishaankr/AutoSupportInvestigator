"""CP4 vertical-slice test: intake -> (load_memory | retrieve_initial) -> triage ->
investigate <-> tools -> resolve -> persist_case -> END. This run makes the model stop
after zero tool calls (an "instant conclusion") to keep the fixture light — the tool-calling
loop itself is exercised live against the real corpus (checkpoints.md CP4 "Done": a run's
`tool_log` shows >=2 different tools called unprompted), and unit-tested directly in
test_tools_node.py."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from autosupport.graph.build import build_graph
from autosupport.graph.nodes import investigate as investigate_node
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
    """Dispatches by schema class name; also stands in for the ReAct decision call via
    `bind_tools(...).invoke(...)`, always returning a tool-call-free `AIMessage` here so the
    loop exits after one round without touching `tools_node`."""

    def __init__(self, outputs_by_schema_name: dict):
        self._outputs = outputs_by_schema_name

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        return AIMessage(content="No further tool calls needed.")

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
    monkeypatch.setattr("autosupport.graph.nodes.retrieve_initial.rag_queries.search", _stub_search)
    monkeypatch.setattr(
        triage_node,
        "fast_llm",
        lambda: _FakeLLM({
            "TriageOutput": triage_node.TriageOutput(
                queue="Technical Support", type="Incident", priority="high",
                tags=["nas", "smb"], rationale="Matches the retrieved NAS/SMB neighbours.",
                active_skills=[],
            )
        }),
    )
    monkeypatch.setattr(
        investigate_node,
        "main_llm",
        lambda: _FakeLLM({
            "InvestigateConclusion": investigate_node.InvestigateConclusion(
                hypothesis="Firmware update disabled SMB2.", root_cause_category="config",
                supporting_case_ids=["HF-1"], contradicting_case_ids=[],
                evidence=[
                    investigate_node.EvidenceItem(
                        case_id="HF-1", summary="NAS shares unreachable -> re-enable SMB2", stance="supports"
                    )
                ],
            )
        }),
    )
    monkeypatch.setattr(
        resolve_node,
        "main_llm",
        lambda: _FakeLLM({
            "DraftOutput": resolve_node.DraftOutput(
                analysis="The firmware update disables SMB2; one historical case [HF-1] matches closely.",
                resolution="Re-enable SMB2 in the NAS control panel [HF-1].",
            )
        }),
    )

    graph = build_graph().compile(checkpointer=InMemorySaver())
    input_state: InputState = {
        "ticket_id": "T-20260924-abcdef",
        "customer_id": "C-1",
        "ticket": TicketInput(subject="NAS shares gone", body="After the firmware update our NAS shares vanished."),
    }
    output = graph.invoke(input_state, {"configurable": {"thread_id": "C-1:T-20260924-abcdef"}})

    assert output["status"] == "resolved"
    result = output["final_output"]
    assert [e.case_id for e in result.evidence] == ["HF-1"]
    assert "[HF-1]" in result.resolution

    conn = store_db.connect()
    row = conn.execute("SELECT status FROM cases WHERE ticket_id = ?", ("T-20260924-abcdef",)).fetchone()
    conn.close()
    assert row["status"] == "resolved"
