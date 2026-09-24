"""`graph/nodes/tools.py::tools_node` — executes model-requested tool calls, logs each one,
and merges `search_similar_tickets` hits into `retrieved_cases` (tools-and-skills.md §1)."""

from __future__ import annotations

import autosupport.tools.search_similar_tickets as search_tool_module
from langchain_core.messages import AIMessage

from autosupport.graph.nodes.tools import tools_node
from autosupport.graph.state import TicketInput
from autosupport.rag.queries import SearchResult


def _stub_search(_text, k=10, where=None, conn=None):
    return [
        SearchResult(
            case_id="HF-1", similarity=0.9, score=0.5, dense_rank=1, lexical_rank=1,
            subject="NAS shares unreachable", queue="Technical Support", type="Incident",
            priority="high", answer_class="resolution", cluster_size=3,
            body_snippet="problem text", answer_snippet="answer text",
        )
    ]


def test_tools_node_executes_search_and_merges_retrieved_cases(monkeypatch):
    monkeypatch.setattr(search_tool_module, "search", _stub_search)
    monkeypatch.setattr("autosupport.graph.nodes.tools.reanchor", lambda cases, _anchor: cases)
    ai_message = AIMessage(
        content="",
        tool_calls=[{"name": "search_similar_tickets", "args": {"query": "NAS SMB shares"}, "id": "call_1"}],
    )
    state = {
        "customer_id": "C-1", "ticket_id": "T-1", "messages": [ai_message],
        "ticket": TicketInput(subject="s", body="b"), "retrieval_round": 1, "tool_calls_this_round": 0,
    }

    result = tools_node(state)

    assert len(result["messages"]) == 1
    assert result["messages"][0].tool_call_id == "call_1"
    assert result["tool_log"][0].name == "search_similar_tickets"
    assert result["tool_log"][0].ok is True
    assert result["tool_calls_this_round"] == 1
    assert [c.case_id for c in result["retrieved_cases"]] == ["HF-1"]
    assert result["retrieved_cases"][0].query_label == "tool:search_similar_tickets"


def test_tools_node_records_failure_without_crashing():
    ai_message = AIMessage(content="", tool_calls=[{"name": "not_a_real_tool", "args": {}, "id": "call_2"}])
    state = {"customer_id": "C-1", "ticket_id": "T-1", "messages": [ai_message], "retrieval_round": 1}

    result = tools_node(state)

    assert result["tool_log"][0].ok is False
    assert "error" in result["messages"][0].content
    assert result["retrieved_cases"] == []


def test_similarities_to_dedupes_ids_before_hitting_chroma(monkeypatch):
    """Regression (found in the CP5 live run): two `search_similar_tickets` calls in one turn
    can return the same case, and Chroma's `get` raises DuplicateIDError on repeated IDs."""
    import numpy as np

    from autosupport.rag import dense

    class _Coll:
        def get(self, ids, include):
            assert len(ids) == len(set(ids)), "duplicate IDs reached Chroma"
            return {"ids": ids, "embeddings": [np.array([1.0, 0.0])] * len(ids)}

    monkeypatch.setattr(dense, "_collection", lambda: _Coll())
    sims = dense.similarities_to(["HF-1", "HF-2", "HF-1"], np.array([1.0, 0.0], dtype=np.float32))
    assert set(sims) == {"HF-1", "HF-2"}
