"""CP5 graph-path tests: the whole topology on a real file-backed `SqliteSaver` (with the same
serde allowlist production uses, so every checkpointed type must round-trip), LLMs and
retrieval stubbed. Covers checkpoints.md CP5 "Done" parts 3 and 4 deterministically:
the rejection loop and forced exhaustion of every loop counter. Part 1 (a real process exit
between `new` and `resume`) is verified live, not here — see the CP5 verification run."""

from __future__ import annotations

import itertools
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from autosupport.graph.build import _allowed_msgpack_modules, build_graph
from autosupport.graph.nodes import investigate as investigate_node
from autosupport.graph.nodes import refine_retrieval as refine_node
from autosupport.graph.nodes import resolve as resolve_node
from autosupport.graph.nodes import update_memory as memory_node
from autosupport.graph.memory import MemoryUpdate, RememberedFact
from autosupport.graph.nodes import verify as verify_node
from autosupport.graph.state import ApproachCluster, EvidenceItem, Findings, TicketInput
from autosupport.ingest import agent_index
from autosupport.rag.queries import SearchResult
from autosupport.store import db as store_db

CASE_IDS = ["HF-1", "HF-2", "HF-3"]


class _Fake:
    """One fake for every LLM tier: structured outputs dispatch by schema name (a value, or a
    list consumed in order with the last repeating). `bind_tools(...).invoke(...)` returns an
    AIMessage requesting `tool_calls_per_turn` real `compute_queue_stats` calls, or — when that
    is 0 or `submit_findings` is forced — a `submit_findings` call carrying `findings`."""

    _ids = itertools.count()

    def __init__(self, outputs: dict, tool_calls_per_turn: int = 0, findings=None):
        self._outputs, self._n, self._findings = outputs, tool_calls_per_turn, findings

    def with_structured_output(self, schema, **_):
        value = self._outputs[schema.__name__]
        if isinstance(value, list):
            value = value.pop(0) if len(value) > 1 else value[0]
        return type("S", (), {"invoke": lambda _self, _m: value})()

    def bind_tools(self, _tools, tool_choice=None):
        submit = tool_choice == "submit_findings" or self._n == 0
        calls = [{"name": "submit_findings", "args": self._findings.model_dump(), "id": f"call_{next(_Fake._ids)}"}] \
            if submit else [{"name": "compute_queue_stats", "args": {}, "id": f"call_{next(_Fake._ids)}"}
                            for _ in range(self._n)]
        return type("B", (), {"invoke": lambda _self, _m, **_k: AIMessage(content="", tool_calls=calls)})()


def _hits(_text, k=10, where=None, conn=None, phrases=()):
    return [SearchResult(
        case_id=cid, similarity=0.9, score=0.5, dense_rank=1, lexical_rank=1, subject=f"s {cid}",
        queue="Technical Support", type="Incident", priority="high", answer_class="resolution",
        cluster_size=3, body_snippet="problem", answer_snippet="Re-enable SMB2.",
    ) for cid in CASE_IDS]


@pytest.fixture
def env(tmp_path, monkeypatch):
    from autosupport.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    conn = store_db.connect()
    now = datetime.now(timezone.utc).isoformat()
    for i, cid in enumerate(CASE_IDS, start=1):
        conn.execute(
            "INSERT INTO dataset_tickets (case_id, source, hf_row, subject, body, answer, subject_ix, body_ix, "
            "answer_ix, title, title_is_derived, queue, type, priority, language, answer_class, "
            "answer_class_source, below_content_threshold, is_canonical, cluster_size, ingested_at) VALUES "
            f"('{cid}', 'dataset', {i}, 's', 'b', 'a', 's', 'b', 'a', 's', 0, 'Technical Support', 'Incident', "
            "'high', 'en', 'resolution', 'heuristic', 0, 1, 3, ?)", (now,))
    conn.commit()
    conn.close()
    monkeypatch.setattr("autosupport.graph.nodes.retrieve_initial.rag_queries.search", _hits)
    monkeypatch.setattr(refine_node.rag_queries, "search", _hits)
    monkeypatch.setattr("autosupport.graph.nodes.tools.reanchor", lambda cases, _a: cases)
    monkeypatch.setattr(refine_node, "reanchor", lambda cases, _a: cases)
    monkeypatch.setattr(agent_index, "embed", lambda texts: np.zeros((len(texts), 4), dtype=np.float32))
    monkeypatch.setattr(agent_index.dense, "upsert", lambda *args: None)
    return tmp_path


def _wire(monkeypatch, *, sufficient: bool, verify_fails: bool, tools_per_turn: int = 0):
    # triage, assess_evidence, refine_retrieval and escalate make no model calls (D19).
    findings = Findings(
        hypothesis="SMB2 disabled", root_cause_category="config", supporting_case_ids=CASE_IDS,
        evidence=[EvidenceItem(case_id=c, summary="re-enable SMB2", stance="supports") for c in CASE_IDS]
        if sufficient else [],
        clusters=[ApproachCluster(label="reenable smb2", case_ids=CASE_IDS)] if sufficient else [],
        missing_slots=[] if sufficient else ["firmware version"], gap_is_retrievable=not sufficient,
        clarification_question=None if sufficient else "Which firmware version are you on?", reason="r")
    draft = resolve_node.DraftOutput(analysis="Matches [HF-1][HF-2][HF-3].", resolution="Re-enable SMB2 [HF-1].")
    verdict = verify_node.VerificationJudgement(
        unsupported_claims=["invented step"] if verify_fails else [], recommended_action="re_reason")
    monkeypatch.setattr(investigate_node, "main_llm", lambda: _Fake({}, tools_per_turn, findings))
    monkeypatch.setattr(resolve_node, "fast_llm", lambda: _Fake({"DraftOutput": draft}))
    monkeypatch.setattr(verify_node, "fast_llm", lambda: _Fake({"VerificationJudgement": verdict}))
    memory = MemoryUpdate(facts=[RememberedFact(key="product", value="NAS", quote="NAS shares gone")], reasoning="r")
    monkeypatch.setattr(memory_node, "fast_llm", lambda: _Fake({"MemoryUpdate": memory}))


def _graph(tmp_path):
    saver = SqliteSaver(sqlite3.connect(tmp_path / "ckpt.sqlite", check_same_thread=False),
                        serde=JsonPlusSerializer(allowed_msgpack_modules=_allowed_msgpack_modules()))
    return build_graph().compile(checkpointer=saver)


def _config(tid, **limits):
    return {"configurable": {"thread_id": tid, "require_acceptance": True, "tau_rel": 0.76, **limits},
            "recursion_limit": 100}


def _start(graph, cfg, tid="T-20260924-abcdef"):
    graph.invoke({"ticket_id": tid, "customer_id": "C-1",
                  "ticket": TicketInput(subject="NAS shares gone", body="shares vanished after update, already restarted the NAS")}, cfg)


def test_happy_path_pauses_at_confirmation_then_accepts(env, monkeypatch):
    _wire(monkeypatch, sufficient=True, verify_fails=False)
    graph, cfg = _graph(env), _config("C-1:T-1")
    _start(graph, cfg)

    snap = graph.get_state(cfg)
    assert snap.next == ("confirm_resolution",)
    assert snap.interrupts[0].value["type"] == "confirmation"
    assert snap.interrupts[0].value["cited_case_ids"] == ["HF-1"]
    conn = store_db.connect()
    assert conn.execute("SELECT status FROM cases").fetchone()["status"] == "awaiting_user"
    conn.close()

    out = graph.invoke(Command(resume={"accepted": True}), cfg)
    result = out["final_output"]
    assert (result.status, result.acceptance, result.verification.passed) == ("resolved", "accepted", True)
    assert result.confidence.band in ("medium", "high")
    # CP6: the accepted resolution is indexed and the customer's memory written.
    conn = store_db.connect()
    assert conn.execute("SELECT indexed_at FROM cases").fetchone()["indexed_at"] is not None
    assert conn.execute("SELECT source FROM dataset_tickets_fts WHERE case_id LIKE 'T-%'").fetchone()[0] == "agent_resolved"
    assert '"product":"NAS"' in conn.execute("SELECT profile FROM customers").fetchone()[0]
    conn.close()


def test_rejection_loops_to_a_new_draft_then_exhausts_to_escalation(env, monkeypatch):
    _wire(monkeypatch, sufficient=True, verify_fails=False)
    graph, cfg = _graph(env), _config("C-1:T-2", max_revisions=1)
    _start(graph, cfg, "T-20260924-000002")

    graph.invoke(Command(resume={"accepted": False, "feedback": "that did not work"}), cfg)
    snap = graph.get_state(cfg)  # revised draft, paused at confirmation again
    assert snap.next == ("confirm_resolution",) and snap.values["revision_count"] == 1
    assert "that did not work" in snap.values["messages"][-1].content or snap.values["user_feedback"]

    out = graph.invoke(Command(resume={"accepted": False, "feedback": "still wrong"}), cfg)
    result = out["final_output"]  # max_revisions=1 spent -> escalate
    assert (result.status, result.escalation.trigger, result.acceptance) == ("escalated", "user_rejected", "rejected")
    conn = store_db.connect()  # CP6: a rejected, escalated case never grows the corpus
    assert conn.execute("SELECT indexed_at FROM cases").fetchone()["indexed_at"] is None
    conn.close()
    assert result.stats.revisions == 2


def test_every_loop_exhausts_to_escalation_with_counters_at_limits(env, monkeypatch):
    """Insufficient evidence + a fixable-looking gap forever, tools every turn: retrieval (B),
    clarification (C) and tool (A) loops all run to their limits and the run still terminates
    in `escalate`. The escalation draft is a template (D19), so `verify` checks it with code
    rules only and it passes first time; loop D's exhaustion is covered by the resolve-path
    test below."""
    _wire(monkeypatch, sufficient=False, verify_fails=True, tools_per_turn=3)
    graph, cfg = _graph(env), _config("C-1:T-3", max_clarifications=2)
    _start(graph, cfg, "T-20260924-000003")

    questions = 0
    while graph.get_state(cfg).next == ("ask_user",):
        questions += 1
        assert graph.get_state(cfg).interrupts[0].value["question"] == "Which firmware version are you on?"
        graph.invoke(Command(resume={"answer": "v1.2"}), cfg)
    assert questions == 2

    v = graph.get_state(cfg).values
    result = v["final_output"]
    assert result.status == "escalated" and v["decision"] == "escalate"
    assert v["retrieval_round"] == 3 and v["clarification_count"] == 2 and v["verify_attempts"] == 1
    assert result.escalation.trigger == "evidence_exhausted" and result.verification.passed is True
    assert result.confidence.band == "low"
    assert result.stats.tool_calls > 0
    # The templated handoff carries what a human needs, citing nothing that wasn't evidence.
    assert "firmware version" in result.escalation.handoff_summary and "[HF-" not in result.resolution


def test_worst_case_fits_one_invocation_under_the_recursion_limit(env, monkeypatch):
    """With no clarification budget the whole worst case runs in ONE invoke — the case the
    recursion_limit=100 derivation (graph-design.md §6) is about. No GraphRecursionError."""
    # One tool call per turn is the true worst case: the model uses all 6 calls of every round
    # one investigate <-> tools pass at a time.
    _wire(monkeypatch, sufficient=False, verify_fails=True, tools_per_turn=1)
    graph, cfg = _graph(env), _config("C-1:T-4", max_clarifications=0)
    steps = sum(1 for _ in graph.stream(
        {"ticket_id": "T-20260924-000004", "customer_id": "C-1",
         "ticket": TicketInput(subject="s", body="b")}, cfg, stream_mode="updates"))
    assert graph.get_state(cfg).values["final_output"].status == "escalated"
    print(f"WORST_CASE_STEPS={steps}")
    assert steps < 100


def test_verify_exhaustion_on_a_resolve_draft_escalates_with_verification_failed(env, monkeypatch):
    _wire(monkeypatch, sufficient=True, verify_fails=True)
    graph, cfg = _graph(env), _config("C-1:T-5")
    _start(graph, cfg, "T-20260924-000005")
    result = graph.get_state(cfg).values["final_output"]
    assert result.status == "escalated" and result.escalation.trigger == "verification_failed"
    assert result.verification.attempts == 3  # two failed resolve drafts + the escalation's own check


def test_a_model_that_never_submits_findings_escalates_instead_of_crashing(env, monkeypatch):
    """D20: GPT-OSS sometimes writes findings as text or calls a tool that wasn't offered, and
    Groq rejects it (`tool_use_failed`). After the forced attempts the round degrades to
    "no findings" and the ticket goes to a person."""
    _wire(monkeypatch, sufficient=True, verify_fails=False)

    class _Failing:
        def bind_tools(self, _tools, **_):
            def fail(_self, _m, **_k):
                raise RuntimeError("Error code: 400 - {'code': 'tool_use_failed'}")
            return type("B", (), {"invoke": fail})()
    monkeypatch.setattr(investigate_node, "main_llm", lambda: _Failing())
    graph, cfg = _graph(env), _config("C-1:T-6")
    # A detailed ticket, so the thin-ticket rule (ask once before escalating) doesn't apply.
    body = ("After last night's firmware update our NAS no longer shows any SMB shares to Windows clients. "
            "We restarted the NAS and the Windows machines, checked the network, and the web UI still lists every share.")
    graph.invoke({"ticket_id": "T-20260924-000006", "customer_id": "C-1",
                  "ticket": TicketInput(subject="NAS shares gone", body=body)}, cfg)
    result = graph.get_state(cfg).values["final_output"]
    assert result.status == "escalated" and result.escalation.trigger == "evidence_exhausted"
    assert any("no valid submit_findings" in e for e in result.errors)
