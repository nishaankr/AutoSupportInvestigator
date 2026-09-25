"""The code-only offline evaluators (evaluation-design.md §4) and the ingest holdout."""

from __future__ import annotations

import json

import pandas as pd

from autosupport.ingest import load
from evals import evaluators

REFERENCE = {"queue": "IT Support", "type": "Incident", "priority": "high", "tags": ["Outage", "Bug"],
             "known_good": ["HF-1", "HF-2"], "allowed_tools": ["search_similar_tickets", "get_ticket_by_id"],
             "expect": {"outcome": "escalated"}}


def _out(**kw):
    base = {"outcome": "escalated", "classification": None, "tool_log": [], "retrieved": [], "verdicts": []}
    return {**base, **kw}


def _call(name, args, ok=True):
    return {"name": name, "args": args, "ok": ok, "round": 1}


def test_classification_scores_labels_and_tag_overlap():
    out = evaluators.classification_accuracy({}, _out(classification={
        "queue": "IT Support", "type": "Request", "priority": "high", "tags": ["outage", "Network"]}), REFERENCE)
    scores = {r["key"]: r["score"] for r in out["results"]}
    assert (scores["classification_queue"], scores["classification_type"], scores["classification_priority"]) == (1, 0, 1)
    assert scores["classification_tags"] == 1 / 3  # {outage} of {outage, bug, network}
    assert scores["classification_accuracy"] == (1 + 0 + 1 + 1 / 3) / 4


def test_tool_usage_checks_the_expected_tool_set(monkeypatch):
    monkeypatch.setattr(evaluators, "_case_exists", lambda cid: cid == "HF-1")
    clean = evaluators.tool_usage_correctness({}, _out(tool_log=[
        _call("search_similar_tickets", {"query": "vpn"}), _call("get_ticket_by_id", {"case_id": "HF-1"})]), REFERENCE)
    assert clean["score"] == 1.0
    wrong = evaluators.tool_usage_correctness({}, _out(tool_log=[
        _call("escalate_ticket", {"reason": "x"}), _call("get_ticket_by_id", {"case_id": "HF-9"})]), REFERENCE)
    assert "only_allowed_tools" in wrong["comment"] and "fetched_ids_exist" in wrong["comment"]
    assert evaluators.tool_usage_correctness({}, _out(), REFERENCE)["score"] == 1.0


def test_retrieval_relevance_counts_known_good_cases_including_the_first_ticket():
    retrieved = [{"case_id": "HF-1", "source": "dataset", "similarity": 0.9},
                 {"case_id": "T-20260925-aaaaaa", "source": "agent_resolved", "similarity": 0.86}]
    half = evaluators.retrieval_relevance({}, _out(retrieved=retrieved), REFERENCE)
    assert half["score"] == 0.5
    memory_ref = {**REFERENCE, "known_good": ["$FIRST_TICKET", "HF-1"]}
    full = evaluators.retrieval_relevance({}, _out(retrieved=retrieved, first_ticket_id="T-20260925-aaaaaa"), memory_ref)
    assert full["score"] == 1.0


def test_pattern_behaviour_checks_each_expectation():
    assert evaluators.pattern_behaviour({}, _out(), REFERENCE)["score"] == 1.0
    conflict = {**REFERENCE, "expect": {"verdict": "conflicting"}}
    assert evaluators.pattern_behaviour({}, _out(verdicts=["insufficient", "conflicting"]), conflict)["score"] == 1.0
    memory = {**REFERENCE, "expect": {"memory_loaded": True, "first_ticket_retrieved": True}}
    assert evaluators.pattern_behaviour({}, _out(memory_loaded=True, first_ticket_retrieved=False), memory)["score"] == 0.5


def test_groundedness_skips_anything_but_a_resolution():
    assert evaluators.response_groundedness({}, _out(outcome="escalated"), REFERENCE)["score"] is None


def test_failed_runs_are_skipped_not_scored():
    for failed in (None, {"output": None}):
        for evaluator in evaluators.ALL:
            assert evaluator({"subject": "s", "body": "b"}, failed, REFERENCE)["score"] is None


def test_ingest_excludes_the_eval_holdout(tmp_path, monkeypatch):
    holdout = tmp_path / "examples.jsonl"
    holdout.write_text(json.dumps({"example_id": "memory_and_new_case", "holdout_ids": ["HF-2", "HF-3"]}) + "\n",
                       encoding="utf-8")
    monkeypatch.setattr(load, "HOLDOUT_PATH", holdout)
    monkeypatch.setattr(load, "load_english_subset",
                        lambda limit=None, rebuild=False: pd.DataFrame({"case_id": ["HF-1", "HF-2", "HF-3"]}))
    assert load.load_ingest_subset()["case_id"].tolist() == ["HF-1"]
