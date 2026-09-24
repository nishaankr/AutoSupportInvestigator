"""The code-only offline evaluators (evaluation-design.md §4) and the ingest holdout."""

from __future__ import annotations

import json

import pandas as pd

from autosupport.ingest import load
from evals import evaluators

REFERENCE = {"queue": "IT Support", "type": "Incident", "priority": "high", "answer_class": "escalation"}


def _call(name, args, ok=True):
    return {"name": name, "args": args, "ok": ok, "round": 1}


def test_tool_usage_flags_repeats_failures_and_invented_ids(monkeypatch):
    monkeypatch.setattr(evaluators, "_case_exists", lambda cid: cid == "HF-1")
    clean = evaluators.tool_usage_correctness({}, {"outcome": "resolved", "tool_log": [
        _call("search_similar_tickets", {"query": "vpn"}), _call("get_ticket_by_id", {"case_id": "HF-1"})]}, {})
    assert clean["score"] == 1.0

    bad = evaluators.tool_usage_correctness({}, {"outcome": "resolved", "tool_log": [
        _call("search_similar_tickets", {"query": "vpn"}), _call("search_similar_tickets", {"query": "vpn"}),
        _call("get_ticket_by_id", {"case_id": "HF-999"}, ok=False)]}, {})
    assert bad["score"] == 0.0 and "no_repeated_call" in bad["comment"]
    assert evaluators.tool_usage_correctness({}, {"outcome": "resolved", "tool_log": []}, {})["score"] == 1.0


def test_classification_accuracy_scores_each_label():
    out = evaluators.classification_accuracy(
        {}, {"outcome": "resolved", "classification": {"queue": "IT Support", "type": "Request", "priority": "high"}}, REFERENCE)
    scores = {r["key"]: r["score"] for r in out["results"]}
    assert scores["classification_accuracy"] == 2 / 3
    assert (scores["classification_queue"], scores["classification_type"]) == (1.0, 0.0)
    assert evaluators.classification_accuracy({}, {"outcome": "resolved", "classification": None}, {})["score"] is None


def test_outcome_appropriateness_maps_answer_class_to_expected_outcome():
    assert evaluators.outcome_appropriateness({}, {"outcome": "escalated"}, REFERENCE)["score"] == 1.0
    assert evaluators.outcome_appropriateness({}, {"outcome": "resolved"}, REFERENCE)["score"] == 0.0
    clarify = {**REFERENCE, "answer_class": "clarification_request"}
    assert evaluators.outcome_appropriateness({}, {"outcome": "asked_clarification"}, clarify)["score"] == 1.0


def test_ingest_excludes_the_eval_holdout(tmp_path, monkeypatch):
    holdout = tmp_path / "examples.jsonl"
    holdout.write_text(json.dumps({"example_id": "HF-2"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(load, "HOLDOUT_PATH", holdout)
    monkeypatch.setattr(load, "load_english_subset",
                        lambda limit=None, rebuild=False: pd.DataFrame({"case_id": ["HF-1", "HF-2", "HF-3"]}))
    assert load.load_ingest_subset()["case_id"].tolist() == ["HF-1", "HF-3"]


def test_failed_runs_are_skipped_not_scored():
    for failed in (None, {"output": None}):
        for evaluator in (evaluators.retrieval_relevance, evaluators.tool_usage_correctness,
                          evaluators.response_groundedness, evaluators.outcome_appropriateness,
                          evaluators.classification_accuracy):
            assert evaluator({"subject": "s", "body": "b"}, failed, REFERENCE)["score"] is None
