"""The five offline evaluators (evaluation-design.md §4). Each takes the target's `outputs`
(a compact final-state view, `evals/run.py`) and the example's gold `reference_outputs`,
and returns LangSmith feedback. Unrelated to the in-graph `verify` node: these never run
inside a ticket and never change its outcome.

LLM judges return per-item verdicts only; the score is computed here from those verdicts.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel

from autosupport.graph.state import CITATION
from autosupport.llm import judge_llm, structured
from autosupport.store import db as store_db
from autosupport.tools.get_ticket_by_id import get_ticket_by_id

TOP_K_JUDGED = 5
EXPECTED_OUTCOME = {"resolution": "resolved", "escalation": "escalated", "clarification_request": "asked_clarification"}


def _skip(key: str, why: str) -> dict:
    return {"key": key, "score": None, "comment": why}


def _run_failed(outputs: dict | None) -> bool:
    """A target that raised reaches evaluators as `None` or `{"output": None}` — not scored,
    or an API outage would read as bad retrieval or perfect tool use."""
    return not outputs or "outcome" not in outputs


# ---------- response_groundedness (LLM judge) ----------
class ClaimCheck(BaseModel):
    claim: str
    supported: bool
    why: str


class GroundednessJudgement(BaseModel):
    claims: list[ClaimCheck]


GROUNDEDNESS_PROMPT = """You are auditing a support reply for groundedness. List every factual or \
technical claim in the REPLY: steps to take, facts about a product, process or policy, and \
statements about what will fix the problem. Ignore greetings, empathy, promises to follow up and \
questions to the customer. For each claim decide `supported`: true only if the CITED HISTORICAL \
CASES below state it or directly imply it. General advice those cases don't contain is not \
supported, however plausible."""


def response_groundedness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    key = "response_groundedness"
    if _run_failed(outputs):
        return _skip(key, "run failed")
    response = outputs.get("response")
    if outputs["outcome"] != "resolved" or not response:
        # An escalation's holding reply is a code template with no factual claims (D19);
        # judging it only measures how the judge treats "a person will follow up".
        return _skip(key, f"no resolution to ground ({outputs['outcome']})")
    cited = list(dict.fromkeys(CITATION.findall(response["resolution"])))
    cases = "\n\n".join(json.dumps(get_ticket_by_id.func(cid), default=str) for cid in cited) or "(none cited)"
    judgement: GroundednessJudgement = structured(judge_llm(), GroundednessJudgement).invoke([
            ("system", GROUNDEDNESS_PROMPT),
            ("user", f"REPLY:\n{response['resolution']}\n\nCITED HISTORICAL CASES:\n{cases}"),
        ])
    if not judgement.claims:
        return {"key": key, "score": 1.0, "comment": "no factual claims (holding reply)"}
    unsupported = [c.claim for c in judgement.claims if not c.supported]
    score = 1 - len(unsupported) / len(judgement.claims)
    return {"key": key, "score": score,
            "comment": f"{len(judgement.claims) - len(unsupported)}/{len(judgement.claims)} supported; "
                       f"unsupported: {unsupported}"}


# ---------- retrieval_relevance (LLM judge) ----------
class CaseRelevance(BaseModel):
    case_id: str
    relevance: Literal["relevant", "partial", "irrelevant"]
    why: str


class RelevanceJudgement(BaseModel):
    cases: list[CaseRelevance]


RELEVANCE_PROMPT = """You are grading retrieval for a support agent. For each RETRIEVED CASE, \
decide `relevance` against the TICKET:
- relevant: the same underlying problem or request, so its historical answer would directly help.
- partial: the same kind of request but missing a specific that matters (a different product, \
platform or integration), so its answer helps only in part.
- irrelevant: shares only a generic topic word ("security", "integration"), or is a different problem.
The ticket's own historical answer is given as a reference for what a helpful answer looked like. \
Judge every case listed, by its case_id."""
RELEVANCE_POINTS = {"relevant": 1.0, "partial": 0.5, "irrelevant": 0.0}


def retrieval_relevance(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    key = "retrieval_relevance"
    if _run_failed(outputs):
        return _skip(key, "run failed")
    retrieved = ((outputs or {}).get("retrieved") or [])[:TOP_K_JUDGED]
    if not retrieved:
        return {"key": key, "score": 0.0, "comment": "nothing retrieved"}
    reference = (reference_outputs or {}).get("answer") or "(not available)"
    listing = "\n\n".join(
        f"[{c['case_id']}] subject: {c['subject']}\nproblem: {c['body_snippet']}\nanswer: {c['answer_snippet']}"
        for c in retrieved)
    judgement: RelevanceJudgement = structured(judge_llm(), RelevanceJudgement).invoke([
            ("system", RELEVANCE_PROMPT),
            ("user", f"TICKET:\n{inputs['subject']}\n\n{inputs['body']}\n\nHISTORICAL ANSWER (reference):\n"
                     f"{reference}\n\nRETRIEVED CASES:\n{listing}"),
        ])
    ids = {c["case_id"] for c in retrieved}
    graded = {c.case_id: c.relevance for c in judgement.cases if c.case_id in ids}
    score = sum(RELEVANCE_POINTS[g] for g in graded.values()) / len(retrieved)  # an unjudged case counts 0
    return {"key": key, "score": score,
            "comment": f"graded@{len(retrieved)}: {graded}; top similarity {retrieved[0]['similarity']:.3f}"}


# ---------- tool_usage_correctness (code) ----------
def _case_exists(case_id: str) -> bool:
    conn = store_db.connect()
    try:
        if case_id.startswith("HF-"):
            return conn.execute("SELECT 1 FROM dataset_tickets WHERE case_id = ?", (case_id,)).fetchone() is not None
        return conn.execute("SELECT 1 FROM cases WHERE ticket_id = ?", (case_id,)).fetchone() is not None
    finally:
        conn.close()


def tool_usage_correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    key = "tool_usage_correctness"
    if _run_failed(outputs):
        return _skip(key, "run failed")
    calls = outputs.get("tool_log") or []
    if not calls:
        return {"key": key, "score": 1.0, "comment": "no tool calls"}
    signatures = [(c["name"], json.dumps(c["args"], sort_keys=True)) for c in calls]
    checks = {
        "all_calls_succeeded": all(c["ok"] for c in calls),
        "no_repeated_call": len(set(signatures)) == len(signatures),
    }
    fetched = [c["args"].get("case_id", "") for c in calls if c["name"] == "get_ticket_by_id"]
    if fetched:
        checks["fetched_ids_exist"] = all(_case_exists(cid) for cid in fetched)
    failed = [name for name, ok in checks.items() if not ok]
    names = [c["name"] for c in calls]
    return {"key": key, "score": sum(checks.values()) / len(checks),
            "comment": f"{len(calls)} calls {sorted(set(names))}; failed: {failed or 'none'}"}


# ---------- classification_accuracy (code, dataset labels as ground truth) ----------
def classification_accuracy(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    key = "classification_accuracy"
    if not (reference_outputs or {}).get("queue"):
        return _skip(key, "no gold labels")
    predicted = None if _run_failed(outputs) else outputs.get("classification")
    if not predicted:
        return _skip(key, "no classification (run failed)")
    fields = ("queue", "type", "priority")
    matches = {f: float(predicted[f] == reference_outputs[f]) for f in fields}
    comment = ", ".join(f"{f}: {predicted[f]} vs {reference_outputs[f]}" for f in fields)
    return {"results": [
        {"key": key, "score": sum(matches.values()) / len(fields), "comment": comment},
        *({"key": f"classification_{f}", "score": matches[f]} for f in fields),
    ]}


# ---------- outcome_appropriateness (code, REQUIREMENTS §8 "overall behaviour") ----------
def outcome_appropriateness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    key = "outcome_appropriateness"
    expected = EXPECTED_OUTCOME.get((reference_outputs or {}).get("answer_class"))
    if expected is None:
        return _skip(key, "no gold answer_class")
    if _run_failed(outputs):
        return _skip(key, "run failed")
    actual = outputs["outcome"]
    return {"key": key, "score": float(actual == expected), "comment": f"expected {expected}, got {actual}"}


ALL = [response_groundedness, retrieval_relevance, tool_usage_correctness, classification_accuracy,
       outcome_appropriateness]
