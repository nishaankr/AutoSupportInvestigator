"""The offline evaluators. Unrelated to the in-graph `verify` node:
these never run inside a ticket and never change its outcome.

Everything is plain code except `response_groundedness`, the one LLM-as-judge, which is also
the only evaluator allowed to create a LangSmith trace. The run disables evaluator tracing and
the judge switches it back on for its own call, so each example costs one pipeline trace plus
at most one judge trace.
"""

from __future__ import annotations

import json

from pydantic import BaseModel

from autosupport.graph.state import CITATION
from autosupport.llm import judge_llm, structured
from autosupport.store import db as store_db
from autosupport.tools.get_ticket_by_id import get_ticket_by_id

RETRIEVED_TOP = 10


def _skip(key: str, why: str) -> dict:
    return {"key": key, "score": None, "comment": why}


def _run_failed(outputs: dict | None) -> bool:
    """A target that raised arrives as `None` or `{"output": None}` — skipped, not scored, or
    an API outage would read as bad retrieval or perfect tool use."""
    return not outputs or "outcome" not in outputs


# ---------- classification_accuracy (code) ----------
def classification_accuracy(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """`triage`'s queue/type/priority (exact match) and tags (overlap) against the dataset's
    own labels for the ticket."""
    key = "classification_accuracy"
    if _run_failed(outputs):
        return _skip(key, "run failed")
    predicted, gold = outputs["classification"], reference_outputs
    matches = {f: float(predicted[f] == gold[f]) for f in ("queue", "type", "priority")}
    p_tags, g_tags = {t.lower() for t in predicted["tags"]}, {t.lower() for t in gold["tags"]}
    matches["tags"] = len(p_tags & g_tags) / len(p_tags | g_tags) if p_tags | g_tags else 1.0  # Jaccard
    comment = "; ".join(f"{f}: {predicted[f]} vs {gold[f]}" for f in ("queue", "type", "priority", "tags"))
    return {"results": [
        {"key": key, "score": sum(matches.values()) / len(matches), "comment": comment},
        *({"key": f"classification_{f}", "score": s} for f, s in matches.items()),
    ]}


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
    """Every tool call checked against the example's expected tool set: only allowed tools
    (escalating a ticket that should resolve or ask is wrong), every call succeeded, no identical
    call repeated, and no invented case id passed to `get_ticket_by_id`. No calls at all scores 1."""
    key = "tool_usage_correctness"
    if _run_failed(outputs):
        return _skip(key, "run failed")
    calls = outputs["tool_log"]
    if not calls:
        return {"key": key, "score": 1.0, "comment": "no tool calls"}
    allowed = set(reference_outputs["allowed_tools"])
    signatures = [(c["name"], json.dumps(c["args"], sort_keys=True)) for c in calls]
    checks = {
        "only_allowed_tools": all(c["name"] in allowed for c in calls),
        "all_calls_succeeded": all(c["ok"] for c in calls),
        "no_repeated_call": len(set(signatures)) == len(signatures),
    }
    fetched = [c["args"].get("case_id", "") for c in calls if c["name"] == "get_ticket_by_id"]
    if fetched:
        checks["fetched_ids_exist"] = all(_case_exists(cid) for cid in fetched)
    failed = [name for name, ok in checks.items() if not ok]
    return {"key": key, "score": sum(checks.values()) / len(checks),
            "comment": f"{len(calls)} calls {sorted({c['name'] for c in calls})}; failed: {failed or 'none'}"}


# ---------- retrieval_relevance (code) ----------
def retrieval_relevance(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Share of the example's known-good historical cases found among the top retrieved
    cases. `$FIRST_TICKET` stands for the memory example's first ticket, known only at run time."""
    key = "retrieval_relevance"
    if _run_failed(outputs):
        return _skip(key, "run failed")
    known = [outputs.get("first_ticket_id") if k == "$FIRST_TICKET" else k for k in reference_outputs["known_good"]]
    retrieved = {c["case_id"]: c for c in outputs["retrieved"][:RETRIEVED_TOP]}
    found = [k for k in known if k in retrieved]
    detail = [f"{k} ({retrieved[k]['source']}, {retrieved[k]['similarity']:.3f})" for k in found]
    return {"key": key, "score": len(found) / len(known),
            "comment": f"found {detail or 'none'} of {known} in top {RETRIEVED_TOP}"}


# ---------- response_groundedness (LLM judge — the only evaluator that traces) ----------
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
    """Each claim of a resolution checked against the full text of the cases it cites.
    Resolutions only: an escalation's holding reply is a code template with no claims."""
    key = "response_groundedness"
    if _run_failed(outputs):
        return _skip(key, "run failed")
    response = outputs.get("response")
    if outputs["outcome"] != "resolved" or not response:
        return _skip(key, f"no resolution to ground ({outputs['outcome']})")
    from langsmith import tracing_context

    cited = list(dict.fromkeys(CITATION.findall(response["resolution"])))
    cases = "\n\n".join(json.dumps(get_ticket_by_id.func(cid), default=str) for cid in cited) or "(none cited)"
    with tracing_context(enabled=True):  # the one evaluator trace per example
        judgement: GroundednessJudgement = structured(judge_llm(), GroundednessJudgement).invoke([
            ("system", GROUNDEDNESS_PROMPT),
            ("user", f"REPLY:\n{response['resolution']}\n\nCITED HISTORICAL CASES:\n{cases}"),
        ])
    if not judgement.claims:
        return {"key": key, "score": 1.0, "comment": "no factual claims"}
    unsupported = [c.claim for c in judgement.claims if not c.supported]
    return {"key": key, "score": 1 - len(unsupported) / len(judgement.claims),
            "comment": f"{len(judgement.claims) - len(unsupported)}/{len(judgement.claims)} supported; "
                       f"unsupported: {unsupported}"}


# ---------- pattern_behaviour (code) ----------
def pattern_behaviour(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Did the agent do what the example's pattern calls for (overall
    behaviour)? Each expectation in `reference["expect"]` is checked; the score is the share met."""
    key = "pattern_behaviour"
    if _run_failed(outputs):
        return _skip(key, "run failed")
    observed = {
        "outcome": outputs["outcome"],
        "verdict": "conflicting" if "conflicting" in outputs["verdicts"] else (outputs["verdicts"] or [None])[0],
        "memory_loaded": outputs.get("memory_loaded"),
        "first_ticket_retrieved": outputs.get("first_ticket_retrieved"),
    }
    expect = reference_outputs["expect"]
    met = {name: observed[name] == wanted for name, wanted in expect.items()}
    return {"key": key, "score": sum(met.values()) / len(met),
            "comment": "; ".join(f"{n}: expected {expect[n]}, got {observed[n]}" for n in expect)}


ALL = [classification_accuracy, tool_usage_correctness, retrieval_relevance, response_groundedness, pattern_behaviour]
