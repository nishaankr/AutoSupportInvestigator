"""Node 13: `verify` — the in-graph evidence check on every draft, before anything is final.

Not to be confused with the offline LangSmith evaluation: this gates one ticket's answer while
it runs; the evaluation scores the whole system afterwards.

1. Code rules G1–G3 (`graph/verification.py`) on every draft: no citing a case that isn't in
   the evidence, a resolution must cite a supporting resolution-class case, and every
   contradicting case must be acknowledged.
2. Confidence is computed without the pass/fail cap, to get the band the wording must respect.
3. For drafts a model wrote (`resolve`), a separate model call looks for claims the cited cases
   don't back and for wording that promises more than the band allows ("this will fix it" at
   `low`). The templated escalation has no model-written claims, so it gets the rules only.
4. Confidence is recomputed with the outcome — a failed check caps it — and written. Nothing
   else ever writes confidence.

If the next stop is `confirm_resolution`, this node marks the case `awaiting_user` itself,
because the interrupt node must not touch the database. `routers.verify_destination` decides
that next stop for both this node and the router, so the two can't disagree.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from autosupport.graph.confidence import compute_confidence
from autosupport.graph.evidence import evidence_context
from autosupport.graph.routers import verify_destination
from autosupport.graph.runconfig import run_setting
from autosupport.graph.state import AgentState, VerificationResult
from autosupport.graph.verification import grounding_issues
from autosupport.llm import fast_llm, structured
from autosupport.store import cases as cases_repo
from autosupport.store import db as store_db

_SYSTEM_PROMPT = """You are a strict reviewer checking a support draft against the historical
cases it cites. You are given the draft, the cases it may rely on (with their real historical
answers) and a confidence band computed from the evidence.

`unsupported_claims`: every concrete claim in the draft (a step, a cause, a promise) that the
cited cases do NOT actually attest. Quote the claim. Empty if every claim is supported.
`overclaims`: wording that is more certain than the confidence band allows — e.g. "this will
fix it" or "the cause is X" at a `low` band. Empty if the tone matches the band.
`recommended_action`: "re_reason" if the draft's claims outran the evidence and it should be
rewritten from the same evidence; "re_retrieve" if the evidence itself is too thin to support
even a cautious answer; "none" if there is nothing to fix.
A holding reply that promises only a human follow-up and cites nothing is fine."""


class VerificationJudgement(BaseModel):
    unsupported_claims: list[str] = Field(default_factory=list)
    overclaims: list[str] = Field(default_factory=list)
    recommended_action: Literal["none", "re_reason", "re_retrieve"] = "none"


def verify(state: AgentState, config: RunnableConfig) -> dict:
    draft, evidence, decision = state["draft"], state.get("evidence", []), state["decision"]
    assessment, tau_rel = state["evidence_assessment"], run_setting(config, "tau_rel")
    rule_issues = grounding_issues(draft, evidence, decision)

    preliminary = compute_confidence(evidence, assessment, None, tau_rel)

    if decision == "escalate":
        # The escalation draft is assembled by code from state (D19): it can only cite
        # `evidence` entries and states no fix, so there is no model-written claim to check.
        judgement = VerificationJudgement()
    else:
        judgement = _claim_check(state, draft, evidence, preliminary)

    passed = not rule_issues and not judgement.unsupported_claims and not judgement.overclaims
    action = "none"
    if not passed:
        # A broken grounding rule is always a rewrite from the same evidence.
        action = "re_reason" if rule_issues or judgement.recommended_action == "none" else judgement.recommended_action
    verification = VerificationResult(
        passed=passed, unsupported_claims=judgement.unsupported_claims,
        issues=[*rule_issues, *judgement.overclaims], recommended_action=action,
    )
    confidence = compute_confidence(evidence, assessment, passed, tau_rel)
    attempts = state.get("verify_attempts", 0) + 1

    update: dict = {
        "verification": verification, "confidence": confidence,
        "verify_attempts": attempts, "tool_calls_this_round": 0,
    }
    destination = verify_destination(
        verification=verification, decision=decision, verify_attempts=attempts,
        retrieval_round=state.get("retrieval_round", 1),
        max_verify_retries=run_setting(config, "max_verify_retries"),
        max_retrieval_rounds=run_setting(config, "max_retrieval_rounds"),
        require_acceptance=run_setting(config, "require_acceptance"),
    )
    if destination == "confirm_resolution":
        conn = store_db.connect()
        try:
            cases_repo.set_status(conn, state["ticket_id"], "awaiting_user")
        finally:
            conn.close()
        update["status"] = "awaiting_user"
    return update


def _claim_check(state: AgentState, draft, evidence, preliminary) -> VerificationJudgement:
    conn = store_db.connect()
    try:
        cases_block = evidence_context(evidence, state.get("retrieved_cases", []), conn)
    finally:
        conn.close()
    return structured(fast_llm(), VerificationJudgement).invoke([
        ("system", _SYSTEM_PROMPT),
        ("user", f"Confidence band: {preliminary.band} (value {preliminary.value})\n\n"
                 f"Draft analysis:\n{draft.analysis}\n\nDraft reply:\n{draft.resolution}\n\n"
                 f"Cases the draft may rely on:\n\n{cases_block}"),
    ])
