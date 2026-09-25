"""Conditional-edge functions. Every loop these routers
close has a counter in state and a limit in `config["configurable"]`, and every branch that
can't make progress routes to `escalate` (or `persist_case` from an already-escalating run),
so the graph always terminates.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.runnables import RunnableConfig

from autosupport.graph.runconfig import run_setting
from autosupport.graph.state import AgentState, VerificationResult


def route_after_investigate(state: AgentState, config: RunnableConfig) -> Literal["tools", "assess_evidence"]:
    last = state["messages"][-1]
    budget = run_setting(config, "max_tool_calls_per_round")
    if getattr(last, "tool_calls", None) and state.get("tool_calls_this_round", 0) < budget:
        return "tools"
    return "assess_evidence"


def route_after_assess(state: AgentState) -> Literal["resolve", "escalate", "refine_retrieval", "ask_user"]:
    return state["evidence_assessment"].next_action


def verify_destination(
    *, verification: VerificationResult, decision: str, verify_attempts: int, retrieval_round: int,
    max_verify_retries: int, max_retrieval_rounds: int, require_acceptance: bool,
) -> Literal["confirm_resolution", "persist_case", "escalate", "investigate", "refine_retrieval"]:
    """The single definition of where a finished `verify` goes. `verify` calls it too, to know
    whether the next hop is `confirm_resolution` (and so must mark the case `awaiting_user`
    itself), so the node and the router can never disagree."""
    if verification.passed:
        return "confirm_resolution" if decision == "resolve" and require_acceptance else "persist_case"
    if verify_attempts >= max_verify_retries:
        return "escalate" if decision == "resolve" else "persist_case"  # escalation persists, flagged
    if verification.recommended_action == "re_retrieve" and retrieval_round < max_retrieval_rounds:
        return "refine_retrieval"
    return "investigate"  # re_reason, or re_retrieve with no retrieval rounds left


def route_after_verify(state: AgentState, config: RunnableConfig):
    return verify_destination(
        verification=state["verification"], decision=state["decision"],
        verify_attempts=state["verify_attempts"], retrieval_round=state["retrieval_round"],
        max_verify_retries=run_setting(config, "max_verify_retries"),
        max_retrieval_rounds=run_setting(config, "max_retrieval_rounds"),
        require_acceptance=run_setting(config, "require_acceptance"),
    )


def route_after_confirm(state: AgentState, config: RunnableConfig) -> Literal["persist_case", "investigate", "escalate"]:
    if state["user_acceptance"] == "accepted":
        return "persist_case"
    # `confirm_resolution` has already counted this rejection, so `<=` allows exactly
    # `max_revisions` revised drafts (`<` would allow none).
    if state["revision_count"] <= run_setting(config, "max_revisions"):
        return "investigate"
    return "escalate"
