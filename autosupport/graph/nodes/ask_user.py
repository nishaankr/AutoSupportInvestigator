"""Node 10: `ask_user` — `interrupt()` #1: clarification (graph-design.md §7.1, §7.3).

What runs before `interrupt()`: only `build_payload`, a pure read of the checkpointed
`pending_question`, `ticket_id` and `evidence_assessment.missing_slots`. No LLM call, no DB
write, no tool call. On resume LangGraph re-executes this node from the top, so anything before
`interrupt()` runs twice — and this is safe because that code is a pure function of the
checkpoint: it rebuilds the identical payload, and the user is shown exactly the question they
answered. The question itself was generated upstream in `assess_evidence`, and the
`awaiting_user` write happened there too (graph-design.md §7.4).

After `interrupt()` returns, everything is a state update — no side effects either.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

from autosupport.graph.state import AgentState, ClarificationTurn


def build_payload(state: AgentState) -> dict:
    assessment = state.get("evidence_assessment")
    return {
        "type": "clarification",
        "ticket_id": state["ticket_id"],
        "question": state["pending_question"],
        "missing_slots": assessment.missing_slots if assessment else [],
    }


def ask_user(state: AgentState) -> dict:
    answer = interrupt(build_payload(state))
    text = answer["answer"] if isinstance(answer, dict) else str(answer)
    return {
        "clarifications": [ClarificationTurn(question=state["pending_question"], answer=text)],
        "messages": [HumanMessage(content=text, name="customer")],
        "clarification_count": state.get("clarification_count", 0) + 1,
        "pending_question": None,
        "tool_calls_this_round": 0,
        "status": "investigating",
    }
