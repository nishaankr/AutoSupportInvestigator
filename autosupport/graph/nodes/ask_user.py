"""Node 10: `ask_user` — pauses the graph to ask the customer a question.

LangGraph re-runs an interrupt node from the top when it resumes, so everything before
`interrupt()` must be safe to run twice. Here that's only `build_payload`, which reads the
checkpoint and nothing else: no model call, no database write. The resumed run therefore shows
exactly the question the customer answered.

The question was written earlier — by the investigator in `submit_findings`, then filtered by
`assess_evidence`, which also marked the case `awaiting_user`. After the answer arrives this
node only updates state.
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
