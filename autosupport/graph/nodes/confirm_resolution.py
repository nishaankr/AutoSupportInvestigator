"""Node 14: `confirm_resolution` — pauses so the customer can accept or reject the answer.

Same rule as `ask_user`: before `interrupt()` there is only `build_payload`, a pure read of the
checkpointed draft and confidence (the cited ids are pulled from the resolution text itself),
so re-running it on resume changes nothing. `verify` marked the case `awaiting_user` before
routing here; `service.resume_ticket` flips it back.

The resume value is `{"accepted": true}` or `{"accepted": false, "feedback": "..."}`. A
rejection counts one revision and sends the feedback back to the investigator.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

from autosupport.graph.state import CITATION, AgentState


def build_payload(state: AgentState) -> dict:
    resolution = state["draft"].resolution
    confidence = state["confidence"]
    return {
        "type": "confirmation",
        "ticket_id": state["ticket_id"],
        "resolution": resolution,
        "confidence": {"value": confidence.value, "band": confidence.band},
        "cited_case_ids": list(dict.fromkeys(CITATION.findall(resolution))),
    }


def confirm_resolution(state: AgentState) -> dict:
    reply = interrupt(build_payload(state))
    if reply.get("accepted"):
        return {"user_acceptance": "accepted", "user_feedback": None, "tool_calls_this_round": 0}
    feedback = reply.get("feedback") or "(no feedback given)"
    return {
        "user_acceptance": "rejected",
        "user_feedback": feedback,
        "revision_count": state.get("revision_count", 0) + 1,
        "messages": [HumanMessage(content=f"The customer rejected the proposed resolution: {feedback}", name="customer")],
        "tool_calls_this_round": 0,
        "status": "investigating",
    }
