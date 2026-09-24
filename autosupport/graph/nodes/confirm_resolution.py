"""Node 14: `confirm_resolution` — `interrupt()` #2: acceptance (graph-design.md §7.2, §7.3).

What runs before `interrupt()`: only `build_payload` — the resolution text, the confidence
`value`/`band` and the cited case IDs, all pure reads of the checkpointed `draft`/`confidence`
(cited IDs come from applying the `CITATION` regex to the resolution text, output-schema.md
§3.4, never stored separately). No LLM, DB or tool. Re-executing it on resume rebuilds the same
payload from the same checkpoint, so it is safe. The `awaiting_user` write was done by `verify`
(the node that routes here), and the flip back to `investigating` on a rejection is done by
`service.resume_ticket` before the graph is re-invoked.

Resume value: `{"accepted": true}` or `{"accepted": false, "feedback": "..."}`.
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
