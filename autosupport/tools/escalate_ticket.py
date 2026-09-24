"""`escalate_ticket` (tools-and-skills.md §1) — CP4 scope: advisory only. It lets the model
formally flag "this needs a human" mid-investigation, logged in `tool_log`. It does not
itself route the graph to an escalation outcome; `assess_evidence`/`escalate` (CP5) own that
decision."""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def escalate_ticket(reason: str, target_queue: str | None = None) -> dict:
    """Flag, as soon as you're confident of it, that this ticket needs a human rather than a
    grounded fix — a refund or other action you can't take, no attested resolution in the
    evidence, or a policy/priority rule. Don't keep investigating past this point just to
    gather more evidence for a conclusion you've already reached."""
    return {"acknowledged": True, "reason": reason, "target_queue": target_queue}
