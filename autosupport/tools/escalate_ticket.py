"""`escalate_ticket` — lets the investigator flag, mid-investigation, that a person is needed.

The tool itself only acknowledges. The flag matters afterwards: a successful call counts as
the `action_beyond_agent` escalation rule when `assess_evidence` decides the route, so the
decision stays in code rather than with whichever tool the model happened to call."""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def escalate_ticket(reason: str, target_queue: str | None = None) -> dict:
    """Flag, as soon as you're confident of it, that this ticket needs a human rather than a
    grounded fix — a refund or other action you can't take, no attested resolution in the
    evidence, or a policy/priority rule. Don't keep investigating past this point just to
    gather more evidence for a conclusion you've already reached."""
    return {"acknowledged": True, "reason": reason, "target_queue": target_queue}
