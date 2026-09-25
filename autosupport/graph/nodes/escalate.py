"""Node 12: `escalate` — builds the handoff to a person, in code, with no model call.

Everything a good handoff needs is already in state: why it's escalating (the trigger and any
matched rule), what the investigator found (hypothesis, evidence, including what contradicts
it), what the customer said, and what's still missing. Assembling it from those fields is
cheaper than asking a model to rewrite them, and a template can only cite cases that are in
the evidence, so it can't invent a fix or a case (decisions.md D19).

The trigger is decided here and saved, rather than recomputed when the case is persisted:
the `verify` pass over this very draft would otherwise change what the precedence rules pick.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from autosupport.graph.runconfig import run_setting
from autosupport.graph.state import AgentState, DraftResponse, EscalationDraft, EscalationTrigger

_RULE_REASONS = {
    "critical_high_stakes": "Critical-priority ticket in a high-stakes area (security, data loss, outage or legal): "
                            "handled by a person by policy.",
    "dominant_cluster_escalated": "The closest historical cases for this problem were escalated to a specialist "
                                  "rather than resolved, so there is no attested fix to offer.",
    "action_beyond_agent": "The fix needs an action the agent can't take itself",
}


def escalation_trigger(state: AgentState, config: RunnableConfig) -> EscalationTrigger:
    if state.get("user_acceptance") == "rejected":
        return "user_rejected"
    verification = state.get("verification")
    if verification and not verification.passed and \
            state.get("verify_attempts", 0) >= run_setting(config, "max_verify_retries"):
        return "verification_failed"
    assessment = state.get("evidence_assessment")
    if assessment and assessment.escalation_rule_hit:
        return "rule"
    return "evidence_exhausted"


def _reason(state: AgentState, trigger: EscalationTrigger) -> str:
    assessment, verification = state.get("evidence_assessment"), state.get("verification")
    if trigger == "user_rejected":
        return f"The customer rejected the proposed resolution: {state.get('user_feedback') or '(no reason given)'}"
    if trigger == "verification_failed":
        issues = [*verification.issues, *verification.unsupported_claims]
        return f"The drafted resolution failed the evidence check {state.get('verify_attempts', 0)} times: {'; '.join(issues)}"
    if trigger == "rule":
        rule = assessment.escalation_rule_hit
        if rule.startswith("customer_flag:"):
            return f"Customer flagged {rule.split(':', 1)[1].replace('_', ' ')}: recent tickets went unresolved."
        text = _RULE_REASONS.get(rule, f"Escalation rule '{rule}' matched.")
        if rule == "action_beyond_agent":
            findings = state.get("findings")
            flags = [t.args.get("reason") for t in state.get("tool_log", []) if t.name == "escalate_ticket" and t.ok]
            needed = (findings.requires_human_action if findings else None) or "; ".join(f for f in flags if f)
            text += f": {needed}." if needed else "."
        return text
    rounds = state.get("retrieval_round", 1)
    return f"No grounded resolution after {rounds} retrieval round(s): {assessment.reason if assessment else ''}".strip()


def _evidence_lines(state: AgentState) -> list[str]:
    return [f"- [{e.case_id}] {e.stance}: {e.summary}" for e in state.get("evidence", [])]


def escalate(state: AgentState, config: RunnableConfig) -> dict:
    trigger = escalation_trigger(state, config)
    ticket, classification = state["ticket"], state["classification"]
    assessment, hypothesis = state.get("evidence_assessment"), state.get("hypothesis")
    reason = _reason(state, trigger)
    missing = assessment.missing_slots if assessment else []
    answers = [f"- Q: {t.question} A: {t.answer}" for t in state.get("clarifications", [])]

    handoff = "\n".join([
        f"Ticket: {ticket.subject} ({classification.queue} / {classification.type} / {classification.priority}).",
        f"Why escalated: {reason}",
        f"Investigator's hypothesis: {hypothesis.statement if hypothesis else '(none)'}",
        "Evidence:", *(_evidence_lines(state) or ["- (no supporting historical cases)"]),
        *(["Customer's answers:", *answers] if answers else []),
        *([f"Still missing: {'; '.join(missing)}"] if missing else []),
    ])
    # Every `contradicts` case is cited in the analysis (escalation skill: conflicts are acknowledged).
    contradicting = [e for e in state.get("evidence", []) if e.stance == "contradicts"]
    analysis = " ".join([
        f"Hypothesis: {hypothesis.statement}" if hypothesis else "No hypothesis could be formed.",
        *(f"[{e.case_id}] cuts against it: {e.summary}" for e in contradicting),
        f"Escalated: {reason}",
    ])

    profile = state.get("customer_profile")
    channel = profile.preferences.get("contact_channel") if profile else None
    reply = (f'Thank you for contacting us about "{ticket.subject}". We have passed your ticket to our '
             f"{classification.queue} team, who will review it and follow up with you"
             f"{f' by {channel}' if channel else ''}.")
    if missing:
        reply += f" To help them move faster, please have these details ready: {'; '.join(missing)}."

    draft = DraftResponse(
        analysis=analysis, resolution=reply,
        escalation=EscalationDraft(target_queue=classification.queue, reason=reason, handoff_summary=handoff),
    )
    return {"draft": draft, "decision": "escalate", "escalation_trigger": trigger}
