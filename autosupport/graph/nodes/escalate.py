"""Node 12: `escalate` (graph-design.md, output-schema.md §2.1) — main tier, `escalation`
skill. Drafts the handoff for a human agent (target queue, reason, handoff summary) plus the
holding reply to the customer, and records *why* it ran.

`trigger` follows output-schema.md §2.1's precedence and is captured here into
`escalation_trigger` rather than recomputed at `persist_case`: a later `verify` pass over this
very draft would otherwise change what the precedence computes (D15 F4).
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from autosupport.graph.evidence import evidence_context
from autosupport.graph.runconfig import run_setting
from autosupport.graph.state import AgentState, DraftResponse, EscalationDraft, EscalationTrigger
from autosupport.llm import main_llm
from autosupport.skills import load_skill
from autosupport.store import db as store_db


class EscalateOutput(BaseModel):
    target_queue: str
    reason: str
    handoff_summary: str
    analysis: str
    resolution: str  # the holding reply: what happens next, never an invented fix


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


def escalate(state: AgentState, config: RunnableConfig) -> dict:
    trigger = escalation_trigger(state, config)
    conn = store_db.connect()
    try:
        evidence_block = evidence_context(state.get("evidence", []), state.get("retrieved_cases", []), conn)
    finally:
        conn.close()

    ticket, classification = state["ticket"], state["classification"]
    assessment, hypothesis = state.get("evidence_assessment"), state.get("hypothesis")
    flags = [t.args.get("reason") for t in state.get("tool_log", []) if t.name == "escalate_ticket" and t.ok]
    verification = state.get("verification")
    user_prompt = (
        f"Subject: {ticket.subject}\n\nBody: {ticket.body}\n\n"
        f"Classification: queue={classification.queue}, type={classification.type}, priority={classification.priority}\n\n"
        f"Why this is being escalated: {trigger}"
        f"{' — rule: ' + assessment.escalation_rule_hit if assessment and assessment.escalation_rule_hit else ''}\n"
        f"Assessment: {assessment.reason if assessment else '(none)'}\n"
        f"Still missing: {assessment.missing_slots if assessment else []}\n"
        f"Investigator's hypothesis: {hypothesis.statement if hypothesis else '(none)'}\n"
        f"Investigator's own escalation flags: {flags or '(none)'}\n"
        f"Customer feedback on a rejected resolution: {state.get('user_feedback') or '(none)'}\n"
        f"Unresolved verifier issues: {verification.issues + verification.unsupported_claims if verification and not verification.passed else '(none)'}\n"
        "What the customer told us when asked:\n"
        + ("\n".join(f"Q: {t.question}\nA: {t.answer}" for t in state.get("clarifications", [])) or "(nothing asked)")
        + "\n\n"
        f"Evidence gathered (cite only these IDs, as [case_id]; cite none if there is none):\n\n{evidence_block}"
    )
    output: EscalateOutput = main_llm().with_structured_output(EscalateOutput, method="json_schema").invoke(
        [("system", load_skill("escalation")), ("user", user_prompt)]
    )
    draft = DraftResponse(
        analysis=output.analysis, resolution=output.resolution,
        escalation=EscalationDraft(
            target_queue=output.target_queue, reason=output.reason, handoff_summary=output.handoff_summary,
        ),
    )
    return {"draft": draft, "decision": "escalate", "escalation_trigger": trigger}
