"""Node 7: `assess_evidence` (graph-design.md §4.2, §5) — pure Python, no model call.

The judgement a second model used to make here (clusters, missing facts, clarification
question, human-action flag) now arrives as part of `investigate`'s `submit_findings`
(`state["findings"]`, decisions.md D19): the investigator has just reasoned over exactly these
cases, so asking another model to re-read them cost a call and added nothing. This node turns
that judgement plus code metrics into a verdict, an escalation-rule check and `next_action`
(`graph/assessment.py`).

When the next action is `ask_user`, this node also marks the case `awaiting_user` — the node
that routes *into* an interrupt does the side effect (graph-design.md §7.4).
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from autosupport.graph import assessment as logic
from autosupport.graph.runconfig import run_setting
from autosupport.graph.state import AgentState, ApproachCluster, EvidenceAssessment
from autosupport.store import cases as cases_repo
from autosupport.store import db as store_db


def assess_evidence(state: AgentState, config: RunnableConfig) -> dict:
    tau_rel = run_setting(config, "tau_rel")
    relevant = logic.relevant_cases(state.get("retrieved_cases", []), tau_rel)
    evidence, findings = state.get("evidence", []), state["findings"]
    slots = logic.askable_slots(findings.missing_slots)  # no secrets, at most 2 (D20)

    relevant_ids = {c.case_id for c in relevant}
    clusters = [  # the model may cite a case that isn't relevant; drop it rather than trust it
        ApproachCluster(label=cl.label, case_ids=[i for i in cl.case_ids if i in relevant_ids])
        for cl in findings.clusters
    ]
    clusters = [cl for cl in clusters if cl.case_ids]

    verdict, share, dom, why = logic.verdict_for(
        relevant=relevant, tau_rel=tau_rel, clusters=clusters, evidence=evidence,
        missing_slots=slots, history_contradicts=findings.history_contradicts,
    )
    slots = logic.thin_ticket_slots(state["ticket"].body, verdict, slots)
    rule_hit = logic.escalation_rule_hit(
        classification=state["classification"], dom=dom, relevant=relevant,
        requires_human_action=findings.requires_human_action,
        tool_log=state.get("tool_log", []), profile=state.get("customer_profile"),
    )
    next_action = logic.next_action_for(
        verdict=verdict, rule_hit=rule_hit, gap_is_retrievable=findings.gap_is_retrievable,
        missing_slots=slots, retrieval_round=state.get("retrieval_round", 1),
        clarification_count=state.get("clarification_count", 0),
        max_retrieval_rounds=run_setting(config, "max_retrieval_rounds"),
        max_clarifications=run_setting(config, "max_clarifications"),
    )

    assessment = EvidenceAssessment(
        verdict=verdict, relevant_count=len(relevant),
        top_score=relevant[0].similarity if relevant else 0.0,
        clusters=clusters, dominant_share=round(share, 3), missing_slots=slots,
        gap_is_retrievable=findings.gap_is_retrievable, escalation_rule_hit=rule_hit,
        next_action=next_action, reason=why,
    )
    update: dict = {
        "evidence_assessment": assessment,
        "decision": next_action if next_action in ("resolve", "escalate") else None,
    }
    if next_action == "ask_user":
        # The model's question asks for its own slot list; if code dropped any, rebuild it.
        question = findings.clarification_question if slots == findings.missing_slots else None
        question = question or "Could you tell us more about: " + "; ".join(slots) + "?"
        conn = store_db.connect()
        try:
            cases_repo.set_status(conn, state["ticket_id"], "awaiting_user", pending_question=question)
        finally:
            conn.close()
        update.update({"pending_question": question, "status": "awaiting_user"})
    return update
