"""Node 7: `assess_evidence` (graph-design.md §4.2, §5) — the runtime evidence check.

Three steps: code metrics over the *relevant* cases (similarity >= tau_rel, weighted by
cluster_size — `graph/assessment.py`); one fast-tier structured call for the judgements code
can't compute (approach clusters, missing slots, whether a better query could close the gap,
the clarification question); then code decides the verdict, the escalation rule and
`next_action`. The model never writes a verdict or a number.

When the next step is `ask_user`, the question is generated *here* and the `awaiting_user`
status is written *here* — the node that routes into an interrupt does the side effects, so
the interrupt node itself has none (graph-design.md §7.3-§7.4).
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from autosupport.graph import assessment as logic
from autosupport.graph.runconfig import run_setting
from autosupport.graph.state import AgentState, ApproachCluster, EvidenceAssessment
from autosupport.llm import fast_llm
from autosupport.store import cases as cases_repo
from autosupport.store import db as store_db

_SYSTEM_PROMPT = """You grade the evidence gathered for a support ticket. You are given the
ticket, the historical cases judged relevant (with their answer_class and how many near-
duplicate cases each stands for), the investigator's hypothesis and evidence, and anything
the customer has already told us.

1. `clusters`: group the relevant cases by the *resolution approach* their historical answers
   took (e.g. "reset credentials", "clear cache and reinstall"). Use only case IDs from the
   relevant list. A case belongs to at most one cluster.
2. `missing_slots`: discriminating facts the clusters split on that are NOT in the ticket,
   the customer profile or the customer's earlier answers — product/version, exact error
   text, OS, plan tier. Only the customer can supply these. Empty if nothing is missing.
3. `gap_is_retrievable`: true ONLY if the ticket (plus anything the customer has already told
   us) already contains enough to write a better *query* — a narrower queue filter, a rewrite
   of the hypothesis, keywords from the customer's answers. If the gap is a missing
   discriminating fact that only the customer can supply (`missing_slots` non-empty and the
   ticket is too thin to search better), retrieval cannot close it: set this to false so the
   customer is asked first, and their answer can then drive a targeted search.
4. `history_contradicts`: true if the customer's own history contradicts the dominant
   approach (e.g. they already tried it on an earlier ticket).
5. `requires_human_action`: if the fix needs something an automated agent can't do (a refund,
   an account change, an on-site visit), say what; otherwise null.
6. `clarification_question`: if `missing_slots` is non-empty, ONE focused question that asks
   for exactly those facts, written to the customer. Otherwise null.
7. `reason`: one or two sentences on why the evidence is or isn't enough."""


class EvidenceJudgement(BaseModel):
    clusters: list[ApproachCluster] = Field(default_factory=list)
    missing_slots: list[str] = Field(default_factory=list)
    gap_is_retrievable: bool
    history_contradicts: bool = False
    requires_human_action: str | None = None
    clarification_question: str | None = None
    reason: str


def assess_evidence(state: AgentState, config: RunnableConfig) -> dict:
    tau_rel = run_setting(config, "tau_rel")
    relevant = logic.relevant_cases(state.get("retrieved_cases", []), tau_rel)
    evidence = state.get("evidence", [])

    judgement: EvidenceJudgement = fast_llm().with_structured_output(
        EvidenceJudgement, method="json_schema"
    ).invoke([("system", _SYSTEM_PROMPT), ("user", _context(state, relevant))])

    relevant_ids = {c.case_id for c in relevant}
    clusters = [  # the model may cite a case that isn't relevant; drop it rather than trust it
        ApproachCluster(label=cl.label, case_ids=[i for i in cl.case_ids if i in relevant_ids])
        for cl in judgement.clusters
    ]
    clusters = [cl for cl in clusters if cl.case_ids]

    verdict, share, dom = logic.verdict_for(
        relevant=relevant, tau_rel=tau_rel, clusters=clusters, evidence=evidence,
        missing_slots=judgement.missing_slots, history_contradicts=judgement.history_contradicts,
    )
    rule_hit = logic.escalation_rule_hit(
        classification=state["classification"], dom=dom, relevant=relevant,
        requires_human_action=judgement.requires_human_action,
        tool_log=state.get("tool_log", []), profile=state.get("customer_profile"),
    )
    next_action = logic.next_action_for(
        verdict=verdict, rule_hit=rule_hit, gap_is_retrievable=judgement.gap_is_retrievable,
        missing_slots=judgement.missing_slots, retrieval_round=state.get("retrieval_round", 1),
        clarification_count=state.get("clarification_count", 0),
        max_retrieval_rounds=run_setting(config, "max_retrieval_rounds"),
        max_clarifications=run_setting(config, "max_clarifications"),
    )

    assessment = EvidenceAssessment(
        verdict=verdict, relevant_count=len(relevant),
        top_score=relevant[0].similarity if relevant else 0.0,
        clusters=clusters, dominant_share=round(share, 3), missing_slots=judgement.missing_slots,
        gap_is_retrievable=judgement.gap_is_retrievable, escalation_rule_hit=rule_hit,
        next_action=next_action, reason=judgement.reason,
    )
    update: dict = {
        "evidence_assessment": assessment,
        "decision": next_action if next_action in ("resolve", "escalate") else None,
    }
    if next_action == "ask_user":
        question = judgement.clarification_question or (
            "Could you tell us more about: " + "; ".join(judgement.missing_slots) + "?"
        )
        conn = store_db.connect()
        try:
            cases_repo.set_status(conn, state["ticket_id"], "awaiting_user", pending_question=question)
        finally:
            conn.close()
        update.update({"pending_question": question, "status": "awaiting_user"})
    return update


def _context(state: AgentState, relevant) -> str:
    ticket, hypothesis = state["ticket"], state.get("hypothesis")
    cases = "\n\n".join(
        f"[{c.case_id}] answer_class={c.answer_class} stands_for={c.cluster_size} cases "
        f"similarity={c.similarity:.2f}\nproblem: {c.body_snippet}\nhistorical answer: {c.answer_snippet}"
        for c in relevant
    ) or "(no relevant cases)"
    evidence = "\n".join(f"- [{e.case_id}] {e.stance}: {e.summary}" for e in state.get("evidence", [])) or "(none)"
    answers = "\n".join(f"Q: {t.question}\nA: {t.answer}" for t in state.get("clarifications", [])) or "(none)"
    history = "\n".join(
        f"- [{h.case_id}] {h.status}: {h.subject}" for h in state.get("customer_history", [])
    ) or "(none)"
    profile = state.get("customer_profile")
    return (
        f"Ticket subject: {ticket.subject}\nTicket body: {ticket.body}\n\n"
        f"Customer profile: {profile.facts if profile else '(none)'}\n"
        f"Customer's earlier answers this ticket:\n{answers}\n\nCustomer history:\n{history}\n\n"
        f"Hypothesis: {hypothesis.statement if hypothesis else '(none)'}\n"
        f"Investigator's evidence:\n{evidence}\n\nRelevant historical cases:\n\n{cases}"
    )
