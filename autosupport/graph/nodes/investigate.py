"""Node 5: `investigate` (graph-design.md) — the ReAct step. Forms/revises a hypothesis and
decides whether to call tools; the model picks tools and arguments freely within
`max_tool_calls_per_round` (route_after_investigate, routers.py).

Two LLM shapes, not one: the tool-deciding turn uses `bind_tools` (the model may emit
`tool_calls` or plain text), which is not a fixed-schema step so it doesn't go through
`.with_structured_output` — CLAUDE.md's rule applies to structured steps, and a ReAct
decision turn isn't one. Once the model stops calling tools, a second, structured call (same
turn, method="json_schema" — decisions.md D13) extracts `hypothesis` and `evidence` from the
transcript, enriched the same way `resolve` did at CP3 (`graph/evidence.enrich`,
decisions.md D13 Q2 — `resolve` no longer builds evidence itself from CP4 on).
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from autosupport.graph.evidence import enrich
from autosupport.graph.runconfig import run_setting
from autosupport.graph.state import AgentState, CustomerMemory, EvidenceItem, Hypothesis, RetrievedCase
from autosupport.llm import main_llm
from autosupport.skills import load_skill
from autosupport.store import db as store_db
from autosupport.tools import build_tools


class InvestigateConclusion(BaseModel):
    hypothesis: str
    root_cause_category: str
    supporting_case_ids: list[str] = Field(default_factory=list)
    contradicting_case_ids: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)


def investigate(state: AgentState, config: RunnableConfig) -> dict:
    skill_text = load_skill("investigation")
    if "escalation" in state.get("active_skills", []):
        skill_text = f"{skill_text}\n\n{load_skill('escalation')}"

    context = _context_block(state)
    system = f"{skill_text}\n\n{context}"
    new_messages: list = []

    # Re-entered after `refine_retrieval` or a failed `verify` (counter reset to 0, last
    # message not from the customer): say what changed. Also keeps the conversation ending on
    # a user turn — Claude 4.6+ rejects an assistant-prefill final message.
    note = _round_note(state)
    if note:
        new_messages.append(note)

    # Budget spent (the last `tools` pass used it up): go straight to the conclusion. Asking the
    # model for one more tool-deciding turn could leave a `tool_use` with no result.
    budget_spent = state.get("tool_calls_this_round", 0) >= run_setting(config, "max_tool_calls_per_round")
    if not budget_spent:
        llm = main_llm().bind_tools(build_tools(state["customer_id"], state["ticket_id"]))
        ai_message = llm.invoke([("system", system), *state["messages"], *new_messages])
        new_messages.append(ai_message)
        if ai_message.tool_calls:
            return {"messages": new_messages}

    transcript = _transcript([*state["messages"], *new_messages])
    conclusion: InvestigateConclusion = main_llm().with_structured_output(
        InvestigateConclusion, method="json_schema"
    ).invoke([
        ("system", f"{skill_text}\n\n{context}\n\nYour investigation so far:\n{transcript}"),
        ("user", "Summarise your hypothesis and the evidence for it."),
    ])

    conn = store_db.connect()
    try:
        evidence, errors = enrich(conclusion.evidence, state.get("retrieved_cases", []), conn)
    finally:
        conn.close()

    hypothesis = Hypothesis(
        statement=conclusion.hypothesis,
        root_cause_category=conclusion.root_cause_category,
        supporting_case_ids=conclusion.supporting_case_ids,
        contradicting_case_ids=conclusion.contradicting_case_ids,
    )
    result: dict = {"hypothesis": hypothesis, "evidence": evidence}
    if new_messages:
        result["messages"] = new_messages
    if errors:
        result["errors"] = errors
    return result


def _round_note(state: AgentState) -> HumanMessage | None:
    last = state["messages"][-1]
    if isinstance(last, HumanMessage) or state.get("tool_calls_this_round", 0) > 0:
        return None
    round_ = state.get("retrieval_round", 1)
    labels = [q.label for q in state.get("retrieval_queries", []) if q.round == round_ and q.round > 1]
    verification, assessment = state.get("verification"), state.get("evidence_assessment")
    parts = [f"Investigation round {round_}."]
    if labels:
        parts.append(f"Additional retrieval ran ({', '.join(labels)}); new cases are in the list above.")
    if assessment and assessment.next_action == "refine_retrieval":
        parts.append(f"The evidence check found a gap: {assessment.reason}")
    if verification and not verification.passed:
        problems = [*verification.issues, *verification.unsupported_claims]
        parts.append("The reviewer rejected your last draft: " + "; ".join(problems))
    parts.append("Revise your hypothesis and evidence accordingly.")
    return HumanMessage(content=" ".join(parts), name="system")


def _context_block(state: AgentState) -> str:
    ticket = state["ticket"]
    classification = state["classification"]
    retrieved: list[RetrievedCase] = state.get("retrieved_cases", [])
    profile: CustomerMemory | None = state.get("customer_profile")

    cases_block = "\n\n".join(
        f"[{c.case_id}] subject={c.subject!r} answer_class={c.answer_class} "
        f"cluster_size={c.cluster_size} similarity={c.similarity:.2f}\n"
        f"problem: {c.body_snippet}\nhistorical answer: {c.answer_snippet}"
        for c in retrieved
    ) or "(no retrieved cases yet — use search_similar_tickets)"
    profile_block = f"facts={profile.facts}, flags={profile.flags}" if profile else "(no prior profile)"

    return (
        f"Subject: {ticket.subject}\n\nBody: {ticket.body}\n\n"
        f"Classification: queue={classification.queue}, type={classification.type}, "
        f"priority={classification.priority}\n\n"
        f"Customer profile: {profile_block}\n\n"
        f"Retrieved cases:\n\n{cases_block}"
    )


def _transcript(messages) -> str:
    lines = []
    for m in messages:
        role = getattr(m, "type", m.__class__.__name__)
        if role == "human" and not getattr(m, "name", None):
            continue  # the ticket itself, already in the context block
        if getattr(m, "tool_calls", None):
            calls = ", ".join(f"{c['name']}({c['args']})" for c in m.tool_calls)
            lines.append(f"[investigator called] {calls}")
        elif role == "tool":
            lines.append(f"[tool result] {m.content}")
        elif m.content:
            lines.append(f"[investigator] {m.content}")
    return "\n".join(lines) or "(no tool calls made)"
