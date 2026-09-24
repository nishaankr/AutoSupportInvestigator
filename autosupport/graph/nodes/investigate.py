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

from pydantic import BaseModel, Field

from autosupport.graph.evidence import enrich
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


def investigate(state: AgentState) -> dict:
    tools = build_tools(state["customer_id"], state["ticket_id"])
    skill_text = load_skill("investigation")
    if "escalation" in state.get("active_skills", []):
        skill_text = f"{skill_text}\n\n{load_skill('escalation')}"

    context = _context_block(state)
    system = f"{skill_text}\n\n{context}"

    llm = main_llm().bind_tools(tools)
    ai_message = llm.invoke([("system", system), *state["messages"]])

    if ai_message.tool_calls:
        return {"messages": [ai_message]}

    transcript = _transcript([*state["messages"], ai_message])
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
    result: dict = {"messages": [ai_message], "hypothesis": hypothesis, "evidence": evidence}
    if errors:
        result["errors"] = errors
    return result


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
        if role == "human":
            continue  # the ticket itself, already in the context block
        if getattr(m, "tool_calls", None):
            calls = ", ".join(f"{c['name']}({c['args']})" for c in m.tool_calls)
            lines.append(f"[investigator called] {calls}")
        elif role == "tool":
            lines.append(f"[tool result] {m.content}")
        elif m.content:
            lines.append(f"[investigator] {m.content}")
    return "\n".join(lines) or "(no tool calls made)"
