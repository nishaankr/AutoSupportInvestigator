"""Node 5: `investigate` — the ReAct step, where the model decides what to look at next.

The model forms a hypothesis and calls whichever tools it thinks will test it, up to
`max_tool_calls_per_round` per round. It ends the round by calling `submit_findings`, whose
arguments (`Findings`) carry the hypothesis, the evidence for and against it, and its judgement
of whether that's enough (clusters, missing facts, a question for the customer). That single
call replaced two: a second full-context "conclusion" call, and a separate model in
`assess_evidence` re-reading the same cases. The arguments are validated
with Pydantic here — structured output delivered as a tool call.

If the model answers in prose, sends invalid arguments, or runs out of tool budget, a forced
`submit_findings` call runs over a transcript, up to three times. Anthropic doesn't allow
forced tool choice while thinking is on, so that call switches thinking off. If even that
fails, the round ends with "no findings" and the ticket goes to a person instead of crashing.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from pydantic import ValidationError

from autosupport.graph.evidence import enrich
from autosupport.graph.memory import profile_block
from autosupport.graph.retrieval import PROMPT_CASES, PROMPT_SNIPPET_CHARS
from autosupport.graph.runconfig import run_setting
from autosupport.graph.state import AgentState, Findings, Hypothesis, RetrievedCase
from autosupport.llm import anthropic_only, is_tool_use_failure, main_llm, must_call_a_tool
from autosupport.skills import load_skill
from autosupport.store import db as store_db
from autosupport.tools import build_tools

CACHE = {"type": "ephemeral"}
SUBMIT = "submit_findings"
SUBMIT_FINDINGS = StructuredTool.from_function(
    func=lambda **_: "Findings recorded.", name=SUBMIT, args_schema=Findings,
    description="End this investigation round: submit your hypothesis, the evidence for and against it, "
                "and your judgement of whether the evidence is enough. Call it once, as your last action.",
)


def investigate(state: AgentState, config: RunnableConfig) -> dict:
    skill_text = load_skill("investigation")
    if "escalation" in state.get("active_skills", []):
        skill_text = f"{skill_text}\n\n{load_skill('escalation')}"
    system = f"{skill_text}\n\n{_context_block(state, run_setting(config, 'tau_rel'))}"

    # Re-entered after `refine_retrieval`, a failed `verify` or a rejection: say what changed.
    # Also keeps the conversation ending on a user turn (Claude rejects assistant prefill).
    new_messages: list = [note] if (note := _round_note(state)) else []

    ai_message = None
    if state.get("tool_calls_this_round", 0) < run_setting(config, "max_tool_calls_per_round"):
        model = main_llm()
        llm = model.bind_tools([*build_tools(state["customer_id"], state["ticket_id"]), SUBMIT_FINDINGS],
                               **must_call_a_tool(model))
        # Prompt caching: each turn re-sends tools + system + the conversation so far; every
        # turn after the first reads that prefix from cache. Anthropic needs the
        # request option; Groq caches prefixes automatically.
        try:
            ai_message = llm.invoke([("system", system), *state["messages"], *new_messages],
                                    **anthropic_only(model, cache_control=CACHE))
        except Exception as exc:  # external API boundary
            if not is_tool_use_failure(exc):
                raise
            ai_message = None  # answered in text despite required tool use: forced submit below
        if ai_message and ai_message.tool_calls and not _submit_call(ai_message):
            return {"messages": [*new_messages, ai_message]}

    findings = _findings(ai_message)
    if findings is None:  # prose answer, invalid arguments, or budget spent: one forced submit
        # The prose/invalid turn is shown in the transcript but kept out of the conversation.
        pending = [*new_messages, ai_message] if ai_message else new_messages
        ai_message = _forced_submit(system, [*state["messages"], *pending])
        findings = _findings(ai_message)
    if findings is None:
        # The model never produced valid findings, even forced. Degrade to "nothing found":
        # the evidence check then hands the ticket to a person instead of the run crashing.
        return {
            "messages": new_messages, "findings": NO_FINDINGS,
            "hypothesis": Hypothesis(statement=NO_FINDINGS.hypothesis, root_cause_category="unknown",
                                     supporting_case_ids=[]),
            "evidence": [], "errors": ["investigate: no valid submit_findings after forced attempts"],
        }
    new_messages.append(ai_message)
    # Every tool_use needs a tool_result, or the next round's conversation is invalid.
    new_messages += [
        ToolMessage("Findings recorded." if c["name"] == SUBMIT else "Not run: findings were already submitted.",
                    tool_call_id=c["id"])
        for c in ai_message.tool_calls
    ]

    conn = store_db.connect()
    try:
        evidence, errors = enrich(findings.evidence, state.get("retrieved_cases", []), conn)
    finally:
        conn.close()
    result: dict = {
        "messages": new_messages,
        "findings": findings,
        "hypothesis": Hypothesis(
            statement=findings.hypothesis, root_cause_category=findings.root_cause_category,
            supporting_case_ids=findings.supporting_case_ids, contradicting_case_ids=findings.contradicting_case_ids,
        ),
        "evidence": evidence,
    }
    if errors:
        result["errors"] = errors
    return result


def _submit_call(message) -> dict | None:
    return next((c for c in message.tool_calls if c["name"] == SUBMIT), None) if message else None


def _findings(message) -> Findings | None:
    call = _submit_call(message)
    if call is None:
        return None
    try:
        return Findings.model_validate(call["args"])
    except ValidationError:  # model-written arguments: a real boundary
        return None


FORCED_SUBMIT_ATTEMPTS = 3
NO_FINDINGS = Findings(
    hypothesis="The investigation did not produce structured findings; a person should review this ticket.",
    root_cause_category="unknown", reason="The investigation model failed to submit findings.",
)


def _forced_submit(system: str, conversation: list):
    """A call that must return `submit_findings`, or None after `FORCED_SUBMIT_ATTEMPTS`. Sent
    as system + one user turn holding the transcript, so a trailing prose turn or earlier
    thinking blocks can't make the request invalid; thinking is disabled because forced tool
    choice doesn't allow it. Resampled when Groq reports `tool_use_failed` (JSON written as
    text, or a call to a tool not offered) or the arguments don't validate."""
    model = main_llm()
    llm = model.bind_tools([SUBMIT_FINDINGS], tool_choice=SUBMIT)
    messages = [("system", system), ("user", f"Your investigation so far:\n{_transcript(conversation)}\n\n"
                                             "Submit your findings now.")]
    for _ in range(FORCED_SUBMIT_ATTEMPTS):
        try:
            ai_message = llm.invoke(messages, **anthropic_only(model, thinking={"type": "disabled"}))
        except Exception as exc:  # external API boundary
            if not is_tool_use_failure(exc):
                raise
            continue
        if _findings(ai_message) is not None:
            return ai_message
    return None


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


def _context_block(state: AgentState, tau_rel: float) -> str:
    ticket = state["ticket"]
    classification = state["classification"]
    # Only graph-retrieved cases, and only the top few: a tool search's hits are already in its
    # tool result, and leaving them out keeps this block (the system prompt) unchanged for the
    # whole round, so the prompt cache holds across every ReAct turn.
    retrieved: list[RetrievedCase] = [
        c for c in state.get("retrieved_cases", []) if not c.query_label.startswith("tool:")
    ][:PROMPT_CASES]

    # `relevant` is the same cut the evidence check applies (similarity >= tau_rel). Stating it
    # lets the model cluster exactly the cases the check will weigh; left to guess, it
    # clustered only the one case it cited.
    cases_block = "\n\n".join(
        f"[{c.case_id}] relevant={'yes' if c.similarity >= tau_rel else 'no'} source={c.source} "
        f"subject={c.subject!r} answer_class={c.answer_class} "
        f"cluster_size={c.cluster_size} similarity={c.similarity:.2f}\n"
        f"problem: {c.body_snippet[:PROMPT_SNIPPET_CHARS]}\n"
        f"historical answer: {c.answer_snippet[:PROMPT_SNIPPET_CHARS]}"
        for c in retrieved
    ) or "(no retrieved cases yet — use search_similar_tickets)"
    memory = profile_block(state.get("customer_profile"), state.get("customer_history", []))

    return (
        f"Subject: {ticket.subject}\n\nBody: {ticket.body}\n\n"
        f"Classification: queue={classification.queue}, type={classification.type}, "
        f"priority={classification.priority}\n\n"
        f"Customer memory:\n{memory}\n\n"
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
