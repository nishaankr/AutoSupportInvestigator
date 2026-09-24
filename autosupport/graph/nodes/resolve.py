"""Node 11: `resolve` (graph-design.md) — main tier, drafts a grounded resolution.

At the full design, `investigate` produces `evidence` (output-schema.md §3.2) and `resolve`
only drafts prose against it. CP3 has no `investigate`, so `resolve` does both in one call:
its structured output includes the `EvidenceItem` list alongside the draft, and this node
runs the same `graph/evidence.enrich()` that `investigate` will reuse at CP4
(docs/project/decisions.md CP3 entry, Q2). Nothing about `enrich()` itself is CP3-specific.

CP3 has no `escalate` node (graph-design.md's decision routing isn't built yet), so every
ticket resolves (F8 in the CP3 plan) — the prompt below is explicit that the model must say
plainly, in the resolution text, when the retrieved evidence doesn't actually support a fix,
rather than inventing one. CP5 adds the real `investigate ⇄ tools` loop and `escalate`.

`method="json_schema"` (not the `with_structured_output` default) — reproduced directly
against the live API: Claude Sonnet 5 runs with reasoning enabled by default, and
`langchain-anthropic`'s default `method="function_calling"` doesn't force the tool call when
reasoning is on, which let the model return an incomplete/malformed tool call (missing
`resolution`, or evidence-shaped text leaking into `analysis`) on a nontrivial prompt, every
time. Claude's native structured-output feature (`method="json_schema"`) doesn't depend on
forced tool choice and was reliable in the same repro. `triage.py` carries the same fix.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from autosupport.graph.evidence import enrich
from autosupport.graph.state import AgentState, DraftResponse, EvidenceItem, RetrievedCase
from autosupport.llm import main_llm
from autosupport.store import db as store_db

_SYSTEM_PROMPT = """You draft a customer-facing resolution for a support ticket, grounded
only in the retrieved historical cases you are shown. You may not invent a fix that isn't
attested by at least one retrieved case.

For every case you rely on, add it to `evidence` with a one-line summary (problem -> what
resolved it) and a stance: "supports" if it backs your resolution, "contradicts" if it
points to a different cause or fix, "neutral" if it's related but doesn't bear on your
conclusion either way. Cite every case you use inline, in your `analysis` and `resolution`
text, as `[case_id]`, e.g. `[HF-10432]`. Only cite a case that's also in `evidence`.

Cases are labelled by `answer_class`: "resolution" means the historical answer contains an
actual fix; "clarification_request" means the historical answer asked the customer a
question; "escalation" means the historical answer was a handoff with no grounded fix. If
the retrieved cases are mostly not `resolution`-class, or don't match this ticket well, say
so plainly in `resolution` rather than manufacturing a confident-sounding fix — a resolution
draft is allowed to tell the customer what's being looked into instead of a fix.

`analysis` is written for a support agent reviewing your work: the hypothesis, how the
evidence agreed or conflicted, what was ruled out. `resolution` is the customer-facing
reply."""


class ResolveOutput(BaseModel):
    evidence: list[EvidenceItem] = Field(default_factory=list)
    analysis: str
    resolution: str


def resolve(state: AgentState) -> dict:
    ticket = state["ticket"]
    retrieved = state.get("retrieved_cases", [])
    classification = state["classification"]

    cases_block = "\n\n".join(_format_case(c) for c in retrieved) or "(no retrieved cases)"
    user_prompt = (
        f"Subject: {ticket.subject}\n\nBody: {ticket.body}\n\n"
        f"Classification: queue={classification.queue}, type={classification.type}, "
        f"priority={classification.priority}\n\n"
        f"Retrieved cases:\n\n{cases_block}"
    )

    output: ResolveOutput = main_llm().with_structured_output(ResolveOutput, method="json_schema").invoke(
        [("system", _SYSTEM_PROMPT), ("user", user_prompt)]
    )

    conn = store_db.connect()
    try:
        evidence, enrich_errors = enrich(output.evidence, retrieved, conn)
    finally:
        conn.close()

    draft = DraftResponse(analysis=output.analysis, resolution=output.resolution, escalation=None)
    result: dict = {"draft": draft, "evidence": evidence, "decision": "resolve"}
    if enrich_errors:
        result["errors"] = enrich_errors
    return result


def _format_case(c: RetrievedCase) -> str:
    return (
        f"[{c.case_id}] subject={c.subject!r} answer_class={c.answer_class} "
        f"cluster_size={c.cluster_size} similarity={c.similarity:.2f}\n"
        f"problem: {c.body_snippet}\nhistorical answer: {c.answer_snippet}"
    )
