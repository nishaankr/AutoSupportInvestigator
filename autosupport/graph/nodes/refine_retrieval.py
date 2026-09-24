"""Nodes 8-9: `refine_retrieval` and `retrieve_variant` (graph-design.md §8, rag-design.md §10).

`refine_retrieval` picks up to three query variants and fans out with `Send`; each
`retrieve_variant` runs the full hybrid pipeline (k=8) independently and the `merge_cases`
reducer dedups their results. Every variant's `similarity` is re-anchored to the *ticket*
before merging (graph/retrieval.py, D15 F1).

Written by the parallel `retrieve_variant` branches: `retrieved_cases` (merge_cases),
`retrieval_queries` (operator.add), `errors` (operator.add) — and nothing else. The counter
`retrieval_round` is written once, here, before the fan-out, so no two branches ever write
the same non-reducer key.
"""

from __future__ import annotations


from langgraph.types import Command, Send
from pydantic import BaseModel, Field

from autosupport.graph import assessment as logic
from autosupport.graph.retrieval import reanchor, ticket_text, to_retrieved
from autosupport.graph.runconfig import run_setting
from autosupport.graph.state import AgentState, RetrievalQuery
from autosupport.ingest.text import ENTITY
from autosupport.llm import fast_llm
from autosupport.rag import queries as rag_queries

MAX_VARIANTS = 3
VARIANT_K = 8
_MIN_RESOLUTION_CASES = 2

_REWRITE_PROMPT = """You rewrite a search query for a corpus of historical support tickets.
Given the ticket and the current investigation hypothesis, write `text`: a short query in the
vocabulary a *historical ticket about this exact problem* would use (product, component,
symptom) — not the customer's literal wording. `keywords`: up to 6 distinctive terms (product
names, error strings, components) that a matching ticket must contain."""


class QueryRewrite(BaseModel):
    text: str
    keywords: list[str] = Field(default_factory=list, max_length=6)


def refine_retrieval(state: AgentState, config) -> Command:
    round_ = state.get("retrieval_round", 1) + 1
    anchor = ticket_text(state["ticket"])
    variants = select_variants(state, config, anchor)

    return Command(
        update={"retrieval_round": round_, "tool_calls_this_round": 0},
        goto=[Send("retrieve_variant", {**v, "round": round_, "anchor_text": anchor}) for v in variants],
    )


def select_variants(state: AgentState, config, anchor: str) -> list[dict]:
    """rag-design.md §10 priority order V1, V2, V3, V4 — the first `MAX_VARIANTS` whose
    trigger holds; V3 always fires."""
    tau_rel = run_setting(config, "tau_rel")
    relevant = logic.relevant_cases(state.get("retrieved_cases", []), tau_rel)
    classification = state["classification"]
    clarifications = state.get("clarifications", [])
    variants: list[dict] = []

    if clarifications:  # V1: the customer just supplied the fact retrieval was missing
        answer = clarifications[-1].answer
        entities = list(dict.fromkeys(ENTITY.findall(answer)))
        variants.append({"label": "clarification_keywords", "text": f"{anchor} {answer}",
                         "where": None, "phrases": entities})

    if sum(1 for c in relevant if c.answer_class == "resolution") < _MIN_RESOLUTION_CASES:  # V2
        variants.append({"label": "resolution_only", "text": anchor,
                         "where": {"answer_class": "resolution"}, "phrases": []})

    rewrite = _rewrite(state, anchor)  # V3: always
    variants.append({"label": "hypothesis_rewrite", "text": rewrite.text,
                     "where": None, "phrases": rewrite.keywords})

    modal_queue = _modal_queue(relevant)
    if logic.top_queue_share(relevant) < logic.QUEUE_AGREEMENT_FLOOR or (
        modal_queue and modal_queue != classification.queue
    ):  # V4
        variants.append({"label": "queue_filtered", "text": anchor,
                         "where": {"queue": classification.queue}, "phrases": []})

    return variants[:MAX_VARIANTS]


def _modal_queue(relevant) -> str | None:
    queues = [c.queue for c in relevant if c.queue]
    return max(set(queues), key=queues.count) if queues else None


def _rewrite(state: AgentState, anchor: str) -> QueryRewrite:
    hypothesis, assessment = state.get("hypothesis"), state.get("evidence_assessment")
    user = (
        f"Ticket: {anchor}\nHypothesis: {hypothesis.statement if hypothesis else '(none yet)'}\n"
        f"Gap to close: {assessment.reason if assessment else '(none stated)'}\n"
        f"Missing facts: {assessment.missing_slots if assessment else []}"
    )
    return fast_llm().with_structured_output(QueryRewrite, method="json_schema").invoke(
        [("system", _REWRITE_PROMPT), ("user", user)]
    )


def retrieve_variant(payload: dict) -> dict:
    """Receives a `Send` payload (a plain dict, not `AgentState`)."""
    try:
        hits = rag_queries.search(payload["text"], k=VARIANT_K, where=payload["where"], phrases=payload["phrases"])
    except Exception as exc:  # one failed variant must not sink the round; the others still merge
        return {"errors": [f"retrieve_variant[{payload['label']}] failed: {exc}"]}

    cases = reanchor(to_retrieved(hits, payload["round"], f"variant:{payload['label']}"), payload["anchor_text"])
    return {
        "retrieved_cases": cases,
        "retrieval_queries": [
            RetrievalQuery(text=payload["text"], filters=payload["where"] or {}, k=VARIANT_K,
                           round=payload["round"], label=payload["label"], n_results=len(cases))
        ],
    }
