"""Nodes 8–9: `refine_retrieval` and `retrieve_variant` — a second, targeted round of retrieval.

`refine_retrieval` picks up to three query variants and fans them out in
parallel with `Send`; each `retrieve_variant` runs the full hybrid search on its own, and the
`merge_cases` reducer combines the results. Each variant's similarity is re-anchored to the
ticket before merging, so every case is compared on the same scale.

The parallel branches only write reducer keys (`retrieved_cases`, `retrieval_queries`,
`errors`). The round counter is written once, here, before the fan-out.
"""

from __future__ import annotations


from langgraph.types import Command, Send

from autosupport.graph import assessment as logic
from autosupport.graph.retrieval import reanchor, ticket_text, to_retrieved
from autosupport.graph.runconfig import run_setting
from autosupport.graph.state import AgentState, RetrievalQuery
from autosupport.ingest.text import ENTITY
from autosupport.rag import queries as rag_queries

MAX_VARIANTS = 3
VARIANT_K = 8
_MIN_RESOLUTION_CASES = 2
_MAX_KEYWORDS = 6


def refine_retrieval(state: AgentState, config) -> Command:
    round_ = state.get("retrieval_round", 1) + 1
    anchor = ticket_text(state["ticket"])
    variants = select_variants(state, config, anchor)

    return Command(
        update={"retrieval_round": round_, "tool_calls_this_round": 0},
        goto=[Send("retrieve_variant", {**v, "round": round_, "anchor_text": anchor}) for v in variants],
    )


def select_variants(state: AgentState, config, anchor: str) -> list[dict]:
    """Priority order V1, V2, V3, V4 — the first `MAX_VARIANTS` whose
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

    text, keywords = hypothesis_query(state, anchor)  # V3: always
    variants.append({"label": "hypothesis_rewrite", "text": text, "where": None, "phrases": keywords})

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


def hypothesis_query(state: AgentState, anchor: str) -> tuple[str, list[str]]:
    """V3 in Python: the hypothesis already states the problem in the
    investigator's terms, so it *is* the rewritten query; its entities (products, versions,
    components — the same `ENTITY` pattern the clustering guard uses) plus the evidence
    summaries' become the forced BM25 phrases. No model call."""
    hypothesis = state.get("hypothesis")
    if hypothesis is None:
        return anchor, []
    text = f"{hypothesis.root_cause_category}: {hypothesis.statement}"
    sources = [hypothesis.statement, *(e.summary for e in state.get("evidence", []))]
    keywords = list(dict.fromkeys(m for s in sources for m in ENTITY.findall(s)))[:_MAX_KEYWORDS]
    return text, keywords


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
