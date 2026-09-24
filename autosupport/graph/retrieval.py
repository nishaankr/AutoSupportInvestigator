"""Shared by every node that merges search results into `retrieved_cases`
(`retrieve_initial`, `retrieve_variant`, `tools`): conversion to `RetrievedCase` and
re-anchoring `similarity` to the ticket.

`rag.queries.search` reports cosine to *its own* query. For a hypothesis rewrite or a tool
search that isn't the ticket, but state-schema.md §2.5 defines `RetrievedCase.similarity` as
cosine to the ticket — `merge_cases` dedups on it and `assess_evidence` thresholds it against
τ_rel, so mixing anchors would compare numbers on different scales (decisions.md D15 F1).
"""

from __future__ import annotations

from autosupport.graph.state import RetrievedCase, TicketInput
from autosupport.ingest.text import ticket_query_text
from autosupport.rag import dense
from autosupport.rag.embedder import embed
from autosupport.rag.queries import SearchResult


def ticket_text(ticket: TicketInput) -> str:
    return ticket_query_text(ticket.subject, ticket.body)


def to_retrieved(hits: list[SearchResult], round_: int, label: str) -> list[RetrievedCase]:
    return [
        RetrievedCase(
            case_id=h.case_id, source="dataset", subject=h.subject,
            body_snippet=h.body_snippet, answer_snippet=h.answer_snippet,
            queue=h.queue, type=h.type, priority=h.priority, tags=h.tags,
            score=h.score, similarity=h.similarity, cluster_size=h.cluster_size,
            answer_class=h.answer_class, retrieval_round=round_, query_label=label,
        )
        for h in hits
    ]


def reanchor(cases: list[RetrievedCase], anchor_text: str) -> list[RetrievedCase]:
    """Replace each case's `similarity` with cosine to `anchor_text` (the ticket)."""
    if not cases:
        return []
    sims = dense.similarities_to([c.case_id for c in cases], embed([anchor_text])[0])
    return [c.model_copy(update={"similarity": sims.get(c.case_id, c.similarity)}) for c in cases]
