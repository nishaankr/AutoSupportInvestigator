"""Node 3: `retrieve_initial` (graph-design.md) — broad semantic search over the whole
index, no metadata filters, k=10. Runs in parallel with `load_memory`.

The query text is normalised the same way the corpus was before embedding
(`ingest/text.py`, reused rather than re-implemented — rag-design.md §2-§3), so a ticket
with a stray literal `\\n` or an anonymisation-placeholder-shaped fragment searches the
same vector space the corpus was indexed into.

`rag/queries.search` only reads `dataset_tickets` at CP3 (CP6 extends it to also surface
indexed `agent_resolved` cases, case-persistence.md §5), so every result here is
`source="dataset"`.
"""

from __future__ import annotations

from autosupport.graph.state import AgentState, RetrievalQuery, RetrievedCase
from autosupport.ingest.text import embed_text, index_body, index_text
from autosupport.rag import queries as rag_queries

K = 10


def retrieve_initial(state: AgentState) -> dict:
    ticket = state["ticket"]
    query_text = embed_text(index_text(ticket.subject), index_body(ticket.body))

    hits = rag_queries.search(query_text, k=K)

    retrieved = [
        RetrievedCase(
            case_id=h.case_id,
            source="dataset",
            subject=h.subject,
            body_snippet=h.body_snippet,
            answer_snippet=h.answer_snippet,
            queue=h.queue,
            type=h.type,
            priority=h.priority,
            tags=h.tags,
            score=h.score,
            similarity=h.similarity,
            cluster_size=h.cluster_size,
            answer_class=h.answer_class,
            retrieval_round=1,
            query_label="initial",
        )
        for h in hits
    ]
    query_log = RetrievalQuery(text=query_text, filters={}, k=K, round=1, label="initial", n_results=len(hits))

    return {
        "retrieved_cases": retrieved,
        "retrieval_queries": [query_log],
        "retrieval_round": 1,
    }
