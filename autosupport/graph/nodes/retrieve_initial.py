"""Node 3: `retrieve_initial` (graph-design.md) — broad semantic search over the whole
index, no metadata filters, k=10. Runs in parallel with `load_memory`.

The query *is* the ticket text, so `similarity` from the search is already anchored to the
ticket — no re-anchoring needed here (compare `retrieve_variant`, `tools`).
`rag/queries.search` only reads `dataset_tickets` until CP6 indexes agent-resolved cases.
"""

from __future__ import annotations

from autosupport.graph.retrieval import ticket_text, to_retrieved
from autosupport.graph.state import AgentState, RetrievalQuery
from autosupport.rag import queries as rag_queries

K = 10


def retrieve_initial(state: AgentState) -> dict:
    query_text = ticket_text(state["ticket"])
    hits = rag_queries.search(query_text, k=K)
    return {
        "retrieved_cases": to_retrieved(hits, round_=1, label="initial"),
        "retrieval_queries": [
            RetrievalQuery(text=query_text, filters={}, k=K, round=1, label="initial", n_results=len(hits))
        ],
        "retrieval_round": 1,
    }
