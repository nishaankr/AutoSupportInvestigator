"""Node 3: `retrieve_initial` — one broad hybrid search with the ticket itself as the query:
no filters, top 10, over dataset and accepted agent-resolved cases alike. Runs alongside
`load_memory`.

Because the query is the ticket, the similarities it returns are already similarity *to the
ticket*; the nodes that search with other text (`retrieve_variant`, `tools`) have to
re-anchor theirs.
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
