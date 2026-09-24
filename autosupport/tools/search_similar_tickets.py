"""`search_similar_tickets` (tools-and-skills.md §1) — targeted, mid-investigation retrieval,
distinct from `retrieve_initial`'s one broad pass. Runs the full hybrid pipeline."""

from __future__ import annotations

from langchain_core.tools import tool

from autosupport.graph.retrieval import PROMPT_SNIPPET_CHARS
from autosupport.rag.queries import search


@tool
def search_similar_tickets(query: str, k: int = 8, queue: str | None = None) -> list[dict]:
    """Search the historical ticket corpus for cases similar to `query`. Use different
    wording or a narrower `queue` filter than the initial retrieval when you need to check a
    specific angle — don't repeat a search you've effectively already run. `queue` must be
    one of the known queue names if given."""
    where = {"queue": queue} if queue else None
    hits = search(query, k=k, where=where)
    return [
        {
            "case_id": h.case_id, "source": h.source, "subject": h.subject, "queue": h.queue, "type": h.type,
            "priority": h.priority, "answer_class": h.answer_class, "cluster_size": h.cluster_size,
            "similarity": h.similarity, "score": h.score,
            # Short snippets: the result stays in the conversation for every later turn;
            # `get_ticket_by_id` gives the full record when one case matters (D18).
            "body_snippet": h.body_snippet[:PROMPT_SNIPPET_CHARS],
            "answer_snippet": h.answer_snippet[:PROMPT_SNIPPET_CHARS], "tags": h.tags,
        }
        for h in hits
    ]
