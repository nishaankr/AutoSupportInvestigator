"""Indexing an agent-resolved case into the shared corpus (case-persistence.md §5): Chroma
+ FTS5 with `source="agent_resolved"`, so the next ticket can retrieve it exactly like a
dataset canonical. Called by the `index_case` node, and by `ingest --rebuild` to restore
agent cases after the dataset index is rebuilt."""

from __future__ import annotations

import sqlite3

from autosupport.graph.state import CaseResult, Classification
from autosupport.ingest.text import index_body, index_text, ticket_query_text
from autosupport.rag import dense
from autosupport.rag.embedder import embed
from autosupport.store import cases as cases_repo
from autosupport.store import dataset_tickets as dataset_repo

SOURCE = "agent_resolved"


def is_indexable(result: CaseResult) -> bool:
    """§5.1: only a resolution the customer explicitly accepted grows the corpus. Escalated,
    rejected and unconfirmed (`not_required`) outcomes never do."""
    return result.status == "resolved" and result.acceptance == "accepted"


def index_agent_case(conn: sqlite3.Connection, ticket_id: str, force: bool = False) -> bool:
    """Returns True if the case was (re-)indexed. `force` skips the `indexed_at` guard —
    only for restoring cases after a rebuild dropped the indexes."""
    row = cases_repo.get(conn, ticket_id)
    if row is None or row["final_output"] is None:
        return False
    if row["indexed_at"] is not None and not force:
        return False
    result = CaseResult.model_validate_json(row["final_output"])
    if not is_indexable(result):
        return False
    classification = Classification.model_validate_json(row["classification"])

    text = ticket_query_text(row["subject"], row["body"])
    # No customer_id or customer facts in the index: the corpus is shared across customers.
    metadata = {
        "source": SOURCE, "queue": classification.queue, "type": classification.type,
        "priority": classification.priority, "answer_class": "resolution", "cluster_size": 1,
        "version": -1, **dense.tag_metadata(classification.tags),
    }
    dense.upsert(ticket_id, embed([text])[0], text, metadata)
    dataset_repo.add_fts_row(
        conn, ticket_id, SOURCE, index_text(row["subject"]), index_body(row["body"]),
        index_text(result.resolution), classification.tags, classification.queue,
        classification.type, "resolution",
    )
    cases_repo.mark_indexed(conn, ticket_id)
    return True


def unindex_agent_case(conn: sqlite3.Connection, ticket_id: str) -> None:
    """Take an agent case back out of the corpus: Chroma, FTS5 and `indexed_at`. Used by the
    offline eval, whose memory example must index a resolution to test retrieval of it but
    must not leave the corpus changed afterwards."""
    dense.delete(ticket_id)
    conn.execute("DELETE FROM dataset_tickets_fts WHERE case_id = ?", (ticket_id,))
    conn.commit()
    cases_repo.mark_unindexed(conn, ticket_id)
