"""`get_ticket_by_id` — small-to-big retrieval:
search returns snippets, this returns the full record, dataset or agent-resolved/customer-
history alike."""

from __future__ import annotations

from langchain_core.tools import tool

from autosupport.store import dataset_tickets as dataset_repo
from autosupport.store import db as store_db

_TAG_COLUMNS = [f"tag_{i}" for i in range(1, 9)]


@tool
def get_ticket_by_id(case_id: str) -> dict:
    """Fetch the full subject, body and answer/resolution for one specific case by ID
    (an `HF-...` dataset case or a `T-...` agent case), when a search snippet isn't enough
    to judge whether it really supports your hypothesis."""
    conn = store_db.connect()
    try:
        if case_id.startswith("HF-"):
            row = dataset_repo.get_by_id(conn, case_id)
            if row is None:
                return {"error": "not found"}
            return {
                "case_id": row["case_id"], "source": "dataset", "subject": row["subject"],
                "body": row["body"], "answer": row["answer"], "answer_class": row["answer_class"],
                "cluster_size": row["cluster_size"] or 1, "queue": row["queue"], "type": row["type"],
                "priority": row["priority"], "tags": [row[c] for c in _TAG_COLUMNS if row[c]],
            }

        row = conn.execute("SELECT * FROM cases WHERE ticket_id = ?", (case_id,)).fetchone()
        if row is None:
            return {"error": "not found"}
        answer = None
        if row["final_output"]:
            from autosupport.graph.state import CaseResult

            answer = CaseResult.model_validate_json(row["final_output"]).resolution
        return {
            "case_id": row["ticket_id"], "source": "agent", "subject": row["subject"],
            "body": row["body"], "answer": answer, "status": row["status"],
        }
    finally:
        conn.close()
