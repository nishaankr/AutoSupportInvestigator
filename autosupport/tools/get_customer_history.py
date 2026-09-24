"""`get_customer_history` (tools-and-skills.md §1, memory-design.md §3) — the current
customer's other cases. `customer_id` is bound by closure at tool-build time, never a
model-supplied argument, so a ticket can only ever see its own customer's history."""

from __future__ import annotations

from langchain_core.tools import tool

from autosupport.store import cases as cases_repo
from autosupport.store import db as store_db


def make_get_customer_history(customer_id: str, exclude_ticket_id: str):
    @tool
    def get_customer_history(limit: int = 5) -> list[dict]:
        """This customer's open cases plus their recent resolved/escalated ones (excluding
        the ticket you're working on now) — whether they've hit this before, or something
        relevant from an earlier case of theirs."""
        conn = store_db.connect()
        try:
            history = cases_repo.history_for(conn, customer_id, exclude_ticket_id=exclude_ticket_id, limit=limit)
        finally:
            conn.close()
        return [s.model_dump(mode="json") for s in history]

    return get_customer_history
