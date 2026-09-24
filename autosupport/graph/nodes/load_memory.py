"""Node 2: `load_memory` (graph-design.md, memory-design.md §3).

Runs in parallel with `retrieve_initial`. At CP3, nothing has ever written `customers`
(that's `update_memory`, CP6), so `customer_profile` is `None` for every customer until
then — the expected common case, not an error."""

from __future__ import annotations

from autosupport.graph.state import AgentState
from autosupport.store import cases as cases_repo
from autosupport.store import customers as customers_repo
from autosupport.store import db as store_db


def load_memory(state: AgentState) -> dict:
    conn = store_db.connect()
    try:
        profile = customers_repo.get(conn, state["customer_id"])
        history = cases_repo.history_for(conn, state["customer_id"], exclude_ticket_id=state["ticket_id"])
    finally:
        conn.close()

    return {"customer_profile": profile, "customer_history": history}
