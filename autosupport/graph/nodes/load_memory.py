"""Node 2: `load_memory`.

Runs in parallel with `retrieve_initial` (disjoint keys). `customer_profile` is `None` for a
customer's first ticket — the expected common case, not an error — and afterwards holds what
`update_memory` kept under the write policy."""

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
