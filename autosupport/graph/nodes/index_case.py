"""Node 17: `index_case` (graph-design.md, case-persistence.md §5) — no LLM.

Runs for every finished ticket, in parallel with `update_memory`; the index policy (only
accepted resolutions) is applied inside `index_agent_case`, so escalated, rejected and
unconfirmed outcomes are a no-op here. Writes no graph state.
"""

from __future__ import annotations

from autosupport.graph.state import AgentState
from autosupport.ingest import agent_index
from autosupport.store import db as store_db


def index_case(state: AgentState) -> dict:
    conn = store_db.connect()
    try:
        agent_index.index_agent_case(conn, state["ticket_id"])
    finally:
        conn.close()
    return {}
