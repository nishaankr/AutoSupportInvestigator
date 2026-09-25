"""Node 1: `intake` — records the ticket and sets every counter to its starting value.

The `cases` row is written straight away, so a ticket exists in the system of record even if
everything after this fails. Nothing before this node can pause, so a resume never runs it
again. The ticket and thread ids arrive already made by `service.new_ticket()`; this node just
copies the thread id into state.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from autosupport.graph.state import AgentState
from autosupport.store import cases as cases_repo
from autosupport.store import db as store_db


def intake(state: AgentState, config: RunnableConfig) -> dict:
    ticket = state["ticket"]
    thread_id = config["configurable"]["thread_id"]
    conn = store_db.connect()
    try:
        cases_repo.insert_open(conn, state["ticket_id"], state["customer_id"], thread_id, ticket)
    finally:
        conn.close()

    return {
        "thread_id": thread_id,
        "status": "open",
        "messages": [HumanMessage(content=f"{ticket.subject}\n\n{ticket.body}")],
        "retrieval_round": 0,
        "clarification_count": 0,
        "verify_attempts": 0,
        "revision_count": 0,
        "tool_calls_this_round": 0,
        "user_acceptance": None,
    }
