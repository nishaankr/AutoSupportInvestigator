"""@tool definitions: search_similar_tickets, get_ticket_by_id, get_customer_history,
compute_queue_stats, escalate_ticket.

`build_tools(customer_id, ticket_id)` assembles the bound set for one graph run — everything
except `get_customer_history` is stateless and shared; that one is built fresh per run so
`customer_id` is closed over rather than model-suppliable."""

from __future__ import annotations

# Aliased on import: each submodule's @tool function shares its module's name, and a plain
# `from .search_similar_tickets import search_similar_tickets` would rebind the package
# attribute `autosupport.tools.search_similar_tickets` from the submodule to the function,
# breaking any `monkeypatch.setattr("autosupport.tools.search_similar_tickets.search", ...)`
# by dotted path. Aliasing keeps the submodule reachable at its own name.
from autosupport.tools.compute_queue_stats import compute_queue_stats as _compute_queue_stats
from autosupport.tools.escalate_ticket import escalate_ticket as _escalate_ticket
from autosupport.tools.get_customer_history import make_get_customer_history
from autosupport.tools.get_ticket_by_id import get_ticket_by_id as _get_ticket_by_id
from autosupport.tools.search_similar_tickets import search_similar_tickets as _search_similar_tickets


def build_tools(customer_id: str, ticket_id: str) -> list:
    return [
        _search_similar_tickets,
        _get_ticket_by_id,
        make_get_customer_history(customer_id, exclude_ticket_id=ticket_id),
        _compute_queue_stats,
        _escalate_ticket,
    ]
