"""StateGraph wiring and `compile(checkpointer)` (graph-design.md). CP3 builds the minimum
vertical slice — `intake -> (load_memory | retrieve_initial) -> triage -> resolve ->
persist_case -> END`, all static edges, no conditional routing (`routers.py` is still empty:
CP3 has no branch to route). Extended through CP4-CP6.
"""

from __future__ import annotations

import sqlite3
from functools import cache

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from autosupport.config import settings
from autosupport.graph.nodes.intake import intake
from autosupport.graph.nodes.load_memory import load_memory
from autosupport.graph.nodes.persist_case import persist_case
from autosupport.graph.nodes.resolve import resolve
from autosupport.graph.nodes.retrieve_initial import retrieve_initial
from autosupport.graph.nodes.triage import triage
from autosupport.graph.state import CHECKPOINTED_MODELS, AgentState, InputState, OutputState


def _allowed_msgpack_modules() -> list[tuple[str, str]]:
    """Every Pydantic type that can appear in `AgentState`, named explicitly for the
    checkpointer's serde allowlist. Without this, `SqliteSaver` refuses to deserialise them
    on reload (state-schema.md §4 "Implementation notes"; the risk noted in the CP3 plan) —
    tested directly against this project's own package name before adopting it, since the
    exact spelling (module, qualname) is what the allowlist matches, not a module prefix."""
    return [(model.__module__, model.__qualname__) for model in CHECKPOINTED_MODELS]


def build_graph() -> StateGraph:
    builder = StateGraph(AgentState, input_schema=InputState, output_schema=OutputState)

    builder.add_node("intake", intake)
    builder.add_node("load_memory", load_memory)
    builder.add_node("retrieve_initial", retrieve_initial)
    builder.add_node("triage", triage)
    builder.add_node("resolve", resolve)
    builder.add_node("persist_case", persist_case)

    builder.add_edge(START, "intake")
    builder.add_edge("intake", "load_memory")
    builder.add_edge("intake", "retrieve_initial")
    builder.add_edge("load_memory", "triage")
    builder.add_edge("retrieve_initial", "triage")
    builder.add_edge("triage", "resolve")
    builder.add_edge("resolve", "persist_case")
    builder.add_edge("persist_case", END)

    return builder


@cache
def _checkpointer() -> SqliteSaver:
    settings.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.checkpoint_path, check_same_thread=False)
    serde = JsonPlusSerializer(allowed_msgpack_modules=_allowed_msgpack_modules())
    return SqliteSaver(conn, serde=serde)


@cache
def compiled_graph():
    return build_graph().compile(checkpointer=_checkpointer())
