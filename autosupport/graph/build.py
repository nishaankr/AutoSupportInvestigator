"""StateGraph wiring and `compile(checkpointer)` — the full CP5 topology of
graph-design.md §2 (every router in routers.py, every loop counter-bounded). `persist_case ->
END` until CP6 adds `index_case | update_memory` after it.
"""

from __future__ import annotations

import sqlite3
from functools import cache

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from autosupport.config import settings
from autosupport.graph.nodes.ask_user import ask_user
from autosupport.graph.nodes.assess_evidence import assess_evidence
from autosupport.graph.nodes.confirm_resolution import confirm_resolution
from autosupport.graph.nodes.escalate import escalate
from autosupport.graph.nodes.intake import intake
from autosupport.graph.nodes.investigate import investigate
from autosupport.graph.nodes.load_memory import load_memory
from autosupport.graph.nodes.persist_case import persist_case
from autosupport.graph.nodes.refine_retrieval import refine_retrieval, retrieve_variant
from autosupport.graph.nodes.resolve import resolve
from autosupport.graph.nodes.retrieve_initial import retrieve_initial
from autosupport.graph.nodes.tools import tools_node
from autosupport.graph.nodes.triage import triage
from autosupport.graph.nodes.verify import verify
from autosupport.graph.routers import (
    route_after_assess, route_after_confirm, route_after_investigate, route_after_verify,
)
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

    for name, fn in [
        ("intake", intake), ("load_memory", load_memory), ("retrieve_initial", retrieve_initial),
        ("triage", triage), ("investigate", investigate), ("tools", tools_node),
        ("assess_evidence", assess_evidence), ("retrieve_variant", retrieve_variant),
        ("ask_user", ask_user), ("resolve", resolve), ("escalate", escalate), ("verify", verify),
        ("confirm_resolution", confirm_resolution), ("persist_case", persist_case),
    ]:
        builder.add_node(name, fn)
    # `refine_retrieval` fans out with `Send` (returns a Command), so it declares its targets.
    builder.add_node("refine_retrieval", refine_retrieval, destinations=("retrieve_variant",))

    builder.add_edge(START, "intake")
    builder.add_edge("intake", "load_memory")
    builder.add_edge("intake", "retrieve_initial")
    builder.add_edge("load_memory", "triage")
    builder.add_edge("retrieve_initial", "triage")
    builder.add_edge("triage", "investigate")
    builder.add_conditional_edges("investigate", route_after_investigate, ["tools", "assess_evidence"])
    builder.add_edge("tools", "investigate")
    builder.add_conditional_edges(
        "assess_evidence", route_after_assess, ["resolve", "escalate", "refine_retrieval", "ask_user"]
    )
    builder.add_edge("retrieve_variant", "investigate")
    builder.add_edge("ask_user", "investigate")
    builder.add_edge("resolve", "verify")
    builder.add_edge("escalate", "verify")
    builder.add_conditional_edges(
        "verify", route_after_verify,
        ["confirm_resolution", "persist_case", "escalate", "investigate", "refine_retrieval"],
    )
    builder.add_conditional_edges(
        "confirm_resolution", route_after_confirm, ["persist_case", "investigate", "escalate"]
    )
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
