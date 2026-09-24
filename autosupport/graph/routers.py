"""Conditional-edge functions (docs/design/graph-design.md §4.2). `route_after_investigate`
lands at CP4; the rest (`route_after_assess`, `route_after_verify`, `route_after_confirm`)
are CP5.

CP4 deviation from graph-design.md's version, tracked in decisions.md: the full design
routes a finished investigation to `assess_evidence`, which doesn't exist until CP5 — CP4
routes straight to `resolve` instead. The tool-vs-stop decision itself is unchanged.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.runnables import RunnableConfig

from autosupport.config import settings
from autosupport.graph.state import AgentState


def route_after_investigate(state: AgentState, config: RunnableConfig) -> Literal["tools", "resolve"]:
    last = state["messages"][-1]
    budget = config.get("configurable", {}).get("max_tool_calls_per_round", settings.max_tool_calls_per_round)
    if getattr(last, "tool_calls", None) and state.get("tool_calls_this_round", 0) < budget:
        return "tools"
    return "resolve"
