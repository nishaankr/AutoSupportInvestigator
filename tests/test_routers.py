"""`route_after_investigate` (graph-design.md §4.2)."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from autosupport.graph.routers import route_after_investigate


def _config(budget: int) -> dict:
    return {"configurable": {"max_tool_calls_per_round": budget}}


def test_routes_to_tools_when_tool_calls_present_and_budget_left():
    state = {"messages": [AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "1"}])],
             "tool_calls_this_round": 2}
    assert route_after_investigate(state, _config(6)) == "tools"


def test_routes_to_assess_when_no_tool_calls():
    state = {"messages": [AIMessage(content="done")], "tool_calls_this_round": 0}
    assert route_after_investigate(state, _config(6)) == "assess_evidence"


def test_routes_to_assess_when_budget_exhausted_even_with_tool_calls():
    state = {"messages": [AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "1"}])],
             "tool_calls_this_round": 6}
    assert route_after_investigate(state, _config(6)) == "assess_evidence"
