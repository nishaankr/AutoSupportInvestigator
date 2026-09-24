"""Node 6: `tools` (graph-design.md) — runs the tool calls requested in the last
`AIMessage`. A custom node rather than LangGraph's prebuilt `ToolNode`, because it also
needs to write `tool_log`/`tool_calls_this_round` and merge `search_similar_tickets` hits
into `retrieved_cases` (state-schema.md §2.5 lists `tools` as a writer of that key,
tools-and-skills.md §1).

A tool raising doesn't crash the graph: `ok=False` and an `{"error": ...}` result go back to
the model on its next turn, same as a tool that legitimately found nothing — a real boundary
(model-chosen tool + model-chosen arguments), not a case CLAUDE.md's "no error handling for
cases that can't happen" rules out.
"""

from __future__ import annotations

import json
import time

from langchain_core.messages import ToolMessage

from autosupport.graph.retrieval import reanchor, ticket_text
from autosupport.graph.state import AgentState, RetrievedCase, ToolCallRecord
from autosupport.tools import build_tools


def tools_node(state: AgentState) -> dict:
    bound = {t.name: t for t in build_tools(state["customer_id"], state["ticket_id"])}
    last = state["messages"][-1]
    calls = getattr(last, "tool_calls", None) or []
    round_ = state.get("retrieval_round", 1)

    tool_messages: list[ToolMessage] = []
    log_entries: list[ToolCallRecord] = []
    new_cases: list[RetrievedCase] = []

    for call in calls:
        name, args, call_id = call["name"], call["args"], call["id"]
        started = time.monotonic()
        tool_obj = bound.get(name)
        try:
            if tool_obj is None:
                raise KeyError(f"unknown tool {name!r}")
            result = tool_obj.func(**args)
            ok = True
        except Exception as exc:
            result = {"error": str(exc)}
            ok = False
        duration_ms = int((time.monotonic() - started) * 1000)

        tool_messages.append(ToolMessage(content=json.dumps(result, default=str), tool_call_id=call_id))
        log_entries.append(ToolCallRecord(name=name, args=args, ok=ok, duration_ms=duration_ms, round=round_))
        if ok and name == "search_similar_tickets":
            new_cases.extend(_to_retrieved_cases(result, round_))

    return {
        "messages": tool_messages,
        "tool_log": log_entries,
        "tool_calls_this_round": state.get("tool_calls_this_round", 0) + len(calls),
        # The model's search query isn't the ticket; re-anchor before merging (D15 F1).
        "retrieved_cases": reanchor(new_cases, ticket_text(state["ticket"])) if new_cases else [],
    }


def _to_retrieved_cases(hits: list[dict], round_: int) -> list[RetrievedCase]:
    return [
        RetrievedCase(
            case_id=h["case_id"], source=h["source"], subject=h["subject"],
            body_snippet=h["body_snippet"], answer_snippet=h["answer_snippet"],
            queue=h["queue"], type=h["type"], priority=h["priority"], tags=h["tags"],
            score=h["score"], similarity=h["similarity"], cluster_size=h["cluster_size"],
            answer_class=h["answer_class"], retrieval_round=round_, query_label="tool:search_similar_tickets",
        )
        for h in hits
    ]
