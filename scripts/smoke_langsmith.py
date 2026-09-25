"""Checks that LangSmith accepts traces from this machine, at the cost of exactly one trace.

Tracing is off by default (`LANGSMITH_TRACING=false`); this script switches it on for itself
only. It sends one traced call — no model call, so no LLM cost — then reads the run back. A
wrong key, the wrong workspace or an exhausted monthly trace quota shows up here in a few
seconds instead of halfway through an eval.

Usage: python scripts/smoke_langsmith.py
"""

from __future__ import annotations

import sys
import time
import uuid

from autosupport.config import settings


def main() -> None:
    if settings.langsmith_api_key is None:
        sys.exit("LANGSMITH_API_KEY is not set in .env")

    from langsmith import Client, traceable, tracing_context
    from langsmith.utils import LangSmithNotFoundError

    client = Client()
    marker = uuid.uuid4().hex[:8]

    @traceable(name="autosupport-smoke")
    def ping(tag: str) -> str:
        return f"pong {tag}"

    with tracing_context(enabled=True, project_name=settings.langsmith_project):
        ping(marker)
    client.flush()

    for _ in range(15):  # ingestion is asynchronous, and a new project appears with its first trace
        time.sleep(2)
        try:
            runs = list(client.list_runs(project_name=settings.langsmith_project,
                                         filter='eq(name, "autosupport-smoke")', limit=20))
        except LangSmithNotFoundError:
            continue
        if any(marker in str(r.inputs) for r in runs):
            print(f"LangSmith OK: 1 trace accepted in project {settings.langsmith_project!r}")
            return
    sys.exit("LangSmith did not show the trace — check the key, the workspace, or the monthly trace quota")


if __name__ == "__main__":
    main()
