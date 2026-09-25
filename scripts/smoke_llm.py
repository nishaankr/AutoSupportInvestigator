"""Quick check that every configured model answers: sends one trivial message to the main
tier, the fast tier and the eval judge and prints each reply. A wrong model string or a bad
key fails here in seconds instead of halfway through a ticket.

Usage: python scripts/smoke_llm.py"""

from __future__ import annotations

import sys

from autosupport.config import settings
from autosupport.llm import fast_llm, judge_llm, main_llm

PROMPT = "Reply with exactly one word: pong"


def _ping(label: str, model_string: str, llm) -> None:
    reply = llm.invoke(PROMPT)
    print(f"{label:5s} {model_string}: {reply.text}")


def main() -> None:
    try:
        _ping("main", settings.main_model, main_llm())
        _ping("fast", settings.fast_model, fast_llm())
        _ping("judge", settings.judge_model, judge_llm())
    except Exception as exc:
        print(f"smoke_llm failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
