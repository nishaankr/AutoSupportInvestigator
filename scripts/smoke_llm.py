"""CP0 smoke test. Model-string mistakes are the single most common way to lose 30
minutes on day one (docs/project/checkpoints.md) — this sends one trivial message to each
tier and prints the reply, so a bad model string or missing key fails immediately
and visibly rather than inside a graph run."""

from __future__ import annotations

import sys

from autosupport.config import settings
from autosupport.llm import fast_llm, main_llm

PROMPT = "Reply with exactly one word: pong"


def _ping(label: str, model_string: str, llm) -> None:
    reply = llm.invoke(PROMPT)
    print(f"{label:5s} {model_string}: {reply.text}")


def main() -> None:
    try:
        _ping("main", settings.main_model, main_llm())
        _ping("fast", settings.fast_model, fast_llm())
    except Exception as exc:
        print(f"smoke_llm failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
