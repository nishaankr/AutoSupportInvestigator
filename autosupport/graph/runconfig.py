"""Run policy read from `config["configurable"]` (state-schema.md §6, graph-design.md §9),
falling back to `settings` so a graph invoked without the service layer (tests, evals) still
gets the documented defaults."""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from autosupport.config import settings


def run_setting(config: RunnableConfig, name: str):
    return config.get("configurable", {}).get(name, getattr(settings, name))
