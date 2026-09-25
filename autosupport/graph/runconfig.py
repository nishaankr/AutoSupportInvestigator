"""Reads a per-run setting (loop limits, τ_rel, require_acceptance) from the run's config,
falling back to `settings` — so a graph invoked directly, as the tests do, still gets the
usual defaults."""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from autosupport.config import settings


def run_setting(config: RunnableConfig, name: str):
    return config.get("configurable", {}).get(name, getattr(settings, name))
