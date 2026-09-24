"""init_chat_model for both tiers (architecture.md §2.3). Nodes call `main_llm()` /
`fast_llm()` rather than constructing a chat model themselves, so the model string,
temperature and API key stay defined in exactly one place: config.py."""

from __future__ import annotations

from functools import cache

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from autosupport.config import settings


@cache
def main_llm() -> BaseChatModel:
    """Main tier: investigation, resolution and escalation drafting, where quality
    matters (architecture.md §2.3). No `temperature` is passed: Claude Sonnet 5 (the
    default main-tier model) rejects it with a 400 ("temperature is deprecated for
    this model") — current-generation Claude models above the Haiku tier removed
    sampling controls in favour of adaptive thinking/effort."""
    return init_chat_model(
        settings.main_model,
        api_key=settings.anthropic_api_key.get_secret_value(),
    )


@cache
def fast_llm() -> BaseChatModel:
    """Fast tier: triage, evidence grading, query rewriting, verification and memory
    extraction — high-volume, structured-output steps (architecture.md §2.3)."""
    return init_chat_model(
        settings.fast_model,
        temperature=settings.fast_temperature,
        api_key=settings.anthropic_api_key.get_secret_value(),
    )
