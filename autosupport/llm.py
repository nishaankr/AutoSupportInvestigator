"""init_chat_model for both tiers and the eval judge (architecture.md §2.3). Nodes call
`main_llm()` / `fast_llm()` rather than constructing a chat model themselves, so model string,
temperature and API key stay defined in exactly one place: config.py.

A tier's model string picks its provider (`anthropic:` or `groq:`, decisions.md D20). Only
Anthropic takes the request options `cache_control` and `thinking`; Groq caches prompt
prefixes automatically. `anthropic_only(model, **kwargs)` returns those options for Anthropic
and nothing for other providers, so a node can pass them without knowing which one it has.
"""

from __future__ import annotations

from functools import cache

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from autosupport.config import PROVIDER_KEYS, settings


def _init(model: str, **kwargs) -> BaseChatModel:
    field, _ = PROVIDER_KEYS[model.split(":", 1)[0]]
    return init_chat_model(model, api_key=getattr(settings, field).get_secret_value(), **kwargs)


def is_anthropic(model: BaseChatModel) -> bool:
    return type(model).__name__ == "ChatAnthropic"


def anthropic_only(model: BaseChatModel, **options) -> dict:
    return options if is_anthropic(model) else {}


def structured(model: BaseChatModel, schema):
    """`.with_structured_output(schema, method="json_schema")` for every structured step (D13).
    On Groq the schema is only a hint unless `strict` is set — measured: GPT-OSS 20B returned
    the schema's own shape for `MemoryUpdate` without it (D20)."""
    return model.with_structured_output(schema, method="json_schema", **({} if is_anthropic(model) else {"strict": True}))


def is_tool_use_failure(exc: Exception) -> bool:
    """Groq's 400 `tool_use_failed`: tool use was required but the model wrote its answer as
    plain text instead (measured with GPT-OSS 20B writing `Findings` JSON as content, D20)."""
    return "tool_use_failed" in str(exc)


def must_call_a_tool(model: BaseChatModel) -> dict:
    """An `investigate` turn always ends in a tool call (a real tool or `submit_findings`).
    Groq enforces that with `tool_choice="required"` — measured: with "auto", GPT-OSS 20B
    sometimes reasoned until its token limit and called nothing (D20). Anthropic can't: forcing
    tool use is incompatible with thinking, so there `submit_findings` falls back instead."""
    return {} if is_anthropic(model) else {"tool_choice": "required"}


@cache
def main_llm() -> BaseChatModel:
    """Main tier: the investigation loop, where tool choice and evidence judgement matter
    (architecture.md §2.3). No `temperature`: Claude Sonnet 5 (the default main-tier model)
    rejects it with a 400 ("temperature is deprecated for this model")."""
    return _init(settings.main_model)


@cache
def fast_llm() -> BaseChatModel:
    """Fast tier: drafting the resolution, the claim check in `verify`, and memory extraction
    (architecture.md §2.3; D19 moved everything else to Python)."""
    return _init(settings.fast_model, temperature=settings.fast_temperature)


@cache
def judge_llm() -> BaseChatModel:
    """Offline-eval judge only (evaluation-design.md §4) — never used inside the graph."""
    return _init(settings.judge_model)
