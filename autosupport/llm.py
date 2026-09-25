"""init_chat_model for both tiers and the eval judge. Nodes call
`main_llm()` / `fast_llm()` rather than constructing a chat model themselves, so model string,
temperature and API key stay defined in exactly one place: config.py.

A tier's model string picks its provider (`anthropic:` or `groq:`). Only
Anthropic takes the request options `cache_control` and `thinking`; Groq caches prompt
prefixes automatically. `anthropic_only(model, **kwargs)` returns those options for Anthropic
and nothing for other providers, so a node can pass them without knowing which one it has.
"""

from __future__ import annotations

from functools import cache

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from autosupport.config import PROVIDER_KEYS, settings


# GPT-OSS counts its reasoning against the output limit, and Groq's default limit truncated a
# `MemoryUpdate` mid-JSON. An explicit, generous ceiling costs nothing unless used.
GROQ_MAX_TOKENS = 8192


def _init(model: str, **kwargs) -> BaseChatModel:
    provider = model.split(":", 1)[0]
    field, _ = PROVIDER_KEYS[provider]
    if provider == "groq":
        kwargs.setdefault("max_tokens", GROQ_MAX_TOKENS)
    return init_chat_model(model, api_key=getattr(settings, field).get_secret_value(), **kwargs)


def is_anthropic(model: BaseChatModel) -> bool:
    return type(model).__name__ == "ChatAnthropic"


def anthropic_only(model: BaseChatModel, **options) -> dict:
    return options if is_anthropic(model) else {}


STRUCTURED_ATTEMPTS = 3


class _RetryMalformed:
    """Resamples a structured call when the provider rejects a malformed generation."""

    def __init__(self, runnable):
        self._runnable = runnable

    def invoke(self, messages):
        for attempt in range(STRUCTURED_ATTEMPTS):
            try:
                return self._runnable.invoke(messages)
            except Exception as exc:  # external API boundary
                if not is_tool_use_failure(exc) or attempt == STRUCTURED_ATTEMPTS - 1:
                    raise


def structured(model: BaseChatModel, schema):
    """`.with_structured_output(schema, method="json_schema")` for every structured step.
    On Groq the schema is only a hint unless `strict` is set (GPT-OSS 20B returned the schema's
    own shape for `MemoryUpdate` without it), and strict mode is then enforced by rejecting
    the generation — e.g. a `VerificationJudgement` missing `recommended_action` — so malformed
    generations are resampled here, once for every call site."""
    kwargs = {} if is_anthropic(model) else {"strict": True}
    return _RetryMalformed(model.with_structured_output(schema, method="json_schema", **kwargs))


# Groq's 400s for a malformed generation, all seen live with GPT-OSS: findings written
# as text or a call to a tool that wasn't offered (`tool_use_failed`), reasoning leaked into
# the output (`output_parse_failed`), JSON cut off or off-schema (`json_validate_failed`). A
# resample usually succeeds, so callers retry these; any other error is a real failure.
_GENERATION_FAILURES = ("tool_use_failed", "output_parse_failed", "json_validate_failed")


def is_tool_use_failure(exc: Exception) -> bool:
    return any(code in str(exc) for code in _GENERATION_FAILURES)


def must_call_a_tool(model: BaseChatModel) -> dict:
    """An `investigate` turn always ends in a tool call (a real tool or `submit_findings`).
    Groq enforces that with `tool_choice="required"` — measured: with "auto", GPT-OSS 20B
    sometimes reasoned until its token limit and called nothing. Anthropic can't: forcing
    tool use is incompatible with thinking, so there `submit_findings` falls back instead."""
    return {} if is_anthropic(model) else {"tool_choice": "required"}


@cache
def main_llm() -> BaseChatModel:
    """Main tier: the investigation loop, where tool choice and evidence judgement matter. No `temperature`: Claude Sonnet 5 (the default main-tier model)
    rejects it with a 400 ("temperature is deprecated for this model")."""
    return _init(settings.main_model)


@cache
def fast_llm() -> BaseChatModel:
    """Fast tier: drafting the resolution, the claim check in `verify`, and memory extraction
    (everything else is Python)."""
    # Drafting, claim-checking and extraction are short, well-specified jobs: on GPT-OSS, low
    # reasoning effort keeps them fast and stops reasoning from eating the output budget.
    extra = {"reasoning_effort": "low"} if settings.fast_model.startswith("groq:") else {}
    return _init(settings.fast_model, temperature=settings.fast_temperature, **extra)


@cache
def judge_llm() -> BaseChatModel:
    """Offline-eval judge only — never used inside the graph."""
    return _init(settings.judge_model)
