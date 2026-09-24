"""Single source of truth for environment and runtime configuration.

Never call `os.getenv` or `load_dotenv` anywhere else in the codebase (CLAUDE.md).
Read values from the module-level `settings` instance instead. Import failures are
fatal and clear: a missing or invalid `.env` raises at import time with a message
naming the offending variable(s), rather than surfacing as a confusing error deep
inside a graph node.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from pydantic import Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_ENV_FILE = _REPO_ROOT / ".env"
_DEFAULT_DATA_DIR = _REPO_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_DEFAULT_ENV_FILE,
        env_prefix="AUTOSUPPORT_",
        extra="ignore",
    )

    # --- secrets & third-party config — no AUTOSUPPORT_ prefix, matches .env.example ---
    anthropic_api_key: SecretStr = Field(validation_alias="ANTHROPIC_API_KEY")
    langsmith_api_key: SecretStr | None = Field(default=None, validation_alias="LANGSMITH_API_KEY")
    langsmith_tracing: bool = Field(default=True, validation_alias="LANGSMITH_TRACING")
    langsmith_project: str = Field(default="autosupport", validation_alias="LANGSMITH_PROJECT")

    # --- models (architecture.md §2.3) ---
    main_model: str = "anthropic:claude-sonnet-5"
    fast_model: str = "anthropic:claude-haiku-4-5"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    # No main-tier temperature: Claude Sonnet 5 rejects `temperature` outright (llm.py).
    fast_temperature: float = 0.0

    # --- storage ---
    data_dir: Path = _DEFAULT_DATA_DIR

    # --- run limits & thresholds (graph-design.md §9) ---
    max_tool_calls_per_round: int = 6
    max_retrieval_rounds: int = 3
    max_clarifications: int = 2
    max_verify_retries: int = 2
    max_revisions: int = 1
    tau_rel: float = 0.76  # measured random-pair p95 (rag-design.md §9), not a guess
    require_acceptance: bool = True
    # Worst single invocation is ~70 supersteps (graph-design.md §6 derivation); 100 keeps the
    # recursion limit a pure backstop that never fires before a loop counter does (D15 F2).
    recursion_limit: int = 100

    @field_validator("main_model", "fast_model")
    @classmethod
    def _model_string_has_provider(cls, v: str) -> str:
        if ":" not in v:
            raise ValueError(
                f"model string {v!r} must be 'provider:model', e.g. 'anthropic:claude-sonnet-5'"
            )
        return v

    @field_validator("langsmith_api_key", mode="before")
    @classmethod
    def _blank_langsmith_key_is_absent(cls, v: object) -> object:
        # .env.example ships `LANGSMITH_API_KEY=` — present but blank, not unset. Without
        # this, the model_validator below only ever sees a (falsy) SecretStr, never None,
        # so an unfilled key with tracing on would silently pass validation.
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator("anthropic_api_key")
    @classmethod
    def _anthropic_key_not_blank(cls, v: SecretStr) -> SecretStr:
        if not v.get_secret_value().strip():
            raise ValueError("ANTHROPIC_API_KEY must not be blank")
        return v

    @field_validator("data_dir", mode="after")
    @classmethod
    def _anchor_relative_data_dir_to_repo_root(cls, v: Path) -> Path:
        # .env.example ships `AUTOSUPPORT_DATA_DIR=./data` (architecture.md §7), which is
        # relative. Resolve it against the repo root, not whatever the current working
        # directory happens to be when `autosupport` or pytest is invoked.
        return v if v.is_absolute() else (_REPO_ROOT / v).resolve()

    @model_validator(mode="after")
    def _langsmith_key_required_when_tracing(self) -> "Settings":
        if self.langsmith_tracing and self.langsmith_api_key is None:
            raise ValueError(
                "LANGSMITH_TRACING is true but LANGSMITH_API_KEY is not set. "
                "Set LANGSMITH_API_KEY, or set LANGSMITH_TRACING=false."
            )
        return self

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "autosupport.sqlite"

    @property
    def checkpoint_path(self) -> Path:
        return self.data_dir / "checkpoints.sqlite"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"


def _load_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as exc:
        missing = [str(err["loc"][0]) for err in exc.errors() if err["type"] == "missing"]
        lines = ["autosupport: configuration error."]
        if missing:
            lines.append(f"Missing required environment variable(s): {', '.join(missing)}.")
        lines.append("Copy .env.example to .env and fill in the values it lists.")
        lines.append(str(exc))
        sys.stderr.write("\n".join(lines) + "\n")
        raise SystemExit(1) from exc


settings = _load_settings()

# LangSmith/LangChain tracing reads its config from the process environment, not from
# pydantic-settings, which only parses .env into the object above. This is the one
# place in the codebase that writes to os.environ — a write, not an os.getenv, and it
# does not use load_dotenv, so it respects CLAUDE.md's rule. The Anthropic key is
# passed explicitly to init_chat_model (llm.py) instead of being exported here.
os.environ["LANGSMITH_TRACING"] = "true" if settings.langsmith_tracing else "false"
os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
if settings.langsmith_api_key is not None:
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key.get_secret_value()
