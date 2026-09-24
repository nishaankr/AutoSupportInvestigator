"""Fake credentials so `autosupport.config` can be imported during test collection
without a real .env. `setdefault` only — a developer's real environment or .env
still wins if either is already set, since pydantic-settings ranks explicit env vars
above the dotenv file, and these are only defaults."""

import os

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-00000000000000000000000000")
os.environ.setdefault("LANGSMITH_TRACING", "false")
