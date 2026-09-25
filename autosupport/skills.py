"""Loads `skills/<name>.md` for the nodes that talk to a model.

Each node loads only the skill it needs (plus the optional `escalation` skill when `triage`
switched it on), so there is never one giant prompt. A missing file raises straight away
rather than falling back to anything: if deleting a skill didn't break its node, the prompt
couldn't really be coming from that file."""

from __future__ import annotations

from functools import cache
from pathlib import Path

_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


@cache
def load_skill(name: str) -> str:
    path = _SKILLS_DIR / f"{name}.md"
    return path.read_text(encoding="utf-8").strip()
