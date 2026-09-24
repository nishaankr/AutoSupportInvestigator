"""Loads skills/*.md by name for graph nodes. Skills are never all concatenated into
one system prompt — each LLM node loads its own fixed skill plus any optional skills
`triage` added to `active_skills` (graph-design.md §3, tools-and-skills.md §2).

No fallback on a missing file: `load_skill` raises `FileNotFoundError` straight through,
so deleting a skill file visibly breaks the node that loads it, proving the prompt text
really is read from disk rather than inlined (checkpoints.md CP4 "Done")."""

from __future__ import annotations

from functools import cache
from pathlib import Path

_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


@cache
def load_skill(name: str) -> str:
    path = _SKILLS_DIR / f"{name}.md"
    return path.read_text(encoding="utf-8").strip()
