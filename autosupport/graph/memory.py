"""Long-term customer memory: the write policy (memory-design.md §4, rules W1-W6) and the one
prompt rendering of a loaded profile + history.

The model only *proposes* a `MemoryUpdate`; `apply_update` is what decides what is stored,
so the graded policy is code, not prompt wording.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, Field

from autosupport.graph.state import AgentState, CaseSummary, CustomerMemory

FactKey = Literal["product", "os", "software_version", "plan", "deployment", "integration"]
PreferenceKey = Literal["contact_channel", "technical_level", "language"]
MAX_TRIED_FIXES = 15
REPEAT_UNRESOLVED_ESCALATIONS = 2  # within REPEAT_UNRESOLVED_DAYS
REPEAT_UNRESOLVED_DAYS = 90

# W3: e-mail addresses, 9+ digit runs (phone/card/account numbers, separators allowed) and
# credential words. Checked against both the value and its quote.
_SENSITIVE = re.compile(
    r"[\w.+-]+@[\w-]+\.[\w.]+|\+?(?:\d[\s().-]?){9,}|\b(?:password|passcode|api[ _-]?key|token|secret)s?\b",
    re.I,
)


# The gate in front of the extraction call (D19): each alternative is the surface form of
# something W2 can keep — a version number, an OS, a deployment/plan word, a tried fix or a
# stated preference. No match means nothing W1-W3 could admit, so the model isn't called.
# Product names alone don't open the gate (almost every ticket names one); they are still
# kept whenever another candidate brings the ticket to the model.
_CANDIDATE = re.compile(
    r"\b\d+(?:\.\d+)+\b"
    r"|\b(?:windows|macos|mac ?os|ubuntu|linux|debian|centos|red ?hat|ios|android|chrome ?os)\b"
    r"|\b(?:aws|azure|gcp|kubernetes|docker|on-?prem(?:ise)?|self-?hosted|cloud|enterprise|premium|plan|subscription)\b"
    r"|\b(?:already|tried|restart(?:ed)?|reinstall(?:ed)?|reboot(?:ed)?|cleared|didn'?t work|did not work|still)\b"
    r"|\b(?:prefer|by (?:e-?mail|phone)|not comfortable|in (?:english|german|french|spanish))\b",
    re.I,
)


def has_memory_candidates(source_text: str) -> bool:
    return bool(_CANDIDATE.search(source_text))


class RememberedFact(BaseModel):
    key: FactKey
    value: str
    quote: str = Field(description="Verbatim words from the customer's own text that state this")


class RememberedPreference(BaseModel):
    key: PreferenceKey
    value: str
    quote: str = Field(description="Verbatim words from the customer's own text that state this")


class RememberedFix(BaseModel):
    fix: str
    quote: str = Field(description="Verbatim words from the customer's own text saying it was tried or didn't work")


class MemoryUpdate(BaseModel):
    facts: list[RememberedFact] = Field(default_factory=list)
    preferences: list[RememberedPreference] = Field(default_factory=list)
    tried_fixes: list[RememberedFix] = Field(default_factory=list)
    reasoning: str


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).lower()).strip(" \"'.,")


def customer_text(state: AgentState) -> str:
    """W1's source: only what the customer wrote — the ticket, clarification answers and
    rejection feedback (the `customer`-named messages)."""
    ticket = state["ticket"]
    parts = [ticket.subject, ticket.body]
    parts += [turn.answer for turn in state.get("clarifications", [])]
    parts += [m.content for m in state.get("messages", []) if getattr(m, "name", None) == "customer"]
    return "\n".join(parts)


def _admissible(value: str, quote: str, source: str) -> bool:
    """W1 (the quote is really the customer's words) and W3 (no secrets or contact PII)."""
    quote_n = _norm(quote)
    if not quote_n or quote_n not in source:
        return False
    return not (_SENSITIVE.search(value) or _SENSITIVE.search(quote))


def apply_update(
    profile: CustomerMemory | None,
    customer_id: str,
    update: MemoryUpdate,
    source_text: str,
    ticket_id: str,
    recent_escalations: int,
) -> CustomerMemory:
    """Merge a proposed update into the stored profile under W1-W6. Pure."""
    memory = (profile or CustomerMemory(customer_id=customer_id)).model_copy(deep=True)
    source = _norm(source_text)

    # W5: facts/preferences describe current state — overwrite by key, record provenance.
    for kind, items in (("facts", update.facts), ("preferences", update.preferences)):
        target: dict[str, str] = getattr(memory, kind)
        for item in items:
            if _admissible(item.value, item.quote, source):
                target[item.key] = item.value.strip()
                memory.provenance[f"{kind}.{item.key}"] = ticket_id

    # W5: tried fixes accumulate — dedupe case-insensitively, keep the most recent N.
    for item in update.tried_fixes:
        if not _admissible(item.fix, item.quote, source):
            continue
        fix = item.fix.strip()
        memory.tried_fixes = [f for f in memory.tried_fixes if f.lower() != fix.lower()] + [fix]
    memory.tried_fixes = memory.tried_fixes[-MAX_TRIED_FIXES:]

    # W6: computed, recomputed every write (so it clears too); `vip` is never written.
    flags = [f for f in memory.flags if f != "repeat_unresolved"]
    if recent_escalations >= REPEAT_UNRESOLVED_ESCALATIONS:
        flags.append("repeat_unresolved")
    memory.flags = flags
    return memory


def profile_block(profile: CustomerMemory | None, history: list[CaseSummary]) -> str:
    """The customer's long-term memory as it appears in every prompt that uses it (triage,
    investigate, resolve) — one rendering so they can't drift. `provenance` is not shown."""
    lines: list[str] = []
    if profile:
        if profile.facts:
            lines.append("Known facts: " + "; ".join(f"{k}={v}" for k, v in profile.facts.items()))
        if profile.preferences:
            lines.append("Preferences: " + "; ".join(f"{k}={v}" for k, v in profile.preferences.items()))
        if profile.tried_fixes:
            lines.append("Already tried without success (don't re-suggest as new): " + "; ".join(profile.tried_fixes))
        if profile.flags:
            lines.append("Flags: " + ", ".join(profile.flags))
    if history:
        lines.append("Earlier tickets from this customer:")
        lines += [
            f"- [{h.case_id}] {h.status}: {h.subject}"
            + (f" — resolved with: {h.resolution_snippet}" if h.resolution_snippet else "")
            for h in history
        ]
    return "\n".join(lines) or "(no prior profile or history for this customer)"
