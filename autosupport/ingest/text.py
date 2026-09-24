"""Shared text normalisation for ingest (docs/design/rag-design.md §2). Used by load.py
(dedup key, FTS5/embed index text), classify.py (pattern matching) and cluster.py (entity
guard). One definition, so the ingested data and the calibration harness can never drift
the way they did before rag-design.md was rewritten (see decisions.md D4/D5 history)."""

from __future__ import annotations

import re
import unicodedata

PLACEHOLDER = re.compile(r"<[a-zA-Z_]+>|\{[a-zA-Z_]+\}|\[[A-Z][a-zA-Z ]*\]")
SALUTATION = re.compile(
    r"^((dear|hello|hi|greetings|respected)\b[^.!?,]*[,.!]?"
    r"|(customer )?(support|service)( team)?,"
    r"|i hope (this|my) (message|email) (finds|reaches) you[^.!?]*[.!?])\s*",
    re.I,
)
ENTITY = re.compile(r"\b(?:[A-Z][a-z]*[A-Z0-9][A-Za-z0-9.]*|[A-Z]{2,}[0-9A-Za-z]*|[A-Za-z]+[0-9][A-Za-z0-9.]*)\b")

MIN_CONTENT_CHARS = 30


def index_text(s: str, placeholder: str = " ") -> str:
    """Display text -> index text: literal '\\n'/<br> to space, anonymisation placeholders
    removed, NFKC + whitespace collapse. Used for FTS5, embedding and dedup keys."""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\\n", " ")  # literal backslash-n in the source, not a real newline
    s = re.sub(r"<br\s*/?>", " ", s, flags=re.I)
    s = PLACEHOLDER.sub(placeholder, s)
    return re.sub(r"\s+", " ", s).strip()


def index_body(s: str) -> str:
    """index_text plus salutation stripping. Bodies only — subjects and answers never
    open with a greeting worth stripping."""
    s = index_text(s)
    for _ in range(3):  # a greeting and "I hope this message..." can stack
        s = SALUTATION.sub("", s).strip()
    return s


def embed_text(subject_ix: str, body_ix: str) -> str:
    """The dense/clustering vector text (rag-design.md §3): subject + body when a subject
    exists, body alone otherwise. Answer text is deliberately excluded — we match problems,
    not the templated phrasing of how they were answered."""
    return f"{subject_ix}. {body_ix}" if subject_ix else body_ix


def ticket_query_text(subject: str, body: str) -> str:
    """A new ticket's text, normalised exactly like the corpus before embedding, so it
    searches (and anchors similarity in) the same vector space the corpus was indexed into."""
    return embed_text(index_text(subject), index_body(body))


def derive_title(body_ix: str, max_chars: int = 90) -> str:
    """Display title for a subjectless record: the first sentence of the normalised body,
    truncated. Never fed into a vector or FTS5 index — display only."""
    first = re.split(r"(?<=[.!?])\s+", body_ix, maxsplit=1)[0]
    return first[:max_chars]
