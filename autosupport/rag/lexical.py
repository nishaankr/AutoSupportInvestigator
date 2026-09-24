"""SQLite FTS5 BM25 query — the sparse (lexical) arm of hybrid retrieval (rag-design.md §6).

Flat column weights (measured indistinguishable from a subject/tags/answer-weighted variant
— rag-design.md §6), `unicode61 remove_diacritics 2` tokenizer with **no Porter stemming**
(Porter mangles entity tokens: "NAS" -> "na" — the FTS5 table itself is built that way by
`store/db.py`, this module just queries it).
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from dataclasses import dataclass

BM25_TERMS = 12
BM25_MAX_DF_FRACTION = 0.20
_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass
class LexicalHit:
    case_id: str
    bm25_score: float  # SQLite's bm25() convention: more negative is a better match


def _candidate_terms(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return {t for t in _TOKEN.findall(normalized) if len(t) > 1}


def query_terms(conn: sqlite3.Connection, text: str, n_docs: int) -> list[str]:
    """Top `BM25_TERMS` terms by IDF (rarest first), excluding any term with document
    frequency >= 20% of the index. That threshold *is* the stoplist (rag-design.md §6) —
    measured, not hand-written: it's exactly where this corpus's own boilerplate vocabulary
    (`assistance`, `issue`, `please`, `problem`...) starts."""
    candidates = _candidate_terms(text)
    if not candidates or n_docs == 0:
        return []
    cap = BM25_MAX_DF_FRACTION * n_docs
    placeholders = ",".join("?" for _ in candidates)
    rows = conn.execute(
        f"SELECT term, doc FROM dataset_tickets_fts_vocab WHERE term IN ({placeholders})",
        tuple(candidates),
    ).fetchall()
    doc_freq = {term: df for term, df in rows}
    scored = [(t, doc_freq[t]) for t in candidates if doc_freq.get(t, cap) < cap]
    scored.sort(key=lambda pair: (pair[1], pair[0]))  # rarest first, alphabetical tie-break
    return [t for t, _ in scored[:BM25_TERMS]]


# UNINDEXED FTS5 columns a caller may filter on (rag-design.md §6, §10 V2/V4). Whitelisted
# because the column name is interpolated into SQL.
FILTERABLE_COLUMNS = frozenset({"queue", "type", "answer_class", "source"})


def search(
    conn: sqlite3.Connection,
    text: str,
    n: int = 50,
    where: dict | None = None,
    phrases: tuple[str, ...] | list[str] = (),
) -> list[LexicalHit]:
    """Top-`n` canonicals by BM25 over the query's rarest terms, plus any `phrases` forced
    in as quoted matches (rag-design.md §10 V1/V3: entities from a clarification answer or
    keywords from a hypothesis rewrite). `where` filters on UNINDEXED metadata columns."""
    n_docs = conn.execute("SELECT COUNT(*) FROM dataset_tickets_fts").fetchone()[0]
    terms = query_terms(conn, text, n_docs)
    quoted = [f'"{t}"' for t in terms] + [_quote_phrase(p) for p in phrases if _quote_phrase(p)]
    if not quoted:
        return []
    filters = where or {}
    unknown = set(filters) - FILTERABLE_COLUMNS
    if unknown:
        raise ValueError(f"unfilterable lexical columns: {sorted(unknown)}")
    predicate = "".join(f" AND {col} = ?" for col in filters)
    rows = conn.execute(
        "SELECT case_id, bm25(dataset_tickets_fts) FROM dataset_tickets_fts "
        f"WHERE dataset_tickets_fts MATCH ?{predicate} ORDER BY bm25(dataset_tickets_fts) LIMIT ?",
        (" OR ".join(quoted), *filters.values(), n),
    ).fetchall()
    return [LexicalHit(case_id=case_id, bm25_score=score) for case_id, score in rows]


def _quote_phrase(phrase: str) -> str:
    tokens = _TOKEN.findall(unicodedata.normalize("NFKC", phrase).lower())
    return f'"{" ".join(tokens)}"' if tokens else ""
