"""Query construction and orchestration for hybrid retrieval — ties the dense arm
(dense.py), the lexical arm (lexical.py) and fusion (fusion.py) into one `search()` call."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import numpy as np

from autosupport.ingest.text import index_body, index_text
from autosupport.rag import dense, fusion, lexical
from autosupport.rag.embedder import embed
from autosupport.store import cases as cases_repo
from autosupport.store import db as store_db

OVERFETCH = 50
SNIPPET_CHARS = 600  # retrieved cases carry snippets, not full text
_TAG_COLUMNS = [f"tag_{i}" for i in range(1, 9)]


@dataclass
class SearchResult:
    case_id: str
    similarity: float  # cosine to the query — always filled, even for a lexical-only hit
    score: float  # fused RRF score, ranking-only within this call
    dense_rank: int | None
    lexical_rank: int | None
    subject: str
    queue: str
    type: str
    priority: str
    answer_class: str
    cluster_size: int
    body_snippet: str
    answer_snippet: str
    tags: list[str] = field(default_factory=list)
    source: str = "dataset"  # or "agent_resolved"


def search(
    text: str,
    k: int = 10,
    where: dict | None = None,
    conn: sqlite3.Connection | None = None,
    phrases: tuple[str, ...] | list[str] = (),
) -> list[SearchResult]:
    """Dense + BM25 -> RRF -> MMR, in one call. `where` is a single-key metadata filter
    (e.g. `{"queue": "Technical Support"}` or `{"answer_class": "resolution"}`, variants
    V2/V4) applied to **both** arms — Chroma metadata on the dense side, the FTS5
    `UNINDEXED` columns on the lexical side — so a filtered variant never leaks unfiltered
    lexical hits. `phrases` are forced quoted BM25 matches (V1/V3).

    `similarity` in the result is cosine to *this query's* text. Graph nodes that merge
    results into `retrieved_cases` re-anchor it to the ticket (`dense.similarities_to`),
    since state defines it as similarity to the ticket.

    `conn` lets a caller reuse an open connection; if omitted, one is opened and closed here.
    """
    own_conn = conn is None
    if own_conn:
        conn = store_db.connect()
    try:
        return _search(conn, text, k, where, phrases)
    finally:
        if own_conn:
            conn.close()


def _search(
    conn: sqlite3.Connection, text: str, k: int, where: dict | None, phrases
) -> list[SearchResult]:
    query_vector = embed([text])[0]

    dense_hits = dense.search(query_vector, n=OVERFETCH, where=where)
    lexical_hits = lexical.search(conn, text, n=OVERFETCH, where=where, phrases=phrases)
    dense_ids = [h.case_id for h in dense_hits]
    lexical_ids = [h.case_id for h in lexical_hits]

    fused = fusion.reciprocal_rank_fusion([dense_ids, lexical_ids])
    if not fused:
        return []
    scores = dict(fused)
    pool_ids = [case_id for case_id, _ in fused[: fusion.MMR_POOL]]

    # MMR needs every pool candidate's own embedding (not just its similarity to the query)
    # to compare candidates to each other. Cheaper than embedding OVERFETCH*2 candidates:
    # only the pool (<=30) round-trips to Chroma for its stored vector.
    collection = dense._collection()
    fetched = collection.get(ids=pool_ids, include=["embeddings"])
    embeddings = {
        cid: np.asarray(vec, dtype=np.float32) for cid, vec in zip(fetched["ids"], fetched["embeddings"])
    }

    selected = fusion.mmr(pool_ids, scores, embeddings, k=k)

    dense_sim = {h.case_id: h.similarity for h in dense_hits}
    dense_rank = {cid: i + 1 for i, cid in enumerate(dense_ids)}
    lexical_rank = {cid: i + 1 for i, cid in enumerate(lexical_ids)}

    rows = _fetch_rows(conn, selected)
    results: list[SearchResult] = []
    for case_id in selected:
        row = rows.get(case_id)
        if row is None:  # the SQLite side and Chroma should never disagree, but don't crash a search over it
            continue
        similarity = dense_sim.get(case_id)
        if similarity is None:  # lexical-only hit — never scored by Chroma's query(), so backfill it
            similarity = float(embeddings[case_id] @ query_vector)
        results.append(
            SearchResult(
                case_id=case_id,
                similarity=similarity,
                score=scores[case_id],
                dense_rank=dense_rank.get(case_id),
                lexical_rank=lexical_rank.get(case_id),
                subject=row["subject"],
                queue=row["queue"],
                type=row["type"],
                priority=row["priority"],
                answer_class=row["answer_class"],
                cluster_size=row["cluster_size"],
                body_snippet=row["body_ix"][:SNIPPET_CHARS],
                answer_snippet=row["answer_ix"][:SNIPPET_CHARS],
                tags=row["tags"],
                source=row["source"],
            )
        )
    return results


def _fetch_rows(conn: sqlite3.Connection, case_ids: list[str]) -> dict[str, dict]:
    """Display fields for each hit: `HF-` ids from `dataset_tickets`, `T-` ids (indexed
    agent-resolved cases) from `cases` — the corpus grows at runtime."""
    dataset_ids = [c for c in case_ids if c.startswith("HF-")]
    agent_ids = [c for c in case_ids if not c.startswith("HF-")]
    out: dict[str, dict] = {}
    if dataset_ids:
        placeholders = ",".join("?" for _ in dataset_ids)
        columns = (
            "case_id, subject, queue, type, priority, answer_class, cluster_size, body_ix, answer_ix, "
            + ", ".join(_TAG_COLUMNS)
        )
        for row in conn.execute(
            f"SELECT {columns} FROM dataset_tickets WHERE case_id IN ({placeholders})", dataset_ids
        ).fetchall():
            out[row["case_id"]] = {
                "subject": row["subject"], "queue": row["queue"], "type": row["type"],
                "priority": row["priority"], "answer_class": row["answer_class"],
                "cluster_size": row["cluster_size"] or 1, "body_ix": row["body_ix"] or "",
                "answer_ix": row["answer_ix"] or "", "tags": [row[c] for c in _TAG_COLUMNS if row[c]],
                "source": "dataset",
            }
    for case_id, row in cases_repo.agent_search_rows(conn, agent_ids).items():
        out[case_id] = {
            **row, "body_ix": index_body(row["body"]), "answer_ix": index_text(row["answer"]),
            "source": "agent_resolved",
        }
    return out
