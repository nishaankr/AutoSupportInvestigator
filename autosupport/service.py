"""The UI seam (CLAUDE.md). Returns Pydantic objects, never formatted strings —
cli.py is a pure renderer over what this module returns. This is the only module
cli.py is allowed to call."""

from __future__ import annotations

from pydantic import BaseModel


class IngestResult(BaseModel):
    rows_loaded: int
    rows_after_dedup: int
    rows_below_content_threshold: int
    canonicals_indexed: int
    answer_class_distribution: dict[str, int]
    answer_class_source_distribution: dict[str, int]
    cluster_size_distribution: dict[str, int]
    duration_seconds: float


def ingest(limit: int | None = None, rebuild: bool = False) -> IngestResult:
    from autosupport.ingest.index import run

    report = run(limit=limit, rebuild=rebuild)
    return IngestResult(
        rows_loaded=report.rows_loaded,
        rows_after_dedup=report.rows_after_dedup,
        rows_below_content_threshold=report.rows_below_content_threshold,
        canonicals_indexed=report.canonicals_indexed,
        answer_class_distribution=report.answer_class_distribution,
        answer_class_source_distribution=report.answer_class_source_distribution,
        cluster_size_distribution=report.cluster_size_distribution,
        duration_seconds=report.duration_seconds,
    )


class SearchHit(BaseModel):
    case_id: str
    subject: str
    queue: str
    type: str
    answer_class: str
    cluster_size: int
    similarity: float
    score: float
    dense_rank: int | None
    lexical_rank: int | None


def search(text: str, k: int = 10, queue: str | None = None) -> list[SearchHit]:
    """CP2 temporary command (checkpoints.md) — exercises rag/queries.py end to end so hybrid
    retrieval can be verified against the real corpus before the graph exists to call it."""
    from autosupport.rag.queries import search as run_search

    where = {"queue": queue} if queue else None
    hits = run_search(text, k=k, where=where)
    return [
        SearchHit(
            case_id=h.case_id, subject=h.subject, queue=h.queue, type=h.type,
            answer_class=h.answer_class, cluster_size=h.cluster_size, similarity=h.similarity,
            score=h.score, dense_rank=h.dense_rank, lexical_rank=h.lexical_rank,
        )
        for h in hits
    ]
