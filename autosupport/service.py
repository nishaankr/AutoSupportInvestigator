"""The UI seam (CLAUDE.md). Returns Pydantic objects, never formatted strings —
cli.py is a pure renderer over what this module returns. This is the only module
cli.py is allowed to call."""

from __future__ import annotations

import secrets
from datetime import datetime, timezone

from pydantic import BaseModel

from autosupport.graph.state import CaseResult, CaseStatus


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


class TicketOutcome(BaseModel):
    """The result of running (`new_ticket`) or looking up (`show`) a ticket. `result` is
    `None` while the ticket is still in progress or awaiting a clarification — `show` never
    invents a `CaseResult` for a case `persist_case` hasn't written yet."""

    ticket_id: str
    status: CaseStatus
    result: CaseResult | None
    pending_question: str | None


def _new_ticket_id() -> str:
    """`T-YYYYMMDD-<6 hex>` (case-persistence.md §6). Generated here, not by `intake` —
    `thread_id` has to exist before the graph can be invoked at all."""
    return f"T-{datetime.now(timezone.utc):%Y%m%d}-{secrets.token_hex(3)}"


def new_ticket(customer_id: str, subject: str, body: str) -> TicketOutcome:
    from autosupport.config import settings
    from autosupport.graph.build import compiled_graph
    from autosupport.graph.state import InputState, TicketInput

    ticket_id = _new_ticket_id()
    thread_id = f"{customer_id}:{ticket_id}"
    input_state: InputState = {
        "ticket_id": ticket_id,
        "customer_id": customer_id,
        "ticket": TicketInput(subject=subject, body=body),
    }
    config = {
        "configurable": {
            "thread_id": thread_id,
            "max_tool_calls_per_round": settings.max_tool_calls_per_round,
            "max_retrieval_rounds": settings.max_retrieval_rounds,
            "max_clarifications": settings.max_clarifications,
            "max_verify_retries": settings.max_verify_retries,
            "max_revisions": settings.max_revisions,
            "tau_rel": settings.tau_rel,
        },
        "recursion_limit": settings.recursion_limit,
    }
    output = compiled_graph().invoke(input_state, config)

    return TicketOutcome(
        ticket_id=output["ticket_id"],
        status=output["status"],
        result=output.get("final_output"),
        pending_question=output.get("pending_question"),
    )


def show(ticket_id: str) -> TicketOutcome:
    from autosupport.store import cases as cases_repo
    from autosupport.store import db as store_db

    conn = store_db.connect()
    try:
        row = cases_repo.get(conn, ticket_id)
    finally:
        conn.close()
    if row is None:
        raise ValueError(f"no such ticket: {ticket_id}")

    result = CaseResult.model_validate_json(row["final_output"]) if row["final_output"] else None
    return TicketOutcome(
        ticket_id=row["ticket_id"], status=row["status"], result=result, pending_question=row["pending_question"]
    )
