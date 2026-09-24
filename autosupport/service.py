"""The UI seam (CLAUDE.md). Returns Pydantic objects, never formatted strings —
cli.py is a pure renderer over what this module returns. This is the only module
cli.py is allowed to call."""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel

from autosupport.graph.state import CaseResult, CaseStatus, CaseSummary, CustomerMemory


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


class PendingInterrupt(BaseModel):
    """What a paused ticket is waiting for — the `interrupt()` payload of `ask_user` or
    `confirm_resolution` (graph-design.md §7), read back from the checkpoint."""

    type: Literal["clarification", "confirmation"]
    question: str | None = None
    missing_slots: list[str] = []
    resolution: str | None = None
    confidence: dict | None = None
    cited_case_ids: list[str] = []


class TicketOutcome(BaseModel):
    """The result of running (`new_ticket`, `resume_ticket`) or looking up (`show`) a ticket.
    `result` is `None` until `persist_case` has written the `CaseResult`; `interrupt` is set
    while the ticket is paused waiting on the customer."""

    ticket_id: str
    status: CaseStatus
    result: CaseResult | None
    pending_question: str | None
    interrupt: PendingInterrupt | None = None


class CaseListItem(BaseModel):
    ticket_id: str
    customer_id: str
    status: CaseStatus
    subject: str
    pending_question: str | None
    updated_at: str


def _new_ticket_id() -> str:
    """`T-YYYYMMDD-<6 hex>` (case-persistence.md §6). Generated here, not by `intake` —
    `thread_id` has to exist before the graph can be invoked at all."""
    return f"T-{datetime.now(timezone.utc):%Y%m%d}-{secrets.token_hex(3)}"


def _run_config(
    thread_id: str, ticket_id: str, customer_id: str, require_acceptance: bool | None = None
) -> dict:
    """Run policy travels in `config["configurable"]` (graph-design.md §9), so a resume uses
    exactly the limits the ticket started with. `metadata` tags every LangSmith trace with
    the ticket's identity, so a trace (or an eval score) can be found from a ticket id."""
    from autosupport.config import settings

    return {
        "configurable": {
            "thread_id": thread_id,
            "max_tool_calls_per_round": settings.max_tool_calls_per_round,
            "max_retrieval_rounds": settings.max_retrieval_rounds,
            "max_clarifications": settings.max_clarifications,
            "max_verify_retries": settings.max_verify_retries,
            "max_revisions": settings.max_revisions,
            "tau_rel": settings.tau_rel,
            "require_acceptance": settings.require_acceptance if require_acceptance is None else require_acceptance,
        },
        "metadata": {"ticket_id": ticket_id, "customer_id": customer_id, "thread_id": thread_id},
        "recursion_limit": settings.recursion_limit,
    }


def _outcome(ticket_id: str, thread_id: str) -> TicketOutcome:
    """Built from the checkpoint, the one source that knows about a pause."""
    from autosupport.graph.build import compiled_graph

    snap = compiled_graph().get_state({"configurable": {"thread_id": thread_id}})
    values = snap.values
    pending = None
    if snap.interrupts:
        payload = snap.interrupts[0].value
        pending = PendingInterrupt(
            type=payload["type"], question=payload.get("question"),
            missing_slots=payload.get("missing_slots", []), resolution=payload.get("resolution"),
            confidence=payload.get("confidence"), cited_case_ids=payload.get("cited_case_ids", []),
        )
    return TicketOutcome(
        ticket_id=ticket_id, status=values["status"], result=values.get("final_output"),
        pending_question=values.get("pending_question"), interrupt=pending,
    )


def new_ticket(
    customer_id: str, subject: str, body: str,
    thread_id: str | None = None, require_acceptance: bool | None = None,
) -> TicketOutcome:
    """`thread_id` / `require_acceptance` are overridden only by the offline eval
    (evaluation-design.md §3: `eval:{run_id}:{example_id}`, no acceptance step)."""
    from autosupport.graph.build import compiled_graph
    from autosupport.graph.state import InputState, TicketInput

    ticket_id = _new_ticket_id()
    thread_id = thread_id or f"{customer_id}:{ticket_id}"
    input_state: InputState = {
        "ticket_id": ticket_id,
        "customer_id": customer_id,
        "ticket": TicketInput(subject=subject, body=body),
    }
    compiled_graph().invoke(input_state, _run_config(thread_id, ticket_id, customer_id, require_acceptance))
    return _outcome(ticket_id, thread_id)


def resume_ticket(
    ticket_id: str, answer: str | None = None, accept: bool = False, reject: str | None = None
) -> TicketOutcome:
    """Resumes a paused ticket from its checkpoint. Validated at this boundary: the ticket
    must exist and be `awaiting_user`, and the flag given must match what it is waiting for
    (an answer for a clarification; accept or reject for a confirmation)."""
    from langgraph.types import Command

    from autosupport.graph.build import compiled_graph
    from autosupport.store import cases as cases_repo
    from autosupport.store import db as store_db

    conn = store_db.connect()
    try:
        row = cases_repo.get(conn, ticket_id)
        if row is None:
            raise ValueError(f"no such ticket: {ticket_id}")
        if row["status"] != "awaiting_user":
            raise ValueError(f"{ticket_id} is not waiting for input (status: {row['status']})")
        thread_id = row["thread_id"]
        pending = _outcome(ticket_id, thread_id).interrupt
        if pending is None:
            raise ValueError(f"{ticket_id} has no pending interrupt in its checkpoint")

        if pending.type == "clarification":
            if not answer:
                raise ValueError(f"{ticket_id} is waiting for an answer: use --answer")
            resume = {"answer": answer}
        else:
            if accept == (reject is not None):
                raise ValueError(f"{ticket_id} is waiting for a decision: use exactly one of --accept / --reject")
            resume = {"accepted": True} if accept else {"accepted": False, "feedback": reject}

        # Flip the row back here, not inside the interrupt node — interrupt nodes have no
        # side effects (graph-design.md §7.3).
        cases_repo.set_status(conn, ticket_id, "investigating", pending_question=None)
    finally:
        conn.close()

    compiled_graph().invoke(Command(resume=resume), _run_config(thread_id, ticket_id, row["customer_id"]))
    return _outcome(ticket_id, thread_id)


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
    interrupt = _outcome(ticket_id, row["thread_id"]).interrupt if row["status"] == "awaiting_user" else None
    return TicketOutcome(
        ticket_id=row["ticket_id"], status=row["status"], result=result,
        pending_question=row["pending_question"], interrupt=interrupt,
    )


class MemoryView(BaseModel):
    """What `load_memory` would give this customer's next ticket (memory-design.md §3)."""

    customer_id: str
    profile: CustomerMemory | None
    history: list[CaseSummary]


def memory(customer_id: str) -> MemoryView:
    from autosupport.store import cases as cases_repo
    from autosupport.store import customers as customers_repo
    from autosupport.store import db as store_db

    conn = store_db.connect()
    try:
        profile = customers_repo.get(conn, customer_id)
        history = cases_repo.history_for(conn, customer_id, exclude_ticket_id="")
    finally:
        conn.close()
    return MemoryView(customer_id=customer_id, profile=profile, history=history)


class EvalRow(BaseModel):
    example_id: str
    ticket_id: str | None
    outcome: str
    scores: dict[str, float | None]
    comments: dict[str, str]


class EvalSummary(BaseModel):
    """One offline LangSmith experiment (evaluation-design.md) — not the in-graph `verify`."""

    run_id: str
    experiment_name: str
    dataset: str
    n_examples: int
    means: dict[str, float | None]
    rows: list[EvalRow]


def run_eval(dataset: str | None = None) -> EvalSummary:
    from autosupport.config import settings
    from evals import run as eval_run
    from evals.dataset import DEFAULT_DATASET

    if settings.langsmith_api_key is None:  # fail fast, before any ticket runs (CLAUDE.md config rule)
        raise ValueError("LANGSMITH_API_KEY is empty in .env — `autosupport eval` uploads to LangSmith")

    raw = eval_run.run(dataset or DEFAULT_DATASET)
    rows = [EvalRow(**r) for r in raw["rows"]]
    keys = sorted({k for r in rows for k in r.scores})
    means: dict[str, float | None] = {}
    for key in keys:
        scored = [r.scores[key] for r in rows if r.scores.get(key) is not None]
        means[key] = sum(scored) / len(scored) if scored else None
    return EvalSummary(run_id=raw["run_id"], experiment_name=raw["experiment_name"], dataset=raw["dataset"],
                       n_examples=raw["n_examples"], means=means, rows=rows)


def list_cases(customer_id: str | None = None, awaiting: bool = False) -> list[CaseListItem]:
    from autosupport.store import cases as cases_repo
    from autosupport.store import db as store_db

    conn = store_db.connect()
    try:
        rows = cases_repo.list_cases(conn, customer_id, awaiting)
    finally:
        conn.close()
    return [CaseListItem(**{k: r[k] for k in r.keys()}) for r in rows]
