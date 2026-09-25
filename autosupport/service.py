"""Everything a user interface can do, as plain functions returning Pydantic objects.

The CLI only renders what these return, and it never talks to the graph or the stores
directly. That keeps a future web frontend a matter of calling the same functions."""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from typing import Any, Callable, Literal

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


class PendingInterrupt(BaseModel):
    """What a paused ticket is waiting for — the `interrupt()` payload of `ask_user` or
    `confirm_resolution`, read back from the checkpoint."""

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


class StepEvent(BaseModel):
    """One graph node finishing, as it happens: the node's name and the facts about what it
    just did, lifted from its state update. The CLI decides how to word them."""

    node: str
    facts: dict[str, Any] = {}


def _case_facts(cases: list, tau_rel: float) -> dict:
    return {
        "n": len(cases),
        "relevant": sum(c.similarity >= tau_rel for c in cases),
        "top": max((c.similarity for c in cases), default=0.0),
        "agent_resolved": [(c.case_id, round(c.similarity, 3)) for c in cases if c.source == "agent_resolved"],
    }


def _step_facts(node: str, update: dict, ticket_id: str, tau_rel: float) -> dict:
    """What each node just did, read from its update (or, for the two post-persist nodes that
    write only to SQLite, from the database). Read-only: observing a run must not change it."""
    if node == "intake":
        return {"ticket_id": ticket_id, "thread_id": update.get("thread_id")}
    if node == "load_memory":
        profile, history = update.get("customer_profile"), update.get("customer_history", [])
        return {
            "facts": profile.facts if profile else {}, "preferences": profile.preferences if profile else {},
            "tried_fixes": profile.tried_fixes if profile else [], "flags": profile.flags if profile else [],
            "history": [(h.case_id, h.status) for h in history],
        }
    if node in ("retrieve_initial", "retrieve_variant"):
        queries = update.get("retrieval_queries", [])
        return {**_case_facts(update.get("retrieved_cases", []), tau_rel),
                "label": queries[0].label if queries else None, "round": queries[0].round if queries else None,
                "failed": update.get("errors", [])}
    if node == "refine_retrieval":
        return {"round": update.get("retrieval_round")}
    if node == "triage":
        c = update["classification"]
        return {"queue": c.queue, "type": c.type, "priority": c.priority, "tags": c.tags,
                "skills": update.get("active_skills", [])}
    if node == "investigate":
        findings = update.get("findings")
        if findings is None:
            calls = [m for m in update.get("messages", []) if getattr(m, "tool_calls", None)]
            return {"tool_requests": [tc["name"] for tc in calls[-1].tool_calls] if calls else []}
        evidence = update.get("evidence", [])
        return {"submitted": True, "supporting": [e.case_id for e in evidence if e.stance == "supports"],
                "evidence": len(evidence), "missing": findings.missing_slots}
    if node == "tools":
        return {"calls": [(t.name, t.ok, next((t.args[k] for k in ("query", "case_id", "queue", "reason")
                                               if t.args.get(k)), ""))
                          for t in update.get("tool_log", [])],
                **_case_facts(update.get("retrieved_cases", []), tau_rel)}
    if node == "assess_evidence":
        a = update["evidence_assessment"]
        return {"verdict": a.verdict, "next": a.next_action, "why": a.reason, "relevant": a.relevant_count,
                "rule": a.escalation_rule_hit}
    if node == "ask_user":
        turns = update.get("clarifications", [])
        return {"answer": turns[-1].answer if turns else None}
    if node == "resolve":
        from autosupport.graph.state import CITATION

        return {"cites": list(dict.fromkeys(CITATION.findall(update["draft"].resolution)))}
    if node == "escalate":
        draft = update["draft"]
        return {"trigger": update.get("escalation_trigger"),
                "queue": draft.escalation.target_queue if draft.escalation else None}
    if node == "verify":
        v, cf = update["verification"], update["confidence"]
        return {"passed": v.passed, "attempt": update.get("verify_attempts"),
                "issues": [*v.issues, *v.unsupported_claims], "confidence": cf.value, "band": cf.band,
                "awaiting": update.get("status") == "awaiting_user"}
    if node == "confirm_resolution":
        return {"acceptance": update.get("user_acceptance"), "feedback": update.get("user_feedback")}
    if node == "persist_case":
        return {"status": update.get("status")}
    if node in ("index_case", "update_memory"):
        from autosupport.store import cases as cases_repo
        from autosupport.store import customers as customers_repo
        from autosupport.store import db as store_db

        conn = store_db.connect()
        try:
            row = cases_repo.get(conn, ticket_id)
            if node == "index_case":
                return {"indexed": row["indexed_at"] is not None, "status": row["status"]}
            profile = customers_repo.get(conn, row["customer_id"])
            return {"facts": profile.facts if profile else {}, "preferences": profile.preferences if profile else {},
                    "tried_fixes": profile.tried_fixes if profile else [], "errors": update.get("errors", [])}
        finally:
            conn.close()
    return {}


def _drive(graph_input, config: dict, ticket_id: str, on_step: Callable[[StepEvent], None] | None) -> None:
    """Run the graph. With `on_step`, stream it instead and report each node as it finishes —
    the same execution, observed; without, a plain `invoke` exactly as before."""
    from autosupport.graph.build import compiled_graph

    graph = compiled_graph()
    if on_step is None:
        graph.invoke(graph_input, config)
        return
    tau_rel = config["configurable"]["tau_rel"]
    for chunk in graph.stream(graph_input, config, stream_mode="updates"):
        for node, update in chunk.items():
            if node == "__interrupt__":
                continue  # the pause itself is read from the checkpoint afterwards
            update = update if isinstance(update, dict) else {}
            on_step(StepEvent(node=node, facts=_step_facts(node, update, ticket_id, tau_rel)))


class CaseListItem(BaseModel):
    ticket_id: str
    customer_id: str
    status: CaseStatus
    subject: str
    pending_question: str | None
    updated_at: str


def _new_ticket_id() -> str:
    """`T-YYYYMMDD-<6 hex>`. Made here rather than in `intake`, because the graph can't be
    invoked without a thread id, and the thread id contains the ticket id."""
    return f"T-{datetime.now(timezone.utc):%Y%m%d}-{secrets.token_hex(3)}"


def _run_config(
    thread_id: str, ticket_id: str, customer_id: str, require_acceptance: bool | None = None
) -> dict:
    """The per-run settings. Loop limits travel in `configurable`, so a resume runs under the
    same limits the ticket started with; `metadata` stamps every LangSmith trace with the
    ticket's identity, so any trace or eval score can be traced back to a ticket id."""
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
    """Read from the checkpoint, since only the checkpoint knows whether a run is paused."""
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
    on_step: Callable[[StepEvent], None] | None = None,
) -> TicketOutcome:
    """Start a ticket and run it until it finishes or pauses. The offline eval is the only
    caller that overrides `thread_id` (to `eval:{run_id}:{example_id}`) and switches the
    acceptance step off."""
    from autosupport.graph.state import InputState, TicketInput

    ticket_id = _new_ticket_id()
    thread_id = thread_id or f"{customer_id}:{ticket_id}"
    input_state: InputState = {
        "ticket_id": ticket_id,
        "customer_id": customer_id,
        "ticket": TicketInput(subject=subject, body=body),
    }
    _drive(input_state, _run_config(thread_id, ticket_id, customer_id, require_acceptance), ticket_id, on_step)
    return _outcome(ticket_id, thread_id)


def resume_ticket(
    ticket_id: str, answer: str | None = None, accept: bool = False, reject: str | None = None,
    on_step: Callable[[StepEvent], None] | None = None,
) -> TicketOutcome:
    """Continue a paused ticket from its checkpoint — possibly in a brand-new process. The
    input is checked here, at the edge: the ticket must exist and be waiting, and the caller
    must give what it's waiting for (an answer to a question, or accept/reject)."""
    from langgraph.types import Command

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

        # The status flips back here rather than in the interrupt node: LangGraph re-runs an
        # interrupt node from the top on resume, so it must not touch the database.
        cases_repo.set_status(conn, ticket_id, "investigating", pending_question=None)
    finally:
        conn.close()

    _drive(Command(resume=resume), _run_config(thread_id, ticket_id, row["customer_id"]), ticket_id, on_step)
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
    """What this customer's next ticket would start with: their profile and earlier tickets."""

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


class TraceCase(BaseModel):
    case_id: str
    source: str
    similarity: float
    query_label: str
    subject: str


class TicketTrace(BaseModel):
    """What one ticket's run retrieved, loaded and did — read from its checkpoint. Used by the
    demo to show retrieval and memory at work."""

    ticket_id: str
    customer_id: str
    indexed: bool
    retrieved: list[TraceCase]
    customer_profile: CustomerMemory | None
    customer_history: list[CaseSummary]
    tools_called: list[str]
    evidence_ids: list[str]


def ticket_trace(ticket_id: str) -> TicketTrace:
    from autosupport.graph.build import compiled_graph
    from autosupport.store import cases as cases_repo
    from autosupport.store import db as store_db

    conn = store_db.connect()
    try:
        row = cases_repo.get(conn, ticket_id)
    finally:
        conn.close()
    if row is None:
        raise ValueError(f"no such ticket: {ticket_id}")
    values = compiled_graph().get_state({"configurable": {"thread_id": row["thread_id"]}}).values
    return TicketTrace(
        ticket_id=ticket_id, customer_id=row["customer_id"], indexed=row["indexed_at"] is not None,
        retrieved=[TraceCase(case_id=c.case_id, source=c.source, similarity=round(c.similarity, 3),
                             query_label=c.query_label, subject=c.subject)
                   for c in values.get("retrieved_cases", [])],
        customer_profile=values.get("customer_profile"),
        customer_history=values.get("customer_history", []),
        tools_called=[t.name for t in values.get("tool_log", [])],
        evidence_ids=[e.case_id for e in values.get("evidence", [])],
    )


def unindex_case(ticket_id: str) -> None:
    """Take an accepted agent resolution back out of the search index (its `cases` row and the
    customer's memory stay). Used by the demo and the eval so repeated runs don't pile up
    near-identical agent cases, which aren't de-duplicated and crowd each other out."""
    from autosupport.ingest.agent_index import unindex_agent_case
    from autosupport.store import db as store_db

    conn = store_db.connect()
    try:
        unindex_agent_case(conn, ticket_id)
    finally:
        conn.close()


class DemoScenario(BaseModel):
    title: str
    passed: bool
    lines: list[str]


class DemoReport(BaseModel):
    run_id: str
    scenarios: list[DemoScenario]


def run_demo() -> DemoReport:
    from scripts.demo import run

    return run()


class EvalRow(BaseModel):
    example_id: str
    ticket_id: str | None
    outcome: str
    scores: dict[str, float | None]
    comments: dict[str, str]


class EvalSummary(BaseModel):
    """One offline LangSmith experiment. Not to be confused with `verify`, which checks a
    single draft inside the graph; this scores the whole system afterwards."""

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

    if settings.langsmith_api_key is None:  # say so now, not after fifteen tickets have run
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
