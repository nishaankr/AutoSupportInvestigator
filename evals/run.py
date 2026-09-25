"""One offline eval experiment: sync the dataset, run every example
through the real graph, score it with `evals/evaluators.py`, and summarise.

Tracing is off for normal use; this run switches it on for itself only. Each example is one
pipeline trace — the memory example's two tickets run inside the same target call — and the
groundedness judge adds at most one more. Evaluator tracing is disabled, so the code-only
evaluators cost nothing.
"""

from __future__ import annotations

from datetime import datetime, timezone

from autosupport import service
from autosupport.config import settings
from autosupport.graph.build import compiled_graph
from evals import dataset, evaluators

TOP_RETRIEVED = 10
FIRST_TICKET_ATTEMPTS = 3


def _outcome(status: str, interrupt) -> str:
    if status == "awaiting_user" and interrupt and interrupt.type == "clarification":
        return "asked_clarification"
    return status


def _snapshot(ticket: service.TicketOutcome, thread_id: str) -> dict:
    """The compact view of the final graph state the evaluators score."""
    config = {"configurable": {"thread_id": thread_id}}
    values = compiled_graph().get_state(config).values
    classification = values.get("classification")
    result = values.get("final_output")
    retrieved = sorted(values.get("retrieved_cases", []), key=lambda c: c.similarity, reverse=True)
    # Every verdict the evidence check reached, oldest first: a first-round "conflicting"
    # matters even if a later round settled it. Consecutive checkpoints repeat the same
    # assessment until a new one is written, so each is recorded once.
    verdicts, last = [], None
    for snap in reversed(list(compiled_graph().get_state_history(config))):
        assessment = snap.values.get("evidence_assessment")
        if assessment is not None and assessment != last:
            verdicts.append(assessment.verdict)
            last = assessment
    return {
        "ticket_id": ticket.ticket_id,
        "outcome": _outcome(ticket.status, ticket.interrupt),
        "classification": classification.model_dump(include={"queue", "type", "priority", "tags"}) if classification else None,
        "verdicts": verdicts,
        "retrieved": [c.model_dump(include={"case_id", "source", "similarity", "subject", "answer_class"})
                      for c in retrieved[:TOP_RETRIEVED]],
        "tool_log": [t.model_dump(include={"name", "args", "ok", "round"}) for t in values.get("tool_log", [])],
        "response": {"resolution": result.resolution, "analysis": result.analysis} if result else None,
        "confidence": result.confidence.value if result else None,
        "customer_profile": values["customer_profile"].model_dump(include={"facts", "preferences", "tried_fixes"})
        if values.get("customer_profile") else None,
    }


def _run_first_ticket(customer: str, first: dict, thread_id: str) -> tuple[str, str, str, int]:
    """The memory example's first ticket: run with the acceptance step on and accepted, so it is
    indexed exactly as a real customer-accepted resolution would be.

    It is the example's precondition, not what it measures, so it gets up to
    FIRST_TICKET_ATTEMPTS tries; the count is reported in the outputs. Each try is a fresh
    customer: a failed try's escalation must not count toward the real customer's
    `repeat_unresolved` flag. Returns (customer, ticket_id, status, attempts)."""
    for attempt in range(1, FIRST_TICKET_ATTEMPTS + 1):
        who = customer if attempt == 1 else f"{customer}-try{attempt}"
        outcome = service.new_ticket(who, first["subject"], first["body"],
                                     thread_id=f"{thread_id}:{attempt}", require_acceptance=True)
        if outcome.interrupt and outcome.interrupt.type == "confirmation":
            outcome = service.resume_ticket(outcome.ticket_id, accept=True)
        if outcome.status == "resolved":
            break
    return who, outcome.ticket_id, outcome.status, attempt


def _target(run_id: str):
    def target(inputs: dict) -> dict:
        example_id = inputs["example_id"]
        thread_id = f"eval:{run_id}:{example_id}"
        # A fresh customer per example and run: no memory carries between eval tickets.
        customer = f"EVAL-{run_id}-{example_id}"
        first_id = None
        if "first_ticket" in inputs:
            customer, first_id, first_status, attempts = _run_first_ticket(
                customer, inputs["first_ticket"], f"{thread_id}:first")
        ticket = service.new_ticket(customer, inputs["subject"], inputs["body"],
                                    thread_id=thread_id, require_acceptance=False)
        out = _snapshot(ticket, thread_id)
        if first_id:
            out.update({
                "first_ticket_id": first_id, "first_ticket_status": first_status, "first_ticket_attempts": attempts,
                "memory_loaded": bool(out["customer_profile"] and out["customer_profile"]["facts"]),
                "first_ticket_retrieved": any(c["case_id"] == first_id and c["source"] == "agent_resolved"
                                              for c in out["retrieved"]),
            })
            service.unindex_case(first_id)  # leave the corpus as it was found
        return out
    return target


def run(dataset_name: str = dataset.DEFAULT_DATASET) -> dict:
    from langsmith import Client, evaluate, tracing_context

    client = Client()
    n_examples = dataset.sync(client, dataset_name)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    with tracing_context(enabled=True):
        results = evaluate(
            _target(run_id),
            data=dataset_name,
            evaluators=evaluators.ALL,
            experiment_prefix=f"autosupport-{run_id}",
            metadata={"run_id": run_id, "main_model": settings.main_model, "fast_model": settings.fast_model,
                      "judge_model": settings.judge_model, "tau_rel": settings.tau_rel},
            max_concurrency=1,  # one SQLite checkpointer connection and store
            disable_evaluator_tracing=True,  # only the groundedness judge traces, by choice
            client=client,
        )
    rows = []
    for row in results:
        outputs = row["run"].outputs or {}
        rows.append({
            "example_id": row["example"].inputs["example_id"],
            "ticket_id": outputs.get("ticket_id"),
            "outcome": outputs.get("outcome", "run failed"),
            "scores": {r.key: r.score for r in row["evaluation_results"]["results"]},
            "comments": {r.key: r.comment for r in row["evaluation_results"]["results"] if r.comment},
        })
    return {"run_id": run_id, "experiment_name": results.experiment_name, "dataset": dataset_name,
            "n_examples": n_examples, "rows": rows}
