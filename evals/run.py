"""One offline eval experiment (evaluation-design.md §3): sync the dataset, run every
example through the real graph with `require_acceptance=false` on thread
`eval:{run_id}:{example_id}`, score with `evals/evaluators.py`, and summarise."""

from __future__ import annotations

from datetime import datetime, timezone

from autosupport import service
from autosupport.config import settings
from autosupport.graph.build import compiled_graph
from evals import dataset, evaluators

TOP_RETRIEVED = 10


def _outcome(status: str, interrupt) -> str:
    if status == "awaiting_user" and interrupt and interrupt.type == "clarification":
        return "asked_clarification"
    return status


def _snapshot(ticket: service.TicketOutcome, thread_id: str) -> dict:
    """The compact view of the final graph state the evaluators score."""
    values = compiled_graph().get_state({"configurable": {"thread_id": thread_id}}).values
    classification = values.get("classification")
    result = values.get("final_output")
    retrieved = sorted(values.get("retrieved_cases", []), key=lambda c: c.similarity, reverse=True)
    return {
        "ticket_id": ticket.ticket_id,
        "thread_id": thread_id,
        "outcome": _outcome(ticket.status, ticket.interrupt),
        "classification": classification.model_dump(include={"queue", "type", "priority"}) if classification else None,
        "retrieved": [
            c.model_dump(include={"case_id", "source", "similarity", "subject", "answer_class",
                                  "body_snippet", "answer_snippet", "retrieval_round", "query_label"})
            for c in retrieved[:TOP_RETRIEVED]
        ],
        "tool_log": [t.model_dump(include={"name", "args", "ok", "round"}) for t in values.get("tool_log", [])],
        "response": {"resolution": result.resolution, "analysis": result.analysis} if result else None,
        "confidence": result.confidence.value if result else None,
        "pending_question": ticket.pending_question,
    }


def _target(run_id: str):
    def target(inputs: dict) -> dict:
        example_id = inputs["example_id"]
        thread_id = f"eval:{run_id}:{example_id}"
        # A fresh customer per example and run: no memory carries between eval tickets.
        ticket = service.new_ticket(
            f"EVAL-{run_id}-{example_id}", inputs["subject"], inputs["body"],
            thread_id=thread_id, require_acceptance=False,
        )
        return _snapshot(ticket, thread_id)
    return target


def run(dataset_name: str = dataset.DEFAULT_DATASET) -> dict:
    from langsmith import Client, evaluate

    client = Client()
    n_examples = dataset.sync(client, dataset_name)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    results = evaluate(
        _target(run_id),
        data=dataset_name,
        evaluators=evaluators.ALL,
        experiment_prefix=f"autosupport-{run_id}",
        metadata={"run_id": run_id, "require_acceptance": False, "main_model": settings.main_model,
                  "fast_model": settings.fast_model, "tau_rel": settings.tau_rel},
        max_concurrency=1,  # one SQLite checkpointer connection and store (evaluation-design.md §3)
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
