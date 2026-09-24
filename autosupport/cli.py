"""Typer commands — rendering only. Every command will call service.py, the only
thing this module is allowed to call once it exists (CLAUDE.md). No business logic
lives here, and this module must never import graph/, rag/, store/ or ingest/
directly. It must not import config or service at module level either, so that
`autosupport --help` works with no `.env` present."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Optional

import typer

if TYPE_CHECKING:
    from autosupport.service import TicketOutcome

app = typer.Typer(name="autosupport", help="Autonomous Support Investigation Agent.", no_args_is_help=True)


@app.callback()
def _utf8_output() -> None:
    """Model text routinely contains characters (arrows, dashes, curly quotes) that a Windows
    console/redirect encoding (cp1252) can't encode, which crashed `--json > file` after the
    graph had already finished. Output is always UTF-8."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _render_interrupt(ticket_id: str, pending) -> None:
    if pending.type == "clarification":
        typer.echo(f'\n[PAUSED] {ticket_id} needs clarification: "{pending.question}"')
        typer.echo(f'   autosupport resume {ticket_id} --answer "..."')
        return
    typer.echo(f"\n[PAUSED] {ticket_id} proposed resolution (confidence {pending.confidence['value']} "
               f"{pending.confidence['band']}; cites {pending.cited_case_ids}):")
    typer.echo(pending.resolution)
    typer.echo(f'\n   autosupport resume {ticket_id} --accept   |   --reject "feedback"')


def _render_outcome(outcome: "TicketOutcome") -> None:
    """Shared by `new` and `show` — both render the same `TicketOutcome` shape."""
    typer.echo(f"ticket_id: {outcome.ticket_id}")
    typer.echo(f"status:    {outcome.status}")
    if outcome.interrupt is not None:
        _render_interrupt(outcome.ticket_id, outcome.interrupt)
    if outcome.result is None:
        typer.echo("(no result yet)")
        return

    result = outcome.result
    c = result.classification
    typer.echo(f"classification: queue={c.queue} type={c.type} priority={c.priority} tags={c.tags}")
    typer.echo(f"rationale: {c.rationale}")
    typer.echo("")
    typer.echo("evidence:")
    for e in result.evidence:
        typer.echo(f"  [{e.case_id}] {e.stance:<11} {e.answer_class or '-':<21} cluster={e.cluster_size} {e.summary}")
    typer.echo("")
    typer.echo(f"analysis:\n{result.analysis}")
    typer.echo("")
    typer.echo(f"resolution:\n{result.resolution}")
    if result.escalation.required:
        typer.echo("")
        typer.echo(
            f"escalation: target_queue={result.escalation.target_queue} trigger={result.escalation.trigger}\n"
            f"reason: {result.escalation.reason}\nhandoff: {result.escalation.handoff_summary}"
        )
    typer.echo("")
    cf = result.confidence
    typer.echo(f"confidence: {cf.value} ({cf.band})  support={cf.support} agreement={cf.agreement} "
               f"relevance={cf.relevance} penalty={cf.penalty} cap={cf.cap_reason}")
    typer.echo(f"verification: passed={result.verification.passed} attempts={result.verification.attempts}"
               + (f" unresolved={result.verification.unresolved_issues}" if result.verification.unresolved_issues else ""))
    typer.echo(f"acceptance: {result.acceptance}")
    typer.echo(f"stats: {result.stats.model_dump()}")
    if result.errors:
        typer.echo(f"errors: {result.errors}")


@app.command()
def ingest(
    limit: Optional[int] = typer.Option(None, "--limit", help="Limit the number of dataset rows ingested."),
    rebuild: bool = typer.Option(False, "--rebuild", help="Rebuild the index from scratch."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Download the HF dataset, filter to English, load, embed and index it."""
    from autosupport import service

    result = service.ingest(limit=limit, rebuild=rebuild)
    if json_output:
        typer.echo(result.model_dump_json())
        return
    typer.echo(f"rows loaded:              {result.rows_loaded}")
    typer.echo(f"below content threshold:  {result.rows_below_content_threshold}")
    typer.echo(f"canonicals indexed:       {result.canonicals_indexed}")
    typer.echo(f"answer_class:             {result.answer_class_distribution}")
    typer.echo(f"answer_class source:      {result.answer_class_source_distribution}")
    typer.echo(f"cluster size:             {result.cluster_size_distribution}")
    typer.echo(f"duration:                 {result.duration_seconds:.1f}s")


@app.command()
def new(
    customer: str = typer.Option(..., "--customer", help="Customer ID."),
    subject: str = typer.Option(..., "--subject", help="Ticket subject."),
    body: str = typer.Option(..., "--body", help="Ticket body."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Start a new ticket and a new thread."""
    from autosupport import service

    outcome = service.new_ticket(customer_id=customer, subject=subject, body=body)
    if json_output:
        typer.echo(outcome.model_dump_json())
        return
    _render_outcome(outcome)


@app.command()
def resume(
    ticket_id: str = typer.Argument(..., help="Ticket ID to resume."),
    answer: Optional[str] = typer.Option(None, "--answer", help="Answer a pending clarification question."),
    accept: bool = typer.Option(False, "--accept", help="Accept the proposed resolution."),
    reject: Optional[str] = typer.Option(None, "--reject", help="Reject the proposed resolution, with feedback."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Resume a paused ticket: answer a clarification, or accept/reject a resolution."""
    from autosupport import service

    try:
        outcome = service.resume_ticket(ticket_id, answer=answer, accept=accept, reject=reject)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(outcome.model_dump_json())
        return
    _render_outcome(outcome)


@app.command()
def show(
    ticket_id: str = typer.Argument(..., help="Ticket ID to show."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Print the structured CaseResult and status for a ticket."""
    from autosupport import service

    try:
        outcome = service.show(ticket_id)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(outcome.model_dump_json())
        return
    _render_outcome(outcome)


@app.command("list")
def list_cases(
    customer: Optional[str] = typer.Option(None, "--customer", help="Filter by customer ID."),
    awaiting: bool = typer.Option(False, "--awaiting", help="Only tickets awaiting a user answer."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """List cases."""
    from autosupport import service

    items = service.list_cases(customer_id=customer, awaiting=awaiting)
    if json_output:
        typer.echo("[" + ", ".join(i.model_dump_json() for i in items) + "]")
        return
    if not items:
        typer.echo("no cases")
        return
    for i in items:
        typer.echo(f"{i.ticket_id}  {i.customer_id:<8} {i.status:<14} {i.subject[:50]}")
        if i.pending_question:
            typer.echo(f"    [PAUSED] {i.pending_question}")


@app.command()
def memory(
    customer_id: str = typer.Argument(..., help="Customer ID."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Print the stored long-term memory for a customer."""
    from autosupport import service

    view = service.memory(customer_id)
    if json_output:
        typer.echo(view.model_dump_json())
        return
    profile = view.profile
    if profile is None:
        typer.echo(f"no stored memory for {customer_id}")
    else:
        for kind in ("facts", "preferences"):
            typer.echo(f"{kind}:")
            for key, value in getattr(profile, kind).items():
                typer.echo(f"  {key}: {value}  (from {profile.provenance.get(f'{kind}.{key}', '?')})")
        typer.echo("tried_fixes:")
        for fix in profile.tried_fixes:
            typer.echo(f"  - {fix}")
        typer.echo(f"flags: {', '.join(profile.flags) or '(none)'}")
    typer.echo("history:")
    for h in view.history:
        typer.echo(f"  {h.case_id}  {h.status:<14} {h.subject[:60]}")


@app.command("eval")
def run_eval(
    dataset: Optional[str] = typer.Option(None, "--dataset", help="LangSmith dataset name."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Run the offline LangSmith evaluation (not the in-graph verify step)."""
    from autosupport import service

    try:
        summary = service.run_eval(dataset)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(summary.model_dump_json())
        return
    typer.echo(f"experiment {summary.experiment_name}  (dataset {summary.dataset}, {summary.n_examples} examples)")
    for key, mean in summary.means.items():
        typer.echo(f"  {key:<28} {'-' if mean is None else f'{mean:.2f}'}")
    main_keys = ["response_groundedness", "retrieval_relevance", "tool_usage_correctness",
                 "classification_accuracy", "outcome_appropriateness"]
    typer.echo("\nexample     outcome               " + "  ".join(k[:10] for k in main_keys))
    for row in summary.rows:
        cells = "  ".join(f"{'-' if row.scores.get(k) is None else f'{row.scores[k]:.2f}':>10}" for k in main_keys)
        typer.echo(f"{row.example_id:<11} {row.outcome:<21} {cells}")


@app.command()
def demo(
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Scripted run of the four required demo scenarios end-to-end."""
    raise NotImplementedError


@app.command()
def search(
    query: str = typer.Argument(..., help="Query text."),
    k: int = typer.Option(10, "--k", help="Number of results."),
    queue: Optional[str] = typer.Option(None, "--queue", help="Filter to one queue (dense arm only)."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """TEMPORARY (CP2, docs/project/checkpoints.md) — prints fused hybrid-retrieval results
    with each arm's rank, the fused RRF score, similarity, cluster_size and answer_class.
    Removed once the graph nodes that call rag/queries.py directly exist."""
    from autosupport import service

    hits = service.search(query, k=k, queue=queue)
    if json_output:
        typer.echo("[" + ", ".join(h.model_dump_json() for h in hits) + "]")
        return
    if not hits:
        typer.echo("no results")
        return
    header = f"{'rank':<4} {'dense':<6} {'lex':<4} {'rrf':<8} {'sim':<6} {'cluster':<7} {'class':<21} case_id  subject"
    typer.echo(header)
    for i, h in enumerate(hits, start=1):
        dense_r = str(h.dense_rank) if h.dense_rank else "-"
        lex_r = str(h.lexical_rank) if h.lexical_rank else "-"
        typer.echo(
            f"{i:<4} {dense_r:<6} {lex_r:<4} {h.score:<8.4f} {h.similarity:<6.3f} "
            f"{h.cluster_size:<7} {h.answer_class:<21} {h.case_id:<9} {h.subject[:60]}"
        )


if __name__ == "__main__":
    app()
