"""Typer commands — rendering only. Every command will call service.py, the only
thing this module is allowed to call once it exists (CLAUDE.md). No business logic
lives here, and this module must never import graph/, rag/, store/ or ingest/
directly. It must not import config or service at module level either, so that
`autosupport --help` works with no `.env` present."""

from __future__ import annotations

from typing import Optional

import typer

app = typer.Typer(name="autosupport", help="Autonomous Support Investigation Agent.", no_args_is_help=True)


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
    raise NotImplementedError


@app.command()
def resume(
    ticket_id: str = typer.Argument(..., help="Ticket ID to resume."),
    answer: Optional[str] = typer.Option(None, "--answer", help="Answer a pending clarification question."),
    accept: bool = typer.Option(False, "--accept", help="Accept the proposed resolution."),
    reject: Optional[str] = typer.Option(None, "--reject", help="Reject the proposed resolution, with feedback."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Resume a paused ticket: answer a clarification, or accept/reject a resolution."""
    raise NotImplementedError


@app.command()
def show(
    ticket_id: str = typer.Argument(..., help="Ticket ID to show."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Print the structured CaseResult and status for a ticket."""
    raise NotImplementedError


@app.command("list")
def list_cases(
    customer: Optional[str] = typer.Option(None, "--customer", help="Filter by customer ID."),
    awaiting: bool = typer.Option(False, "--awaiting", help="Only tickets awaiting a user answer."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """List cases."""
    raise NotImplementedError


@app.command()
def memory(
    customer_id: str = typer.Argument(..., help="Customer ID."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Print the stored long-term memory for a customer."""
    raise NotImplementedError


@app.command("eval")
def run_eval(
    dataset: Optional[str] = typer.Option(None, "--dataset", help="LangSmith dataset name."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Run the LangSmith evaluation suite."""
    raise NotImplementedError


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
