"""The `autosupport` command line. It only renders: every command calls `service.py` and
prints what comes back (or dumps it with `--json`). Nothing here imports the graph or the
stores, and even `service` is imported inside each command, so `autosupport --help` works
before a `.env` exists."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Optional

import typer

if TYPE_CHECKING:
    from autosupport.service import TicketOutcome

app = typer.Typer(name="autosupport", help="Autonomous Support Investigation Agent.", no_args_is_help=True)


@app.callback()
def _utf8_output() -> None:
    """Always write UTF-8. Model text is full of arrows, dashes and curly quotes that the
    default Windows encoding (cp1252) can't represent — it once crashed `--json > file` after
    the whole run had already finished."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


# The agent's nine behaviour steps, numbered in the order this graph runs them: retrieval
# comes before classification because triage votes over the retrieved neighbours. Nodes marked
# None report under the step already on screen, as an indented line.
_STEPS = {
    "intake": (1, "Persisting ticket as open case"),
    "load_memory": (2, "Loading customer memory + short-term state"),
    "retrieve_initial": (3, "Retrieving historical cases"),
    "refine_retrieval": (3, "Retrieving historical cases"),
    "retrieve_variant": None,
    "triage": (4, "Classifying against historical evidence"),
    "investigate": (5, "Investigating with tools + skills"),
    "tools": None,
    "assess_evidence": (6, "Checking evidence sufficiency"),
    "ask_user": None,
    "resolve": (7, "Generating resolution"),
    "escalate": (7, "Generating escalation handoff"),
    "verify": (8, "Verifying against evidence"),
    "confirm_resolution": None,
    "persist_case": (9, "Persisting final case + indexing for retrieval"),
    "index_case": None,
    "update_memory": None,
}
_NEXT = {"resolve": "proceeding to resolution", "ask_user": "asking the customer",
         "refine_retrieval": "searching again", "escalate": "escalating to a person"}
_STEP_WIDTH = 56


def _clip(text: str, n: int = 90) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _pairs(d: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in d.items())


def _agent_note(facts: dict) -> str:
    hits = facts.get("agent_resolved") or []
    return "; includes " + ", ".join(f"{cid} source=agent_resolved sim {sim:.2f}" for cid, sim in hits) if hits else ""


def _step_note(node: str, f: dict) -> str:
    """The short result note after a step: what the node just did, in plain words."""
    if node == "intake":
        return f"done ({f['ticket_id']})"
    if node == "load_memory":
        parts = []
        if f["facts"]:
            parts.append(f"remembered {_pairs(f['facts'])}")
        if f["preferences"]:
            parts.append(f"prefers {_pairs(f['preferences'])}")
        if f["tried_fixes"]:
            parts.append(f"already tried: {'; '.join(f['tried_fixes'])}")
        if f["flags"]:
            parts.append(f"flags: {', '.join(f['flags'])}")
        if f["history"]:
            parts.append("earlier tickets: " + ", ".join(f"{cid} ({status})" for cid, status in f["history"]))
        return "done (new thread; " + ("; ".join(parts) if parts else "no prior history for this customer") + ")"
    if node == "retrieve_initial":
        return f"{f['n']} cases, {f['relevant']} relevant (top sim {f['top']:.2f}){_agent_note(f)}"
    if node == "refine_retrieval":
        return f"round {f['round']}: evidence thin, running targeted searches"
    if node == "retrieve_variant":
        return (f"search '{f['label']}' failed" if f["failed"] else
                f"search '{f['label']}': {f['n']} cases, {f['relevant']} relevant{_agent_note(f)}")
    if node == "triage":
        return (f"{f['queue']} / {f['type']} / {f['priority']}; tags: {', '.join(f['tags'])}; "
                f"skills: {', '.join(['investigation', *f['skills']])}")  # investigation is always loaded
    if node == "investigate":
        if not f.get("submitted"):
            return "calling " + ", ".join(f["tool_requests"])
        note = f"findings submitted ({len(f['supporting'])} supporting cases, {f['evidence']} evidence entries)"
        return note + (f"; missing: {'; '.join(f['missing'])}" if f["missing"] else "")
    if node == "tools":
        return "; ".join(f"{name}({_clip(arg, 40)!r}) {'ok' if ok else 'FAILED'}" for name, ok, arg in f["calls"]) + (
            f" -> {f['n']} cases{_agent_note(f)}" if f["n"] else "")
    if node == "assess_evidence":
        note = f"{f['verdict']}, {_NEXT.get(f['next'], f['next'])}"
        return note + (f" (rule: {f['rule']})" if f["rule"] else f" ({f['why']})")
    if node == "ask_user":
        return f"customer answered: {_clip(f['answer'], 70)!r}"
    if node == "resolve":
        return f"done, cites {', '.join(f['cites']) or 'nothing'}"
    if node == "escalate":
        return f"done, to {f['queue']} ({f['trigger']})"
    if node == "verify":
        note = (f"supported (attempt {f['attempt']}), confidence {f['confidence']:.2f} ({f['band']})" if f["passed"]
                else f"not supported (attempt {f['attempt']}): {_clip('; '.join(f['issues']), 70)}")
        return note + ("; waiting for customer acceptance" if f["awaiting"] else "")
    if node == "confirm_resolution":
        return ("customer accepted the proposed resolution" if f["acceptance"] == "accepted"
                else f"customer rejected it: {_clip(f['feedback'], 60)!r}")
    if node == "persist_case":
        return f"saved as {f['status']}"
    if node == "index_case":
        return ("indexed for future retrieval as source=agent_resolved" if f["indexed"]
                else "not indexed (only customer-accepted resolutions are)")
    if node == "update_memory":
        kept = {**f["facts"], **f["preferences"]}
        return ("customer memory now: " + _pairs(kept) if kept else "customer memory: nothing durable to keep") + (
            " (extraction failed; profile unchanged)" if f["errors"] else "")
    return "done"


def _print_step(event) -> None:
    step = _STEPS.get(event.node)
    note = _step_note(event.node, event.facts)
    if step is None:
        typer.echo(f"      -> {note}")
        return
    n, label = step
    head = f"[{n}/9] {label} "
    typer.echo(typer.style(head + "." * max(3, _STEP_WIDTH - len(head)), bold=True) + f" {note}")


_LABEL_WIDTH = 17


def _width() -> int:
    import shutil

    return min(max(shutil.get_terminal_size((100, 24)).columns, 80), 100)


def _field(label: str, lines: list[str]) -> None:
    """One labelled field of the panel: the label once, every line wrapped under the value column."""
    import textwrap

    wrap = _width() - _LABEL_WIDTH
    out: list[str] = []
    for line in lines:
        indent = len(line) - len(line.lstrip(" "))
        out += textwrap.wrap(line, wrap, subsequent_indent=" " * indent) or [""]
    for i, line in enumerate(out):
        typer.echo((typer.style(f"{label:<{_LABEL_WIDTH}}", bold=True) if i == 0 else " " * _LABEL_WIDTH) + line)


def _rule(title: str = "", char: str = "=") -> str:
    return f" {title} ".center(_width(), char) if title else char * _width()


def _render_result(result) -> None:
    """The six result fields, each under its own label."""
    c, cf, esc = result.classification, result.confidence, result.escalation
    typer.echo("")
    typer.echo(typer.style(_rule(f"CASE RESULT  {result.ticket_id}"), bold=True))
    typer.echo(f"{result.status.upper()} for customer {result.customer_id}; acceptance: {result.acceptance}")
    typer.echo(_rule(char="-"))
    _field("Classification", [f"Queue: {c.queue} | Type: {c.type} | Priority: {c.priority}", f"Tags: {', '.join(c.tags)}"])
    evidence = []
    for e in result.evidence:
        evidence.append(f"[{e.case_id}] source={e.source} | {e.stance} | {e.answer_class or '-'} | "
                        f"cluster_size={e.cluster_size} | sim {e.similarity:.2f}")
        evidence.append(f"    {e.summary}")
    _field("Evidence", evidence or ["(no historical cases cited)"])
    _field("Analysis", [result.analysis])
    _field("Resolution", [p for p in result.resolution.splitlines() if p.strip()])
    if esc.required:
        _field("Escalation", [f"Required -> {esc.target_queue} (trigger: {esc.trigger}"
                              + (f", rule: {esc.rule}" if esc.rule else "") + ")", f"Why: {esc.reason}"])
    else:
        _field("Escalation", ["Not required"])
    _field("Confidence", [f"{cf.value:.2f} ({cf.band})", f"support {cf.support:.2f} | agreement {cf.agreement:.2f} | "
                          f"relevance {cf.relevance:.2f} | penalty {cf.penalty:.2f}"
                          + (f" | capped: {cf.cap_reason}" if cf.cap_reason else "")])
    typer.echo(_rule(char="-"))
    v = result.verification
    typer.echo(f"Evidence check: {'passed' if v.passed else 'failed'} after {v.attempts} attempt(s)"
               + (f"; unresolved: {v.unresolved_issues}" if v.unresolved_issues else ""))
    if result.errors:
        typer.echo(f"Errors recorded: {result.errors}")
    typer.echo(typer.style(_rule(), bold=True))


def _render_pause(ticket_id: str, pending, exiting: bool) -> None:
    import textwrap

    typer.echo("")
    if pending.type == "clarification":
        typer.echo(typer.style("PAUSED: the agent needs more information before it can continue.", bold=True))
        typer.echo(f"Question: {pending.question}")
        commands = [f'autosupport resume {ticket_id} --answer "<your answer>"']
    else:
        conf = pending.confidence or {}
        typer.echo(typer.style("PAUSED: proposed resolution, waiting for the customer to accept or reject it.", bold=True))
        typer.echo(f"Confidence {conf.get('value')} ({conf.get('band')}); cites {', '.join(pending.cited_case_ids) or 'nothing'}")
        typer.echo("")
        for para in (pending.resolution or "").splitlines():
            typer.echo(textwrap.fill(para, _width() - 2, initial_indent="  ", subsequent_indent="  ") if para.strip() else "")
        commands = [f"autosupport resume {ticket_id} --accept",
                    f'autosupport resume {ticket_id} --reject "<what didn\'t work>"']
    typer.echo("")
    if exiting:
        typer.echo("The graph state is checkpointed and the ticket is awaiting_user. This process exits now;")
        typer.echo("resuming continues the same thread, in a new process:")
    else:
        typer.echo("To continue:")
    for command in commands:
        typer.echo(f"  {command}")


def _run_live(start, ticket_id: str | None = None) -> "TicketOutcome":
    """Run a ticket with live step lines. If the run fails partway (usually a model provider
    or network error), say where, why, and what state the ticket is left in, instead of a
    full traceback in the middle of a demo; `--json` still raises it raw for debugging."""
    import re

    seen = {"ticket_id": ticket_id}

    def on_step(event) -> None:
        if event.node == "intake":
            seen["ticket_id"] = event.facts["ticket_id"]
        _print_step(event)

    try:
        return start(on_step)
    except Exception as exc:  # external API boundary
        notes = " ".join(getattr(exc, "__notes__", []))
        node = re.search(r"task with name '(\w+)'", notes)
        step = _STEPS.get(node.group(1)) if node else None
        where = f"[{step[0]}/9] {step[1]}" if step else (node.group(1) if node else "an unknown step")
        provider = type(exc).__module__.split(".")[0]
        typer.echo("")
        typer.echo(typer.style(f"STOPPED: {where} failed.", bold=True), err=True)
        typer.echo(f"  {type(exc).__name__} ({provider}): {_clip(exc, 300)}", err=True)
        if seen["ticket_id"]:
            typer.echo(f"  Ticket {seen['ticket_id']} was saved when it arrived and stays open; nothing was "
                       "sent to the customer.", err=True)
        typer.echo("  Fix the cause (API key, credits, network) and submit the ticket again. "
                   "Run the same command with --json for the full traceback.", err=True)
        raise typer.Exit(code=1) from None


def _render_outcome(outcome: "TicketOutcome", exiting: bool = False) -> None:
    """Shared by `new`, `resume` and `show`: the pause and what to run next, or the result panel."""
    if outcome.interrupt is not None:
        _render_pause(outcome.ticket_id, outcome.interrupt, exiting)
    elif outcome.result is not None:
        _render_result(outcome.result)
    else:
        typer.echo(f"{outcome.ticket_id}: {outcome.status} (no result yet)")


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


def _ask(label: str, json_output: bool) -> str:
    # PowerShell prefixes piped input with a byte-order mark; left in, "C-1" and "\ufeffC-1"
    # would silently be two different customers. With --json the prompt goes to stderr, so
    # stdout stays pure JSON.
    return typer.prompt(label, err=json_output).lstrip("\ufeff").strip()


@app.command()
def new(
    customer: Optional[str] = typer.Option(None, "--customer", help="Customer ID (prompted if omitted)."),
    subject: Optional[str] = typer.Option(None, "--subject", help="Ticket subject (prompted if omitted)."),
    body: Optional[str] = typer.Option(None, "--body", help="Ticket body (prompted if omitted)."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Start a new ticket and a new thread, printing each step as it runs."""
    from autosupport import service

    customer = customer or _ask("Customer ID", json_output)
    subject = subject or _ask("Subject", json_output)
    body = body or _ask("Body", json_output)
    if json_output:
        typer.echo(service.new_ticket(customer_id=customer, subject=subject, body=body).model_dump_json())
        return
    typer.echo("")
    outcome = _run_live(lambda on_step: service.new_ticket(customer_id=customer, subject=subject, body=body,
                                                           on_step=on_step))
    _render_outcome(outcome, exiting=True)


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
        if json_output:
            typer.echo(service.resume_ticket(ticket_id, answer=answer, accept=accept, reject=reject).model_dump_json())
            return
        outcome = _run_live(lambda on_step: service.resume_ticket(ticket_id, answer=answer, accept=accept,
                                                                  reject=reject, on_step=on_step), ticket_id)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    _render_outcome(outcome, exiting=True)


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
    main_keys = ["pattern_behaviour", "retrieval_relevance", "tool_usage_correctness",
                 "classification_accuracy", "response_groundedness"]
    typer.echo(f"\n{'example':<22} {'outcome':<20} " + "  ".join(k[:10] for k in main_keys))
    for row in summary.rows:
        cells = "  ".join(f"{'-' if row.scores.get(k) is None else f'{row.scores[k]:.2f}':>10}" for k in main_keys)
        typer.echo(f"{row.example_id:<22} {row.outcome:<20} {cells}")


@app.command()
def demo(
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Scripted run of the four core demo scenarios end-to-end (scripts/demo.py)."""
    from autosupport import service

    report = service.run_demo()
    if json_output:
        typer.echo(report.model_dump_json())
        return
    typer.echo(f"AutoSupport demo — run {report.run_id}")
    for scenario in report.scenarios:
        typer.echo(f"\n{'PASS' if scenario.passed else 'NOT MET'}  {scenario.title}")
        for line in scenario.lines:
            typer.echo(f"    {line}")
    met = sum(s.passed for s in report.scenarios)
    typer.echo(f"\n{met}/{len(report.scenarios)} scenarios demonstrated.")


if __name__ == "__main__":
    app()
