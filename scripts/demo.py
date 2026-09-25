"""Scripted demo of the agent's four core scenarios, in one run:

1. Normal resolution         — a ticket is investigated, resolved, accepted and indexed.
2. Clarification + resume     — a thin ticket makes the agent ask; the answer resumes the graph.
3. Long-term memory reuse     — the first customer's next ticket loads what the first one taught.
4. Newly resolved case reuse  — a *different* customer's ticket retrieves scenario 1's case as
                                `source="agent_resolved"` evidence.

Run with `autosupport demo` or `python scripts/demo.py`. Every run uses fresh customer IDs
(`DEMO-<run>-A/B/C`), so runs never see each other's memory. It goes only through
`autosupport.service`, exactly like the CLI.

The tickets below are written for this corpus; nothing else depends on their wording.
Each scenario reports what actually happened. The agent's decisions vary from run to run, so a
scenario can honestly come out unmet. On the full index, seven consecutive runs demonstrated
all four; one of them needed scenario 1's built-in second attempt.
"""

from __future__ import annotations

import secrets

from autosupport import service
from autosupport.service import DemoReport, DemoScenario

TICKETS = {
    # Validated on the full index.
    "resolve": (
        "API integration options for our project management platform",
        "We're on the enterprise plan and run our services on AWS with Node.js 18. We want to "
        "integrate our internal tools with your project management SaaS through its API. What "
        "integration options are available and where is the documentation? We prefer email updates.",
    ),
    "thin": ("App stopped syncing", "My app stopped syncing since yesterday. What do I do?"),
    "thin_answer": (
        "It's our project management web app's calendar sync with Google Calendar. Since yesterday "
        "new tasks no longer show up in Google Calendar and nothing is shown in the app."
    ),
    "follow_up": (
        "API integration for our project management platform: syncing tasks",
        "Following up on our API integration with your project management platform: what is the "
        "recommended way to keep our task data in sync through the integration, and which "
        "documentation covers it?",
    ),
    "other_customer": (
        "API integration options for a project management SaaS",
        "What API integration options does your project management SaaS platform support, and where "
        "can our developers find the documentation?",
    ),
}


def _outcome_line(outcome: service.TicketOutcome) -> str:
    if outcome.interrupt and outcome.interrupt.type == "confirmation":
        conf = outcome.interrupt.confidence or {}
        return f"{outcome.ticket_id}: proposed resolution, confidence {conf.get('value')} ({conf.get('band')})"
    if outcome.interrupt:
        return f"{outcome.ticket_id}: paused to ask — {outcome.interrupt.question}"
    if outcome.result and outcome.result.escalation.required:
        return f"{outcome.ticket_id}: escalated — {outcome.result.escalation.reason}"
    return f"{outcome.ticket_id}: {outcome.status}"


def _top_cases(trace: service.TicketTrace, n: int = 5) -> list[str]:
    return [f"  [{c.case_id}] {c.source:<15} sim={c.similarity:.3f}  {c.subject[:60]}" for c in trace.retrieved[:n]]


def _attempt_resolution(customer: str) -> tuple[list[str], str | None]:
    subject, body = TICKETS["resolve"]
    outcome = service.new_ticket(customer, subject, body)
    lines = [f"Customer {customer} submits: {subject!r}", _outcome_line(outcome)]
    trace = service.ticket_trace(outcome.ticket_id)
    lines += ["Retrieved historical cases (top 5):", *_top_cases(trace),
              f"Tools the agent chose to call: {trace.tools_called or 'none'}"]
    if not (outcome.interrupt and outcome.interrupt.type == "confirmation"):
        return lines, None

    lines += ["Proposed resolution:", *[f"  {line}" for line in outcome.interrupt.resolution.splitlines() if line]]
    accepted = service.resume_ticket(outcome.ticket_id, accept=True)
    trace = service.ticket_trace(outcome.ticket_id)
    lines += [f"Customer accepts -> status {accepted.status}; indexed for future retrieval: {trace.indexed}"]
    return lines, outcome.ticket_id if accepted.status == "resolved" and trace.indexed else None


def scenario_resolution(customer: str) -> tuple[DemoScenario, str | None, str]:
    """Scenarios 3 and 4 build on this one, so an escalation (the evidence check rejecting a
    draft, about one run in four on this ticket) gets one more try from a fresh customer. Both
    attempts are shown. Returns the scenario, the accepted ticket and the customer who got it."""
    lines, ticket = _attempt_resolution(customer)
    if ticket is None:
        customer = f"{customer}2"
        retry, ticket = _attempt_resolution(customer)
        lines += ["", "Attempt 1 did not end in a proposed resolution; the agent would rather escalate "
                      "than send an unsupported answer. Attempt 2, same ticket, fresh customer:", *retry]
    return DemoScenario(title="1. Normal resolution", passed=ticket is not None, lines=lines), ticket, customer


def scenario_clarification(customer: str) -> DemoScenario:
    subject, body = TICKETS["thin"]
    outcome = service.new_ticket(customer, subject, body)
    lines = [f"Customer {customer} submits a thin ticket: {subject!r} — {body!r}", _outcome_line(outcome)]
    asked = bool(outcome.interrupt and outcome.interrupt.type == "clarification")
    answers = [TICKETS["thin_answer"], "That's all the detail I have."]
    while outcome.interrupt and outcome.interrupt.type == "clarification" and answers:
        answer = answers.pop(0)
        lines.append(f"Graph paused (status {outcome.status}). Customer answers: {answer!r}")
        outcome = service.resume_ticket(outcome.ticket_id, answer=answer)
        lines.append(f"Resumed -> {_outcome_line(outcome)}")
    if outcome.interrupt and outcome.interrupt.type == "confirmation":
        outcome = service.resume_ticket(outcome.ticket_id, accept=True)
        lines.append(f"Customer accepts -> status {outcome.status}")
    reached = outcome.status in ("resolved", "escalated")
    return DemoScenario(title="2. Clarification + resume", passed=asked and reached, lines=lines)


def scenario_memory(customer: str, first_ticket: str | None) -> DemoScenario:
    title = "3. Long-term memory reuse"
    if first_ticket is None:
        return DemoScenario(title=title, passed=False, lines=["Skipped: scenario 1 produced no accepted resolution."])
    remembered = service.memory(customer).profile
    lines = [f"What the agent remembered about {customer} after scenario 1:",
             f"  facts={remembered.facts if remembered else {}}",
             f"  preferences={remembered.preferences if remembered else {}}"]
    subject, body = TICKETS["follow_up"]
    outcome = service.new_ticket(customer, subject, body)
    trace = service.ticket_trace(outcome.ticket_id)
    loaded = trace.customer_profile
    lines += [f"Same customer, new conversation: {subject!r}", _outcome_line(outcome),
              f"Loaded into the new ticket's state: facts={loaded.facts if loaded else {}}",
              f"  earlier tickets: {[h.case_id for h in trace.customer_history]}"]
    text = (outcome.interrupt.resolution if outcome.interrupt and outcome.interrupt.resolution
            else outcome.result.resolution if outcome.result else "") or ""
    # Only details the new ticket doesn't mention itself count as memory being used.
    remembered_values = [*(loaded.facts.values() if loaded else []), *(loaded.preferences.values() if loaded else [])]
    ticket_text = f"{subject} {body}".lower()
    used = [v for v in remembered_values if v and v.lower() in text.lower() and v.lower() not in ticket_text]
    lines.append(f"Remembered details used in the reply (not stated in this ticket): {used or 'none this run'}")
    passed = bool(loaded and (loaded.facts or loaded.preferences)) and first_ticket in [h.case_id for h in trace.customer_history]
    lines.append("(Left as-is for inspection: `autosupport show " + outcome.ticket_id + "`.)")
    return DemoScenario(title=title, passed=passed, lines=lines)


def scenario_new_case_retrieval(customer: str, first_ticket: str | None) -> DemoScenario:
    title = "4. Newly resolved case retrieval"
    if first_ticket is None:
        return DemoScenario(title=title, passed=False, lines=["Skipped: scenario 1 produced no accepted resolution."])
    subject, body = TICKETS["other_customer"]
    outcome = service.new_ticket(customer, subject, body)
    trace = service.ticket_trace(outcome.ticket_id)
    hit = next((c for c in trace.retrieved if c.case_id == first_ticket), None)
    lines = [f"A different customer, {customer} (no shared memory), submits: {subject!r}", _outcome_line(outcome),
             "Retrieved cases (top 5):", *_top_cases(trace),
             f"Scenario 1's case {first_ticket}: "
             + (f"retrieved as source={hit.source}, similarity {hit.similarity:.3f}, "
                f"cited as evidence: {first_ticket in trace.evidence_ids}" if hit else "not retrieved"),
             f"Customer profile loaded for {customer}: {trace.customer_profile}"]
    passed = bool(hit and hit.source == "agent_resolved")
    return DemoScenario(title=title, passed=passed, lines=lines)


def run() -> DemoReport:
    run_id = secrets.token_hex(3)
    first, second, third = (f"DEMO-{run_id}-{x}" for x in "ABC")
    s1, first_ticket, first = scenario_resolution(first)
    scenarios = [
        s1,
        scenario_clarification(second),
        scenario_memory(first, first_ticket),
        scenario_new_case_retrieval(third, first_ticket),
    ]
    if first_ticket:
        # Leave the index as we found it: agent resolutions aren't de-duplicated, so every run
        # adding the same answer again would slowly crowd out the next run's scenario 4.
        service.unindex_case(first_ticket)
        scenarios[-1].lines.append(f"(Cleanup: {first_ticket} removed from the index again; its case and memory remain.)")
    return DemoReport(run_id=run_id, scenarios=scenarios)


if __name__ == "__main__":
    from autosupport.cli import app

    app(["demo"])  # same run and rendering as `autosupport demo`
