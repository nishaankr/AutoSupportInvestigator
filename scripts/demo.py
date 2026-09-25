"""Scripted demo of the four scenarios the assignment requires (REQUIREMENTS §12 D), in one run:

1. Normal resolution         — a ticket is investigated, resolved, accepted and indexed.
2. Clarification + resume     — a thin ticket makes the agent ask; the answer resumes the graph.
3. Long-term memory reuse     — the first customer's next ticket loads what the first one taught.
4. Newly resolved case reuse  — a *different* customer's ticket retrieves scenario 1's case as
                                `source="agent_resolved"` evidence.

Run with `autosupport demo` or `python scripts/demo.py`. Every run uses fresh customer IDs
(`DEMO-<run>-A/B/C`), so runs never see each other's memory. It goes only through
`autosupport.service`, exactly like the CLI.

The tickets below are written for this corpus. The brief's own example tickets (§5) aren't
in the repo; when they are, swap them into `TICKETS` — nothing else depends on the wording.
Each scenario reports what actually happened. The agent's decisions vary from run to run, so a
scenario can honestly come out unmet; after the fixes in decisions.md D22, three consecutive
runs demonstrated all four.
"""

from __future__ import annotations

import secrets

from autosupport import service
from autosupport.service import DemoReport, DemoScenario

TICKETS = {
    "resolve": (
        "MongoDB 4.4 integration options for our SaaS project management platform",
        "We run a multi-tenant SaaS project management platform on Ubuntu 22.04 servers and want to "
        "integrate MongoDB 4.4. What integration methods do you recommend (API connections, data "
        "synchronization) and which resources should we start with? Our team prefers email updates.",
    ),
    "thin": ("App stopped syncing", "My app stopped syncing since yesterday. What do I do?"),
    "thin_answer": (
        "It's our project management web app's calendar sync with Google Calendar. Since yesterday "
        "new tasks no longer show up in Google Calendar and nothing is shown in the app."
    ),
    "follow_up": (
        "Keeping MongoDB in sync with our project data",
        "Following up on our MongoDB integration: what is the recommended way to keep project data "
        "synchronized between the platform and MongoDB, and which resources cover it?",
    ),
    "other_customer": (
        "Recommended MongoDB integration for a SaaS project management tool",
        "Which integration methods do you recommend for connecting MongoDB to a SaaS project "
        "management platform, and where should our developers start?",
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


def scenario_resolution(customer: str) -> tuple[DemoScenario, str | None]:
    subject, body = TICKETS["resolve"]
    outcome = service.new_ticket(customer, subject, body)
    lines = [f"Customer {customer} submits: {subject!r}", _outcome_line(outcome)]
    trace = service.ticket_trace(outcome.ticket_id)
    lines += ["Retrieved historical cases (top 5):", *_top_cases(trace),
              f"Tools the agent chose to call: {trace.tools_called or 'none'}"]
    if not (outcome.interrupt and outcome.interrupt.type == "confirmation"):
        return DemoScenario(title="1. Normal resolution", passed=False, lines=lines), None

    lines += ["Proposed resolution:", *[f"  {line}" for line in outcome.interrupt.resolution.splitlines() if line]]
    accepted = service.resume_ticket(outcome.ticket_id, accept=True)
    trace = service.ticket_trace(outcome.ticket_id)
    lines += [f"Customer accepts -> status {accepted.status}; indexed for future retrieval: {trace.indexed}"]
    passed = accepted.status == "resolved" and trace.indexed
    return DemoScenario(title="1. Normal resolution", passed=passed, lines=lines), outcome.ticket_id if passed else None


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
    s1, first_ticket = scenario_resolution(first)
    return DemoReport(run_id=run_id, scenarios=[
        s1,
        scenario_clarification(second),
        scenario_memory(first, first_ticket),
        scenario_new_case_retrieval(third, first_ticket),
    ])


if __name__ == "__main__":
    from autosupport.cli import app

    app(["demo"])  # same run and rendering as `autosupport demo`
