# Escalation skill

You are preparing information for a human agent who will pick up this ticket next. They have
not read the ticket yet — everything they need has to be in what you write. You are also
writing the customer's holding reply.

**`target_queue`** — the team that should actually own this, which may differ from the
ticket's classified queue (e.g. a billing dispute that needs a refund goes to Billing and
Payments even if it was filed as General Inquiry). Use one of the known queue names.

**`reason`** — one or two sentences: why this needs a human rather than a grounded fix. Name
the specific blocker (no attested resolution in the historical evidence, an action the agent
can't take like a refund or account change, a policy/priority rule, the customer rejecting the
proposed fix) rather than a vague "this is complex."

**`handoff_summary`** — what's known, what was tried or ruled out, and what's still missing,
citing case IDs as `[case_id]` for anything drawn from historical evidence. Write it so the
human doesn't have to re-read the ticket and the investigation from scratch.

**`analysis`** — for the reviewing support agent: the hypothesis and how the evidence did or
didn't support it. Cite only cases you were shown; if you were shown none, cite none. Every case marked `contradicts` in the evidence must be cited in `analysis` (with why it cuts against the hypothesis) — conflicts are acknowledged, never dropped.

**`resolution`** — the customer-facing holding reply: acknowledge the problem, say plainly
that it is being passed to a person, name what happens next, and repeat any specific question
still open. Never include a fix the evidence doesn't attest.

**Mid-investigation flag vs. a real escalation.** `escalate_ticket` (a tool available during
investigation) is the investigator noting early that this looks headed for escalation — it
doesn't decide the outcome by itself. This skill is used by the `escalate` node, once the
evidence check has decided.
