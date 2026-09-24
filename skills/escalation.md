# Escalation skill

You are preparing information for a human agent who will pick up this ticket next. They have
not read the ticket yet — everything they need has to be in what you write.

**`target_queue`** — the team that should actually own this, which may differ from the
ticket's classified queue (e.g. a billing dispute that needs a refund goes to Billing and
Payments even if it was filed as General Inquiry).

**`reason`** — one or two sentences: why this needs a human rather than a grounded fix. Name
the specific blocker (no attested resolution in the historical evidence, an action the agent
can't take like a refund or account change, a policy/priority rule) rather than a vague "this
is complex."

**`handoff_summary`** — what's known, what was tried or ruled out, and what's still missing,
citing case IDs for anything drawn from historical evidence. Write it so the human doesn't
have to re-read the ticket and the investigation from scratch.

**Mid-investigation flag vs. a real escalation.** `escalate_ticket` (a tool available during
investigation) is the model noting early that this looks headed for escalation — it doesn't
decide the outcome by itself. The actual decision, and this skill's real use, is in the
`escalate` node once the evidence is in.
