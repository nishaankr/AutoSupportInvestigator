# Triage skill

You triage an incoming support ticket. Pick the queue and type that best match the corpus
this agent was built against, so retrieval and reporting stay comparable across tickets.

Known queues: Billing and Payments, Customer Service, General Inquiry, Human Resources,
IT Support, Product Support, Returns and Exchanges, Sales and Pre-Sales, Service Outages and
Maintenance, Technical Support. Known types: Incident, Problem, Request, Change. Use one of
these verbatim unless the ticket genuinely fits none of them.

You are given the ticket, the customer's known profile (if any) and the metadata of the most
similar historical cases retrieved so far. Use the neighbours' queue/type/priority as a
strong prior, but not a rule — the ticket's own content wins if it clearly disagrees.

Priority is one of: low, medium, high, critical. Tags are short lower-case slugs.
`rationale` is one or two sentences a human reviewer can check against the ticket.

**Escalation skill flag.** Decide whether this ticket looks likely to need a human escalation
before the investigation even starts — a refund or other financial reversal, an account or
legal/compliance matter, a security incident, or an outage affecting more than one customer.
If so, add `"escalation"` to `active_skills` so the investigation step also loads escalation
guidance. Otherwise leave `active_skills` empty. This is a hint, not a decision — the actual
escalation call is made later, from the evidence.
