# Escalation skill

This ticket looks like it may need a human (the historical answers for it were mostly
escalations, or it touches security, data loss, an outage or a legal matter). Investigate it
so that, if it is escalated, the person who picks it up doesn't have to start over: the handoff
note is assembled directly from what you submit.

**Decide early whether a human is needed.** If the fix needs something you can't do — a
refund, an account or billing change, access to the customer's systems, a policy exception —
say exactly what in `requires_human_action`, and call `escalate_ticket` with that reason as
soon as you're sure. Don't keep searching past that point just to fill out evidence.

**Make the findings a usable handoff.**
- `hypothesis`: what is wrong and what a person should do next, not "needs escalation".
- `evidence`: cite the cases that show how this was handled before, with honest stances.
  Every case that cuts against your hypothesis is marked `contradicts` — conflicts are
  acknowledged, never dropped.
- `missing_slots`: the specific facts the human will need from the customer, if any.

**Never promise a fix the evidence doesn't attest.** If no historical case resolved this,
say so in the hypothesis: that is the most useful thing the human can know.
