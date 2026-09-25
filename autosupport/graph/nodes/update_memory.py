"""Node 16: `update_memory` — decides what's worth remembering about this customer.

The model only proposes a `MemoryUpdate`; `graph/memory.apply_update` applies the write policy
(W1–W6, memory-design.md §4) in code, and only what survives is stored. A regex check runs
first, so the model is called only when the customer said something the policy could keep.
It runs for every outcome, alongside `index_case`. A failed extraction is recorded in
`errors`, which has a reducer, and never fails the ticket: the case is already saved.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from autosupport.graph.memory import (
    REPEAT_UNRESOLVED_DAYS, MemoryUpdate, apply_update, customer_text, has_memory_candidates,
)
from autosupport.graph.state import AgentState
from autosupport.llm import fast_llm, structured
from autosupport.store import cases as cases_repo
from autosupport.store import customers as customers_repo
from autosupport.store import db as store_db

SYSTEM = """You maintain a support customer's long-term profile. From ONE finished ticket, propose \
only what would change how we handle this customer's NEXT ticket, on a different subject.

Keep (and nothing else):
- facts, keys: product, os, software_version, plan, deployment, integration — the customer's \
environment as they stated it.
- preferences, keys: contact_channel, technical_level, language — how they want to be helped.
- tried_fixes — a fix the customer says they already tried, or a fix we proposed that the \
customer rejected as not working.

Discard: this incident's specifics (error codes/messages, timestamps, order/invoice/ticket \
numbers), the fix that worked (it is stored elsewhere), one-off circumstances (travel, \
deadlines), anything you'd have to infer, and any password, key, token, e-mail address or \
phone/card/account number.

Every item needs `quote`: the exact words from the CUSTOMER'S OWN TEXT below that state it. \
Items without a verbatim customer quote are discarded automatically. An empty update is a \
correct answer when nothing qualifies. `reasoning`: one line per item on why it qualifies."""


def update_memory(state: AgentState) -> dict:
    result = state["final_output"]
    source = customer_text(state)
    conn = store_db.connect()
    try:
        profile = customers_repo.get(conn, state["customer_id"])
        outcome = (
            f"Outcome: {result.status}. Resolution/reply sent: {result.resolution}"
            + (f"\nEscalated because: {result.escalation.reason}" if result.escalation.required else "")
        )
        current = profile.model_dump(exclude={"provenance", "customer_id"}) if profile else "(empty)"
        errors: list[str] = []
        update = MemoryUpdate(reasoning="skipped: no candidate facts in the customer's text")
        if has_memory_candidates(source):
            try:
                update = structured(fast_llm(), MemoryUpdate).invoke([
                    ("system", SYSTEM),
                    ("user", f"CUSTOMER'S OWN TEXT:\n{source}\n\n{outcome}\n\nCurrent profile: {current}"),
                ])
            except Exception as exc:  # external API boundary
                # The case is already persisted; losing one memory extraction must not fail
                # the ticket. Flags (W6) are still recomputed below.
                errors.append(f"update_memory: extraction failed, profile facts unchanged: {exc}"[:300])
        since = datetime.now(timezone.utc) - timedelta(days=REPEAT_UNRESOLVED_DAYS)
        memory = apply_update(
            profile, state["customer_id"], update, source, state["ticket_id"],
            recent_escalations=cases_repo.escalations_since(conn, state["customer_id"], since),
        )
        customers_repo.upsert(conn, memory)
    finally:
        conn.close()
    # `errors` has an `operator.add` reducer, so writing it alongside `index_case` is safe.
    return {"errors": errors} if errors else {}
