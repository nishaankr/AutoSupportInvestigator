"""Builds `evals/examples.jsonl`: the five eval examples, one per behaviour pattern, from real tickets in the corpus. Run it, check the printed
evidence, commit the output:

    python -m evals.select_examples

Every corpus ticket used as input is one that retrieval can't return as itself: a cluster *member*
(merged into a canonical at ingest, so never searchable) or a row outside the index. Otherwise
the agent would retrieve the ticket's own historical answer at similarity 1.0. The member's
canonical, or the named neighbours, are the known-good evidence the retrieval evaluator checks.

The labels (queue/type/priority/tags) are the dataset's own. The rest of each reference — the
known-good ids, allowed tools and expected behaviour — is the pattern's definition, stated
below with the reason each ticket fits it.
"""

from __future__ import annotations

import json
from collections import Counter

from autosupport.ingest.load import HOLDOUT_PATH, load_english_subset
from autosupport.ingest.text import ticket_query_text
from autosupport.rag.queries import search
from autosupport.store import db as store_db

READ_ONLY_TOOLS = ["search_similar_tickets", "get_ticket_by_id", "get_customer_history", "compute_queue_stats"]
ALL_TOOLS = [*READ_ONLY_TOOLS, "escalate_ticket"]

# The memory example's setup ticket; see its "why" for the reason it isn't from the corpus.
MEMORY_FIRST_TICKET = {
    "subject": "MongoDB 4.4 integration for our project management platform",
    "body": "We're on the enterprise plan and run our services on AWS with MongoDB 4.4. We want to integrate "
            "MongoDB 4.4 with your SaaS project management platform. What integration options are available "
            "and where is the documentation? We prefer email updates.",
}

SELECTION = [
    {
        "example_id": "clean_resolution", "ticket": "HF-9322",
        "why": "Member of the MongoDB-integration cluster HF-2958. On the full index 5 of its 10 relevant "
               "neighbours are resolution-class and agree on the fix; the rest are escalated incident-style "
               "tickets, which propose no rival fix.",
        "known_good": ["HF-2958"], "allowed_tools": READ_ONLY_TOOLS, "expect": {"outcome": "resolved"},
    },
    {
        "example_id": "clarification_needed", "ticket": "HF-49750",
        "why": "A 12-word report ('unusual decrease in engagement metrics') with no product, metric or "
               "timeframe; on the full index 6 of its 10 relevant neighbours were answered with clarification "
               "requests and none with a fix.",
        "known_good": ["HF-12458"], "allowed_tools": READ_ONLY_TOOLS, "expect": {"outcome": "asked_clarification"},
    },
    {
        "example_id": "conflicting_evidence", "ticket": "HF-61377",
        "why": "'Which analytics tools integrate with Evernote for investment optimisation' — on the full index "
               "its relevant resolution neighbours solve the same-looking request differently: HF-54514 lists "
               "Evernote-specific tools, HF-54480 names concrete tools (Excel, Power BI), HF-50616 defers to "
               "'we'll send a list and help set it up'. The corpus has no sharper contradictions; this is the "
               "largest measured divergence between near-identical requests' fixes.",
        "known_good": ["HF-54514", "HF-54480", "HF-50616"], "allowed_tools": ALL_TOOLS,
        "expect": {"verdict": "conflicting"},
    },
    {
        "example_id": "escalation_worthy", "ticket": "HF-7725",
        "why": "High priority, tagged Outage and Disruption: a financial firm's tool integrations failing "
               "after an upgrade, already rebooted and caches cleared. The historical answer escalated.",
        "known_good": ["HF-8283"], "allowed_tools": ALL_TOOLS, "expect": {"outcome": "escalated"},
    },
    {
        "example_id": "memory_and_new_case", "ticket": "HF-3745", "first_ticket": MEMORY_FIRST_TICKET,
        "why": "The first ticket is authored: the corpus almost never has a customer state their own "
               "environment (4 of 11,868 cluster members do, none in a resolution neighbourhood), and corpus "
               "tickets that only *ask about* MongoDB 4.4 were correctly not remembered as facts. It states plan, "
               "deployment, version and a contact preference, on the HF-2958 MongoDB-integration resolution "
               "neighbourhood; it is resolved and accepted (3/3 on the full index). The second ticket, HF-3745, "
               "is real and held out: same customer, integration options for MongoDB 4.4 in the same platform. "
               "It must load the remembered facts and retrieve the first ticket as agent_resolved (similarity "
               "0.90, 3/3).",
        "known_good": ["$FIRST_TICKET", "HF-2958"], "allowed_tools": READ_ONLY_TOOLS,
        "expect": {"memory_loaded": True, "first_ticket_retrieved": True},
    },
]


def _row(case_id: str, conn, corpus) -> dict:
    row = conn.execute("SELECT * FROM dataset_tickets WHERE case_id = ?", (case_id,)).fetchone()
    if row is None:  # a row outside the index (the memory example's second ticket)
        row = corpus[corpus.case_id == case_id].iloc[0]
    return {k: row[k] for k in ("case_id", "subject", "body", "queue", "type", "priority", "answer")} | {
        "tags": [row[f"tag_{i}"] for i in range(1, 9) if row[f"tag_{i}"] not in (None, "") and row[f"tag_{i}"] == row[f"tag_{i}"]]}


def build() -> list[dict]:
    conn = store_db.connect()
    corpus = load_english_subset()
    examples = []
    for spec in SELECTION:
        ticket = _row(spec["ticket"], conn, corpus)
        hits = search(ticket_query_text(ticket["subject"], ticket["body"]), k=10, conn=conn)
        relevant = [h for h in hits if h.similarity >= 0.76]
        print(f"{spec['example_id']:<22} {spec['ticket']}: relevant={len(relevant)} "
              f"classes={dict(Counter(h.answer_class for h in relevant))} "
              f"known-good retrieved={[k for k in spec['known_good'] if any(h.case_id == k for h in hits)]}")
        example = {
            "example_id": spec["example_id"],
            "inputs": {"subject": ticket["subject"], "body": ticket["body"]},
            "reference": {
                "ticket": spec["ticket"], "why": spec["why"],
                "queue": ticket["queue"], "type": ticket["type"], "priority": ticket["priority"], "tags": ticket["tags"],
                "known_good": spec["known_good"], "allowed_tools": spec["allowed_tools"], "expect": spec["expect"],
                "answer": ticket["answer"],
            },
            "holdout_ids": [spec["ticket"]],
        }
        if "first_ticket" in spec:
            example["inputs"]["first_ticket"] = spec["first_ticket"]
        examples.append(example)
    conn.close()
    return examples


if __name__ == "__main__":
    examples = build()
    HOLDOUT_PATH.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in examples), encoding="utf-8")
    print(f"wrote {len(examples)} examples to {HOLDOUT_PATH}")
