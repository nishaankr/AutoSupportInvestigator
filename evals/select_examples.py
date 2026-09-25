"""Builds `evals/examples.jsonl`: the five eval examples, one per behaviour pattern
(evaluation-design.md §2), from real tickets in the corpus. Run it, check the printed
evidence, commit the output:

    python -m evals.select_examples

Every ticket used as input is one that retrieval can't return as itself: a cluster *member*
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

SELECTION = [
    {
        "example_id": "clean_resolution", "ticket": "HF-9322",
        "why": "Member of the MongoDB-integration cluster HF-2958; 8 of its 9 relevant neighbours are "
               "resolution-class, the strongest consistent resolution neighbourhood in the index.",
        "known_good": ["HF-2958"], "allowed_tools": READ_ONLY_TOOLS, "expect": {"outcome": "resolved"},
    },
    {
        "example_id": "clarification_needed", "ticket": "HF-49750",
        "why": "A 12-word report ('unusual decrease in engagement metrics') with no product, metric or "
               "timeframe; 9 of its 10 relevant neighbours were answered with clarification requests.",
        "known_good": ["HF-12458"], "allowed_tools": READ_ONLY_TOOLS, "expect": {"outcome": "asked_clarification"},
    },
    {
        "example_id": "conflicting_evidence", "ticket": "HF-61377",
        "why": "'Which analytics tools integrate with Evernote for investment optimisation' — its relevant "
               "resolution neighbours solve the same-looking request differently: HF-43855 names concrete "
               "tools (Tableau, Power BI, Python), HF-50033 (Xero) defers to a call-back, HF-54771 lists "
               "Evernote-specific tools. The corpus has no sharper contradictions; this is the largest "
               "measured divergence between near-identical requests' fixes.",
        "known_good": ["HF-54771", "HF-50033", "HF-43855"], "allowed_tools": ALL_TOOLS,
        "expect": {"verdict": "conflicting"},
    },
    {
        "example_id": "escalation_worthy", "ticket": "HF-7725",
        "why": "High priority, tagged Outage and Disruption: a financial firm's tool integrations failing "
               "after an upgrade, already rebooted and caches cleared. The historical answer escalated.",
        "known_good": ["HF-8283"], "allowed_tools": ALL_TOOLS, "expect": {"outcome": "escalated"},
    },
    {
        "example_id": "memory_and_new_case", "ticket": "HF-59267", "first_ticket": "HF-3745",
        "why": "First ticket HF-3745 ('options for integrating MongoDB 4.4 into a scalable SaaS project "
               "management platform') sits on the HF-2958 neighbourhood that resolves reliably, and 'MongoDB "
               "4.4' is a fact the memory policy keeps; it is resolved and accepted. Second ticket HF-59267, "
               "same customer, asks for documentation on the same integration (cosine 0.920 to the first) and "
               "is not in the index; it must load the remembered facts and retrieve the first ticket as "
               "agent_resolved. (A first pick, HF-22907 on Docker Django security, escalated in the dry run: "
               "the draft added security detail no cited case contained and verify rejected it twice.)",
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
            first = _row(spec["first_ticket"], conn, corpus)
            example["inputs"]["first_ticket"] = {"subject": first["subject"], "body": first["body"]}
            example["holdout_ids"].append(spec["first_ticket"])
        examples.append(example)
    conn.close()
    return examples


if __name__ == "__main__":
    examples = build()
    HOLDOUT_PATH.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in examples), encoding="utf-8")
    print(f"wrote {len(examples)} examples to {HOLDOUT_PATH}")
