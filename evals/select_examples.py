"""Draws the held-out corpus examples for the eval dataset (evaluation-design.md §2) and
writes `evals/examples.jsonl`. Run once, output committed:

    python -m evals.select_examples

Held out = in the English subset but not in `dataset_tickets`, and with no indexed
near-duplicate (nearest neighbour cosine < the clustering threshold). `ingest/load.py` then
excludes these ids from every future ingest.
"""

from __future__ import annotations

import json

import pandas as pd

from autosupport.ingest.classify import classify_answer
from autosupport.ingest.load import HOLDOUT_PATH, load_english_subset
from autosupport.rag import dense
from autosupport.rag.embedder import embed
from autosupport.store import db as store_db

PER_CLASS = 5
CLASSES = ("resolution", "escalation", "clarification_request")
NEAR_DUPLICATE = 0.92  # clustering threshold T, rag-design.md §4
SEED = 7


def _candidates() -> pd.DataFrame:
    df = load_english_subset()
    conn = store_db.connect()
    try:
        indexed = {r[0] for r in conn.execute("SELECT case_id FROM dataset_tickets").fetchall()}
    finally:
        conn.close()
    df = df[~df["case_id"].isin(indexed) & ~df["below_content_threshold"] & (df["subject_ix"] != "")]
    df = df[df["body_ix"].str.len().between(120, 700)]
    df = df.sample(frac=1.0, random_state=SEED)
    df["answer_class"] = df["answer"].map(classify_answer)
    return df[df["answer_class"].isin(CLASSES)]


def _is_near_duplicate(embed_text: str) -> bool:
    hits = dense.search(embed([embed_text])[0], n=1)
    return bool(hits) and hits[0].similarity >= NEAR_DUPLICATE


def select() -> list[dict]:
    df = _candidates()
    chosen: list[dict] = []
    for answer_class in CLASSES:
        pool = df[df["answer_class"] == answer_class]
        queues_used: set[str] = set()
        picked = 0
        # Two passes: first prefer a queue not yet used for this class, then fill.
        for prefer_new_queue in (True, False):
            for _, row in pool.iterrows():
                if picked == PER_CLASS:
                    break
                if row["case_id"] in {c["example_id"] for c in chosen}:
                    continue
                if prefer_new_queue and row["queue"] in queues_used:
                    continue
                if _is_near_duplicate(row["embed_text"]):
                    continue
                chosen.append({
                    "example_id": row["case_id"],
                    "inputs": {"subject": row["subject"], "body": row["body"]},
                    "reference": {
                        "queue": row["queue"], "type": row["type"], "priority": row["priority"],
                        "answer_class": answer_class, "answer": row["answer"],
                    },
                })
                queues_used.add(row["queue"])
                picked += 1
    return chosen


if __name__ == "__main__":
    examples = select()
    HOLDOUT_PATH.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in examples), encoding="utf-8")
    print(f"wrote {len(examples)} examples to {HOLDOUT_PATH}")
