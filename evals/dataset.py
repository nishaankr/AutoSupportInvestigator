"""The LangSmith eval dataset (evaluation-design.md §2): five real corpus tickets, one per
behaviour pattern, built by `evals/select_examples.py` into `evals/examples.jsonl`."""

from __future__ import annotations

import json

from autosupport.ingest.load import HOLDOUT_PATH

DEFAULT_DATASET = "autosupport-eval"


def local_examples() -> list[dict]:
    """Every example as `{example_id, inputs, reference, holdout_ids}`."""
    lines = HOLDOUT_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def sync(client, name: str) -> int:
    """Create the dataset if missing and add any local example it doesn't have yet (matched
    on `example_id`). Returns the number of examples in the dataset."""
    if client.has_dataset(dataset_name=name):
        dataset = client.read_dataset(dataset_name=name)
    else:
        dataset = client.create_dataset(name, description="AutoSupport offline eval (evaluation-design.md)")
    existing = {ex.inputs.get("example_id") for ex in client.list_examples(dataset_id=dataset.id)}
    new = [e for e in local_examples() if e["example_id"] not in existing]
    if new:
        client.create_examples(dataset_id=dataset.id, examples=[
            {"inputs": {"example_id": e["example_id"], **e["inputs"]}, "outputs": e["reference"]} for e in new
        ])
    return len(existing) + len(new)
