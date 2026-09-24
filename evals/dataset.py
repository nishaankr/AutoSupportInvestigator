"""The LangSmith eval dataset (evaluation-design.md §2): held-out corpus examples with their
gold labels, plus the brief's example tickets once `brief_examples.json` exists."""

from __future__ import annotations

import json
from pathlib import Path

from autosupport.ingest.load import HOLDOUT_PATH

DEFAULT_DATASET = "autosupport-eval"
BRIEF_PATH = Path(__file__).parent / "brief_examples.json"


def local_examples() -> list[dict]:
    """Every example as `{example_id, inputs, reference}`. Brief examples have no gold labels
    (`reference` is `{}`), so the label-based evaluators skip them."""
    lines = HOLDOUT_PATH.read_text(encoding="utf-8").splitlines()
    examples = [json.loads(line) for line in lines if line.strip()]
    if BRIEF_PATH.exists():
        for i, ticket in enumerate(json.loads(BRIEF_PATH.read_text(encoding="utf-8")), start=1):
            examples.append({"example_id": f"BRIEF-{i}",
                             "inputs": {"subject": ticket["subject"], "body": ticket["body"]}, "reference": {}})
    return examples


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
