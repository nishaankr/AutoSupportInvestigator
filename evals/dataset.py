"""The LangSmith eval dataset: five real corpus tickets, one per
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
    """Make the LangSmith dataset match `examples.jsonl`: create it if missing, add new
    examples, and update any whose inputs or reference changed (matched on `example_id`).
    Adding alone once left a stale reference in place after the examples were re-validated on
    the full index, and the eval scored against it. Returns the example count."""
    if client.has_dataset(dataset_name=name):
        dataset = client.read_dataset(dataset_name=name)
    else:
        dataset = client.create_dataset(name, description="AutoSupport offline eval: one example per behaviour pattern")
    remote = {ex.inputs.get("example_id"): ex for ex in client.list_examples(dataset_id=dataset.id)}
    new = []
    for e in local_examples():
        inputs = {"example_id": e["example_id"], **e["inputs"]}
        current = remote.get(e["example_id"])
        if current is None:
            new.append({"inputs": inputs, "outputs": e["reference"]})
        elif current.inputs != inputs or current.outputs != e["reference"]:
            client.update_example(example_id=current.id, inputs=inputs, outputs=e["reference"])
    if new:
        client.create_examples(dataset_id=dataset.id, examples=new)
    return len(remote) + len(new)
