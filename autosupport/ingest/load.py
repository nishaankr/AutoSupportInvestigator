"""HF dataset -> English filter -> exact dedup -> local parquet snapshot
(architecture.md §4.1.1; decisions.md D6; rag-design.md §1).

`HF-<row>` uses the row's position in the *unfiltered* HF `train` split (61,765 rows), not
its position after filtering — that position is stable across re-ingests regardless of
future filtering changes, which a post-filter index wouldn't be (decisions.md D1).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from autosupport.config import settings
from autosupport.ingest.text import MIN_CONTENT_CHARS, derive_title, embed_text, index_text, index_body

DATASET_NAME = "Tobi-Bueck/customer-support-tickets"
# Eval holdout (evaluation-design.md §2): these tickets must never be indexed, or the agent
# would retrieve an eval ticket's own historical answer.
HOLDOUT_PATH = Path(__file__).resolve().parents[2] / "evals" / "examples.jsonl"


def _holdout_ids() -> set[str]:
    if not HOLDOUT_PATH.exists():
        return set()
    lines = HOLDOUT_PATH.read_text(encoding="utf-8").splitlines()
    return {case_id for line in lines if line.strip() for case_id in json.loads(line)["holdout_ids"]}


def _snapshot_path() -> Path:
    return settings.data_dir / "dataset_tickets.parquet"


def _load_and_prepare() -> pd.DataFrame:
    from datasets import load_dataset

    raw = load_dataset(DATASET_NAME, split="train").to_pandas()
    raw["hf_row"] = raw.index  # position in the unfiltered split — the deterministic ID source
    for col in ("subject", "body", "answer"):
        raw[col] = raw[col].fillna("").astype(str)

    english = raw[raw["language"] == "en"].copy()
    english["subject_ix"] = english["subject"].map(index_text)
    english["body_ix"] = english["body"].map(index_body)
    english["answer_ix"] = english["answer"].map(index_text)
    english["version_key"] = english["version"].fillna(-1).astype(int)
    english["dedup_key"] = (
        english["subject_ix"] + "|" + english["body_ix"] + "|" + english["answer_ix"]
    ).str.lower()

    # Exact duplicates are a merge artifact of two dataset generations (rag-design.md §1):
    # keep the labelled-version copy, drop its unlabelled (version=None) twin.
    deduped = (
        english.sort_values("version_key", ascending=False, kind="stable")
        .drop_duplicates("dedup_key")
        .sort_values("hf_row")
        .reset_index(drop=True)
    )

    deduped["case_id"] = "HF-" + deduped["hf_row"].astype(str)
    deduped["source"] = "dataset"
    deduped["embed_text"] = [
        embed_text(s, b) for s, b in zip(deduped["subject_ix"], deduped["body_ix"])
    ]
    deduped["title_is_derived"] = deduped["subject_ix"] == ""
    deduped["title"] = [
        derive_title(b) if derived else s
        for s, b, derived in zip(deduped["subject"], deduped["body_ix"], deduped["title_is_derived"])
    ]
    deduped["below_content_threshold"] = deduped["embed_text"].str.len() < MIN_CONTENT_CHARS
    return deduped


def load_english_subset(limit: int | None = None, rebuild: bool = False) -> pd.DataFrame:
    """Returns the deduplicated English subset, one row per distinct historical ticket.
    Cached to a local parquet snapshot so repeated `--limit` runs during development don't
    re-download or re-normalise the full corpus. `rebuild=True` forces a fresh pull."""
    path = _snapshot_path()
    if rebuild or not path.exists():
        df = _load_and_prepare()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
    else:
        df = pd.read_parquet(path)

    if limit:
        df = df.sample(min(limit, len(df)), random_state=0).sort_values("hf_row").reset_index(drop=True)
    return df


def load_ingest_subset(limit: int | None = None, rebuild: bool = False) -> pd.DataFrame:
    """`load_english_subset` minus the eval holdout. Filtered *after* the `--limit` sample,
    so adding a holdout never changes which rows a given `--limit` ingests."""
    df = load_english_subset(limit=limit, rebuild=rebuild)
    return df[~df["case_id"].isin(_holdout_ids())].reset_index(drop=True)
