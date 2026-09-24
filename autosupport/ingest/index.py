"""Ingest orchestration: load -> classify -> cluster -> SQLite write -> FTS5 build ->
embed canonicals -> Chroma upsert (architecture.md §4.1.2-4.1.6). The one entry point,
`run`, is what `autosupport ingest` calls (via service.py — CLAUDE.md's UI-seam rule)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from autosupport.config import settings
from autosupport.ingest import agent_index, classify, cluster, load
from autosupport.rag import dense
from autosupport.rag.embedder import embed
from autosupport.store import cases as cases_repo
from autosupport.store import dataset_tickets, db

_TAG_COLUMNS = [f"tag_{i}" for i in range(1, 9)]
COLLECTION_NAME = "support_cases"


@dataclass
class IngestReport:
    rows_loaded: int
    rows_after_dedup: int
    rows_below_content_threshold: int
    canonicals_indexed: int
    answer_class_distribution: dict[str, int]
    answer_class_source_distribution: dict[str, int]
    cluster_size_distribution: dict[str, int]
    duration_seconds: float
    sampled_precision: dict[str, float] = field(default_factory=dict)


def _classify_all(records: pd.DataFrame) -> pd.DataFrame:
    from autosupport.llm import fast_llm, structured

    records = records.copy()
    records["answer_class"] = records["answer"].map(classify.classify_answer)
    records["answer_class_source"] = "heuristic"

    residue = records.index[records["answer_class"] == "residue"]
    if len(residue):
        # method="json_schema": decisions.md D13 (the default is unreliable for these models)
        llm = structured(fast_llm(), classify.ResidueClassification)
        for idx in residue:
            result = classify.classify_residue(records.at[idx, "answer"], llm)
            records.at[idx, "answer_class"] = result.answer_class
            records.at[idx, "answer_class_source"] = "llm"
    return records


def _cluster_all(records: pd.DataFrame) -> pd.DataFrame:
    records = records.copy()
    embed_vectors = embed(records["embed_text"].tolist())
    answer_vectors = embed(records["answer_ix"].tolist())
    eligible = ~records["below_content_threshold"].to_numpy()

    result = cluster.cluster(records, embed_vectors, answer_vectors, eligible)
    leader = result.leader
    case_ids = records["case_id"].to_numpy()
    cluster_size = pd.Series(leader).value_counts()

    records["is_canonical"] = result.canonical_mask
    records["canonical_of"] = [
        case_ids[leader[i]] if (leader[i] != i and eligible[i]) else None for i in range(len(records))
    ]
    records["cluster_size"] = [
        int(cluster_size[i]) if result.canonical_mask[i] else None for i in range(len(records))
    ]
    records.attrs["embed_vectors"] = embed_vectors
    return records


def _chroma_metadata(row: pd.Series) -> dict:
    metadata = {
        "source": row["source"],
        "queue": row["queue"],
        "type": row["type"],
        "priority": row["priority"],
        "answer_class": row["answer_class"],
        "cluster_size": int(row["cluster_size"] or 1),
        "version": int(row["version"]) if pd.notna(row["version"]) else -1,
    }
    tags = [row[c] for c in _TAG_COLUMNS if pd.notna(row[c]) and row[c] != ""]
    return {**metadata, **dense.tag_metadata(tags)}


def _upsert_chroma(records: pd.DataFrame, embed_vectors: np.ndarray, rebuild: bool) -> list[str]:
    import chromadb

    canonical = records[records["is_canonical"]]
    if canonical.empty:
        return []

    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    if rebuild and COLLECTION_NAME in {c.name for c in client.list_collections()}:
        client.delete_collection(COLLECTION_NAME)
    # Chroma's default HNSW space is squared L2, not cosine. embed_vectors are already
    # L2-normalised (rag/embedder.py), so ranking order would come out the same either way,
    # but the *value* Chroma reports as "distance" would be the wrong number — rag-design.md
    # §9's tau_rel/SIM_CEILING and output-schema.md's confidence formula both compare against
    # true cosine similarity, not an L2 distance transform of it. Set explicitly so `1 -
    # distance` in dense.py is actually cosine similarity, not something that merely ranks
    # the same. get_or_create_collection only applies this metadata on first creation; an
    # existing collection created without it keeps its original (L2) space regardless.
    collection = client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    canonical_idx = np.where(records["is_canonical"].to_numpy())[0]
    collection.upsert(
        ids=canonical["case_id"].tolist(),
        embeddings=embed_vectors[canonical_idx].tolist(),
        documents=canonical["embed_text"].tolist(),
        metadatas=[_chroma_metadata(row) for _, row in canonical.iterrows()],
    )
    return canonical["case_id"].tolist()


def run(limit: int | None = None, rebuild: bool = False) -> IngestReport:
    start = datetime.now(timezone.utc)

    records = load.load_ingest_subset(limit=limit, rebuild=rebuild)
    rows_loaded = len(records)

    records = _classify_all(records)
    records = _cluster_all(records)
    embed_vectors = records.attrs["embed_vectors"]

    conn = db.connect(rebuild=rebuild)
    dataset_tickets.insert_all(conn, records)
    dataset_tickets.rebuild_fts(conn)
    indexed_ids = _upsert_chroma(records, embed_vectors, rebuild)
    dataset_tickets.mark_indexed(conn, indexed_ids)
    if rebuild:
        # --rebuild dropped the FTS5 table and the Chroma collection; put back what the agent
        # learned at runtime (case-persistence.md §5.3).
        for ticket_id in cases_repo.indexed_ticket_ids(conn):
            agent_index.index_agent_case(conn, ticket_id, force=True)

    report = IngestReport(
        rows_loaded=rows_loaded,
        rows_after_dedup=rows_loaded,  # load.py already deduplicates before returning
        rows_below_content_threshold=int(records["below_content_threshold"].sum()),
        canonicals_indexed=len(indexed_ids),
        answer_class_distribution=dataset_tickets.answer_class_distribution(conn),
        answer_class_source_distribution=dataset_tickets.answer_class_source_distribution(conn),
        cluster_size_distribution=dataset_tickets.cluster_size_distribution(conn),
        duration_seconds=(datetime.now(timezone.utc) - start).total_seconds(),
    )
    conn.close()
    return report
