"""Chroma similarity search — the dense arm of hybrid retrieval (rag-design.md §3, §7).

Every result carries a true cosine similarity (`1 - distance`), not a ranking-only score —
the collection is created with `hnsw:space: "cosine"` explicitly (ingest/index.py), since
Chroma's default is squared L2, and `output-schema.md`'s confidence formula and
`graph-design.md`'s `tau_rel` both compare against real cosine similarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np

from autosupport.config import settings

COLLECTION_NAME = "support_cases"


@dataclass
class DenseHit:
    case_id: str
    similarity: float
    metadata: dict


@cache
def _client():
    import chromadb

    return chromadb.PersistentClient(path=str(settings.chroma_dir))


def _collection():
    return _client().get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def search(query_vector: np.ndarray, n: int = 50, where: dict | None = None) -> list[DenseHit]:
    """Top-`n` canonicals by cosine similarity to `query_vector` (already embedded —
    callers own the embed() call so a query is only ever embedded once per retrieval)."""
    result = _collection().query(
        query_embeddings=[query_vector.tolist()],
        n_results=n,
        where=where,
        include=["distances", "metadatas"],
    )
    if not result["ids"] or not result["ids"][0]:
        return []
    ids, distances, metadatas = result["ids"][0], result["distances"][0], result["metadatas"][0]
    return [
        DenseHit(case_id=cid, similarity=1.0 - dist, metadata=meta)
        for cid, dist, meta in zip(ids, distances, metadatas)
    ]


def similarity_to(case_id: str, query_vector: np.ndarray) -> float | None:
    """Cosine similarity between one specific case's stored embedding and `query_vector` —
    used to backfill `RetrievedCase.similarity` for a lexical-only hit (rag-design.md §7),
    which never went through `search()` and so never got a `distance` from Chroma."""
    record = _collection().get(ids=[case_id], include=["embeddings"])
    if not record["ids"]:
        return None
    stored = np.asarray(record["embeddings"][0], dtype=np.float32)
    return float(stored @ query_vector)
