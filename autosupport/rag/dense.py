"""Chroma similarity search — the dense arm of hybrid retrieval (rag-design.md §3, §7).

Every result carries a true cosine similarity (`1 - distance`), not a ranking-only score —
the collection is created with `hnsw:space: "cosine"` explicitly (ingest/index.py), since
Chroma's default is squared L2, and `output-schema.md`'s confidence formula and
`graph-design.md`'s `tau_rel` both compare against real cosine similarity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache

import numpy as np

from autosupport.config import settings

COLLECTION_NAME = "support_cases"
_TAG_SLUG = re.compile(r"[^a-z0-9]+")


def tag_metadata(tags: list[str]) -> dict:
    """Chroma metadata for a case's tags: the joined list plus one boolean per tag slug
    (Chroma can't filter inside a list). Same shape for dataset and agent-resolved cases."""
    if not tags:
        return {}
    slugs = {"tag_" + _TAG_SLUG.sub("_", t.strip().lower()).strip("_"): True for t in tags}
    return {"tags": ", ".join(tags), **slugs}


def upsert(case_id: str, vector: np.ndarray, document: str, metadata: dict) -> None:
    _collection().upsert(ids=[case_id], embeddings=[vector.tolist()], documents=[document], metadatas=[metadata])


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


def similarities_to(case_ids: list[str], query_vector: np.ndarray) -> dict[str, float]:
    """Batched `similarity_to`: cosine between each case's stored embedding and
    `query_vector`, in one Chroma round-trip. Used to re-anchor `RetrievedCase.similarity`
    to the ticket for results found by a different query (a rewrite, a tool search)."""
    unique = list(dict.fromkeys(case_ids))  # Chroma rejects duplicate IDs; two searches can share a hit
    if not unique:
        return {}
    record = _collection().get(ids=unique, include=["embeddings"])
    return {
        cid: float(np.asarray(vec, dtype=np.float32) @ query_vector)
        for cid, vec in zip(record["ids"], record["embeddings"])
    }


def similarity_to(case_id: str, query_vector: np.ndarray) -> float | None:
    """Cosine similarity between one specific case's stored embedding and `query_vector` —
    used to backfill `RetrievedCase.similarity` for a lexical-only hit (rag-design.md §7),
    which never went through `search()` and so never got a `distance` from Chroma."""
    record = _collection().get(ids=[case_id], include=["embeddings"])
    if not record["ids"]:
        return None
    stored = np.asarray(record["embeddings"][0], dtype=np.float32)
    return float(stored @ query_vector)
