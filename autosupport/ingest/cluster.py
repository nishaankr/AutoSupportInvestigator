"""Near-duplicate canonicalisation (docs/design/rag-design.md §4; decisions.md D3).

Star (leader) clustering over a kNN graph, not connected components: a member must be
within T of its *canonical*, so there is no transitive chaining through intermediate
members (rag-design.md §4 found connected components collapses the corpus into a handful
of giant clusters). A neighbour only joins when it also passes the guard: same
`answer_class`, answer-text similarity >= T_ANSWER, and no conflicting named entity —
otherwise a "near-duplicate problem" with a different real answer would silently merge
into one canonical and misrepresent `cluster_size` as agreement that never happened.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from autosupport.ingest.text import ENTITY

KNN_K = 50
CLUSTER_T = 0.92
ANSWER_T = 0.85


@dataclass
class ClusterResult:
    leader: np.ndarray  # leader[i] = index of i's canonical (== i for a canonical itself)
    canonical_mask: np.ndarray  # True where leader[i] == i and the row is index-eligible


def _knn(embeddings: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(embeddings)
    neighbours = np.zeros((n, k), dtype=np.int32)
    similarities = np.zeros((n, k), dtype=np.float32)
    for start in range(0, n, 1024):
        block = embeddings[start : start + 1024] @ embeddings.T
        block[np.arange(len(block)), np.arange(start, start + len(block))] = -1  # never your own neighbour
        top = np.argpartition(-block, k, axis=1)[:, :k]
        top_sims = np.take_along_axis(block, top, 1)
        order = np.argsort(-top_sims, axis=1)
        neighbours[start : start + 1024] = np.take_along_axis(top, order, 1)
        similarities[start : start + 1024] = np.take_along_axis(top_sims, order, 1)
    return neighbours, similarities


def cluster(
    records: pd.DataFrame,
    embed_vectors: np.ndarray,
    answer_vectors: np.ndarray,
    eligible: np.ndarray,
    t: float = CLUSTER_T,
    t_answer: float = ANSWER_T,
) -> ClusterResult:
    """`records` needs `embed_text` (for entities) and `answer_class` columns, aligned
    row-for-row with `embed_vectors`/`answer_vectors`. `eligible` marks rows past the
    minimum-content guard — ineligible rows are never a leader and never join a cluster,
    so they stay in SQLite as singletons, reachable by ID but not by similarity search."""
    n = len(records)
    neighbours, similarities = _knn(embed_vectors, min(KNN_K, n - 1))
    entities = [set(ENTITY.findall(t)) for t in records["embed_text"]]
    answer_class = records["answer_class"].to_numpy()
    body_len = records["body_ix"].str.len().to_numpy()

    def compatible(i: int, j: int) -> bool:
        if answer_class[i] != answer_class[j]:
            return False
        if float(answer_vectors[i] @ answer_vectors[j]) < t_answer:
            return False
        a, b = entities[i], entities[j]
        return not (a - b and b - a)  # conflict = each names an entity the other lacks

    degree = (similarities >= t).sum(axis=1)
    # Leaders are chosen most-connected first, then most-complete body — both deterministic,
    # so re-running ingest on the same data always canonicalises the same way.
    order = np.lexsort((-body_len, -degree))
    leader = np.full(n, -1, dtype=np.int64)
    for i in order:
        if leader[i] >= 0:
            continue
        leader[i] = i
        if not eligible[i]:
            continue
        for k in range(similarities.shape[1]):
            if similarities[i, k] < t:
                break
            j = int(neighbours[i, k])
            if leader[j] >= 0 or not eligible[j]:
                continue
            if compatible(i, j):
                leader[j] = i

    canonical_mask = (leader == np.arange(n)) & eligible
    return ClusterResult(leader=leader, canonical_mask=canonical_mask)
