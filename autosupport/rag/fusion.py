"""Reciprocal Rank Fusion + MMR, written in-repo rather than pulled from a library. Every
constant here is measured, not a default.
"""

from __future__ import annotations

import numpy as np

RRF_K = 10  # not the literature default of 60 — that structurally can't let a lexical-only
            # hit outrank a dense hit within 50-deep result lists
MMR_POOL = 30
MMR_LAMBDA = 0.7  # chosen over 0.85 specifically because 0.85 swaps under half as many
                  # results — MMR should still be doing real diversification


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """`ranked_lists` is one ranked case_id list per retrieval arm — dense first by
    convention, since ties break toward the first list's own order. Returns (case_id, score)
    pairs sorted by fused score descending. `score` is ranking-only within this one call;
    it is never comparable across two different queries' results (`merge_cases` keys
    on `similarity` instead, for exactly this reason)."""
    score: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, case_id in enumerate(ranked, start=1):
            score[case_id] = score.get(case_id, 0.0) + 1.0 / (k + rank)
    dense_order = {case_id: i for i, case_id in enumerate(ranked_lists[0])} if ranked_lists else {}
    return sorted(score.items(), key=lambda item: (-item[1], dense_order.get(item[0], 10**9)))


def mmr(
    pool_ids: list[str],
    scores: dict[str, float],
    embeddings: dict[str, np.ndarray],
    k: int,
    lam: float = MMR_LAMBDA,
) -> list[str]:
    """Maximal Marginal Relevance re-ranking of `pool_ids` (the caller passes at most
    `MMR_POOL` ids — this function doesn't re-truncate). `scores` is each id's fused RRF
    score, min-max normalised here into a [0, 1] relevance term; redundancy is the cosine
    similarity to whichever already-selected result is most similar. Every id in `pool_ids`
    must have an embedding in `embeddings` — MMR needs candidate-to-candidate similarity,
    which `similarity`-to-the-query alone doesn't give it."""
    if not pool_ids:
        return []
    raw = np.array([scores[i] for i in pool_ids])
    spread = raw.max() - raw.min()
    relevance = (raw - raw.min()) / spread if spread > 0 else np.ones_like(raw)

    chosen: list[int] = []
    remaining = list(range(len(pool_ids)))
    while remaining and len(chosen) < k:
        def value(idx: int) -> float:
            redundancy = max(
                (float(embeddings[pool_ids[idx]] @ embeddings[pool_ids[c]]) for c in chosen),
                default=0.0,
            )
            return lam * relevance[idx] - (1 - lam) * redundancy

        best = max(remaining, key=value)
        chosen.append(best)
        remaining.remove(best)
    return [pool_ids[i] for i in chosen]
