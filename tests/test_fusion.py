import numpy as np

from autosupport.rag.fusion import mmr, reciprocal_rank_fusion


def test_rrf_favours_documents_in_both_arms():
    dense = ["a", "b", "c"]
    lexical = ["c", "d", "a"]
    fused = reciprocal_rank_fusion([dense, lexical], k=10)
    ranked_ids = [case_id for case_id, _ in fused]
    # "a" (dense rank 1, lexical rank 3) and "c" (dense rank 3, lexical rank 1) both appear
    # in both arms and should outrank "b"/"d", which appear in only one.
    assert ranked_ids.index("a") < ranked_ids.index("b")
    assert ranked_ids.index("c") < ranked_ids.index("d")


def test_rrf_lexical_only_hit_can_outrank_weak_dense_hit_at_low_k():
    # A rank-1 lexical-only hit should be able to beat a weak (rank-50) dense-only hit at
    # k=10 — this is exactly the structural property k=10 was chosen for.
    dense = [f"d{i}" for i in range(1, 51)]
    lexical = ["lex1"]
    fused = reciprocal_rank_fusion([dense, lexical], k=10)
    ranked_ids = [case_id for case_id, _ in fused]
    assert ranked_ids.index("lex1") < ranked_ids.index("d50")


def test_rrf_ties_break_on_first_list_order():
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=10)
    # "a" and "b" get the same summed score (rank 1 + rank 2 each); the tie breaks toward
    # the first (dense) list's own order, so "a" (dense rank 1) sorts first.
    assert [case_id for case_id, _ in fused] == ["a", "b"]


def test_mmr_respects_k():
    ids = ["a", "b", "c", "d"]
    scores = {i: 1.0 - 0.1 * n for n, i in enumerate(ids)}
    embeddings = {i: np.array([1.0, 0.0], dtype=np.float32) for i in ids}
    selected = mmr(ids, scores, embeddings, k=2)
    assert len(selected) == 2


def test_mmr_prefers_diverse_candidate_over_near_duplicate_of_top_pick():
    # b is a near-duplicate of a; c is distinct but slightly lower-scored. At a redundancy-
    # sensitive lambda, MMR should still pick a first (highest score) but should not simply
    # pick the next-highest score if it's a duplicate of what's already chosen.
    ids = ["a", "b", "c"]
    scores = {"a": 1.0, "b": 0.99, "c": 0.80}
    embeddings = {
        "a": np.array([1.0, 0.0], dtype=np.float32),
        "b": np.array([0.999, 0.045], dtype=np.float32),  # nearly identical to a
        "c": np.array([0.0, 1.0], dtype=np.float32),  # orthogonal to a
    }
    selected = mmr(ids, scores, embeddings, k=2, lam=0.5)
    assert selected[0] == "a"
    assert selected[1] == "c"  # not "b", despite b's higher raw score
