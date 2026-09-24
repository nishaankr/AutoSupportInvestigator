"""`compute_confidence` against the five worked examples in output-schema.md §4.7."""

from __future__ import annotations

from autosupport.graph.confidence import compute_confidence
from autosupport.graph.state import EvidenceAssessment, EvidenceEntry

TAU_REL = 0.76


def _entry(case_id, cluster_size, similarity, stance="supports", answer_class="resolution"):
    return EvidenceEntry(
        case_id=case_id, summary="s", stance=stance, source="dataset", subject="subj",
        answer_class=answer_class, cluster_size=cluster_size, similarity=similarity,
    )


def _assessment(verdict="sufficient", missing=0):
    return EvidenceAssessment(
        verdict=verdict, relevant_count=1, top_score=1.0, clusters=[], dominant_share=1.0,
        missing_slots=["slot"] * missing, gap_is_retrievable=False, next_action="resolve", reason="r",
    )


def test_example_a_strong_resolve():
    evidence = [_entry("HF-1", 8, 0.91), _entry("HF-2", 3, 0.87), _entry("HF-3", 1, 0.83)]
    result = compute_confidence(evidence, _assessment(), True, TAU_REL)
    assert result.value == 0.86
    assert result.band == "high"
    assert result.cap_reason is None


def test_example_b_contested_resolve_capped_conflicting():
    evidence = [
        _entry("HF-1", 15, 0.85), _entry("HF-2", 1, 0.80),
        _entry("HF-3", 3, None, stance="contradicts"),
    ]
    result = compute_confidence(evidence, _assessment("conflicting"), True, TAU_REL)
    assert result.value == 0.60
    assert result.band == "medium"
    assert result.cap_reason == "conflicting"


def test_example_c_thin_resolve():
    evidence = [_entry("HF-1", 1, 0.80), _entry("HF-2", 1, 0.78), _entry("HF-3", 1, 0.77)]
    result = compute_confidence(evidence, _assessment(), True, TAU_REL)
    assert result.value == 0.52
    assert result.band == "medium"


def test_example_d_rule_hit_escalation_scores_high():
    evidence = [_entry("HF-1", 4, 0.88), _entry("HF-2", 2, 0.84), _entry("HF-3", 1, 0.81)]
    result = compute_confidence(evidence, _assessment(), True, TAU_REL)
    assert result.value == 0.78
    assert result.band == "high"


def test_example_e_no_substantive_support_floors_at_zero():
    evidence = [_entry("HF-1", 1, 0.77, answer_class="escalation")]
    result = compute_confidence(evidence, _assessment("insufficient", missing=2), None, TAU_REL)
    assert result.value == 0.00
    assert result.band == "low"
    assert result.cap_reason == "no_substantive_support"
