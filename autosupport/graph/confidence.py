"""The confidence score: computed from the evidence, never reported by a model
(output-schema.md §4, decisions.md D12).

Only `verify` calls this, once per draft. Roughly: how much resolution-class evidence
supports the answer (weighted by how many historical tickets each case stands for), how much
contradicts it, and how close the supporting cases are to this ticket — minus a penalty per
missing fact, and capped when the evidence was thin, conflicting or failed the check.

The constants below *define* the scale, so changing one changes the meaning of every stored
score. That's why they live here as code, not in config.
"""

from __future__ import annotations

import math

from autosupport.graph.state import Confidence, EvidenceAssessment, EvidenceEntry

K = 2.5
W_MAX = 5.0
SIM_CEILING = 0.92
AGREEMENT_FLOOR = 0.4
RELEVANCE_FLOOR = 0.7
PENALTY_STEP = 0.10
PENALTY_CEILING = 0.30
BAND_HIGH = 0.75
BAND_MEDIUM = 0.50

# (cap_reason, cap_value), checked in this order; the lowest *applicable* value wins
# (output-schema.md §4.4 "cap = lowest applicable of").
_CAP_NO_SUBSTANTIVE_SUPPORT = ("no_substantive_support", 0.20)
_CAP_INSUFFICIENT = ("insufficient", 0.45)
_CAP_CONFLICTING = ("conflicting", 0.60)
_CAP_VERIFICATION_FAILED = ("verification_failed", 0.30)


def cluster_weight(cluster_size: int) -> float:
    return min(1.0 + math.log2(cluster_size), W_MAX)


def _substantive(entry: EvidenceEntry) -> bool:
    # Only resolutions count: a clarification request shows someone asked a question, an
    # escalation carries no fix, and an open customer case has no answer yet.
    return entry.answer_class == "resolution"


def compute_confidence(
    evidence: list[EvidenceEntry],
    assessment: EvidenceAssessment,
    verification_passed: bool | None,
    tau_rel: float,
) -> Confidence:
    """`verification_passed=None` means the check hasn't run yet: `verify` computes the band
    first (so its claim check knows what wording the evidence allows), then again with the
    outcome, which applies the cap on failure."""
    supports = [e for e in evidence if e.stance == "supports"]
    contradicts = [e for e in evidence if e.stance == "contradicts"]

    w_s = sum(cluster_weight(e.cluster_size) for e in supports if _substantive(e))
    w_c = sum(cluster_weight(e.cluster_size) for e in contradicts if _substantive(e))

    support = 1.0 - math.exp(-w_s / K)
    agreement = w_s / (w_s + w_c) if (w_s + w_c) > 0 else 0.0

    # Relevance asks whether the supporting cases really resemble this ticket, from their
    # measured similarity — a check on the model's "supports" judgements, not a repeat of them.
    ranked_similarities = sorted(
        (e.similarity for e in supports if e.similarity is not None), reverse=True
    )[:3]
    if ranked_similarities:
        relevance = sum(
            _clamp((sim - tau_rel) / (SIM_CEILING - tau_rel), 0.0, 1.0) for sim in ranked_similarities
        ) / len(ranked_similarities)
    else:
        relevance = 0.0

    base = support * (AGREEMENT_FLOOR + (1 - AGREEMENT_FLOOR) * agreement)
    base *= RELEVANCE_FLOOR + (1 - RELEVANCE_FLOOR) * relevance
    penalty = min(PENALTY_STEP * len(assessment.missing_slots), PENALTY_CEILING)

    cap_reason, cap_value = _lowest_applicable_cap(w_s, assessment.verdict, verification_passed)
    value = round(_clamp(min(base - penalty, cap_value), 0.0, 1.0), 2)

    return Confidence(
        value=value,
        support=round(support, 3),
        agreement=round(agreement, 3),
        relevance=round(relevance, 3),
        penalty=round(penalty, 3),
        cap_reason=cap_reason,
    )


def _lowest_applicable_cap(
    w_s: float, verdict: str, verification_passed: bool | None
) -> tuple[str | None, float]:
    candidates: list[tuple[str, float]] = []
    if w_s == 0:
        candidates.append(_CAP_NO_SUBSTANTIVE_SUPPORT)
    if verdict == "insufficient":
        candidates.append(_CAP_INSUFFICIENT)
    if verdict == "conflicting":
        candidates.append(_CAP_CONFLICTING)
    if verification_passed is False:
        candidates.append(_CAP_VERIFICATION_FAILED)
    if not candidates:
        return None, 1.0
    return min(candidates, key=lambda c: c[1])


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))
