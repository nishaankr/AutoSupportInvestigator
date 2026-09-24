"""The rule pass of `verify` (output-schema.md §3.5): grounding rules G1-G3, checked in code
against the citation regex and the enriched evidence list — deterministic, so a draft that
cites a case it never gathered can't slip past on the model's say-so."""

from __future__ import annotations

from autosupport.graph.state import CITATION, DraftResponse, EvidenceEntry


def cited_ids(text: str) -> list[str]:
    return list(dict.fromkeys(CITATION.findall(text)))


def grounding_issues(draft: DraftResponse, evidence: list[EvidenceEntry], decision: str) -> list[str]:
    by_id = {e.case_id: e for e in evidence}
    handoff = draft.escalation.handoff_summary if draft.escalation else ""
    issues: list[str] = []

    for cid in cited_ids(f"{draft.analysis}\n{draft.resolution}\n{handoff}"):  # G1
        if cid not in by_id:
            issues.append(f"G1: cites [{cid}], which is not in the evidence list")

    if decision == "resolve":  # G2
        cited = set(cited_ids(draft.resolution))
        if not any(by_id[c].stance == "supports" and by_id[c].answer_class == "resolution" for c in cited if c in by_id):
            issues.append("G2: the resolution cites no supporting case whose historical answer was a resolution")

    analysis_cited = set(cited_ids(draft.analysis))  # G3
    for e in evidence:
        if e.stance == "contradicts" and e.case_id not in analysis_cited:
            issues.append(f"G3: contradicting case [{e.case_id}] is not acknowledged in the analysis")
    return issues
