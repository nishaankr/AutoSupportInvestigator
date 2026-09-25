"""The rule pass of `verify`: grounding rules G1-G4, checked in code
against the citation regex and the enriched evidence list — deterministic, so a draft that
cites a case it never gathered can't slip past on the model's say-so."""

from __future__ import annotations

import re

from autosupport.graph.state import CITATION, DraftResponse, EvidenceEntry

# G4: template leftovers a customer must never receive. Measured: an accepted demo reply said
# "https://developer.example.com (replace with the actual link from the portal)" even though the
# skill forbids placeholders, and the claim check passed it. The corpus's own anonymisation
# tokens (<tel_num>, <name>) count too: copying one into a reply is the same failure.
PLACEHOLDER = re.compile(
    r"\bexample\.(?:com|org|net)\b|\breplace (?:this|it|with)\b|\bplaceholder\b|\{\{|<[a-z_]+>"
    r"|\[(?:your|insert|customer|company|name|link|url)\b[^\]]*\]",
    re.I,
)


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

    if decision == "resolve":  # G4
        leftovers = list(dict.fromkeys(m.group(0) for m in PLACEHOLDER.finditer(draft.resolution)))
        if leftovers:
            issues.append(f"G4: the resolution contains template placeholders: {', '.join(leftovers)}")

    analysis_cited = set(cited_ids(draft.analysis))  # G3
    for e in evidence:
        if e.stance == "contradicts" and e.case_id not in analysis_cited:
            issues.append(f"G3: contradicting case [{e.case_id}] is not acknowledged in the analysis")
    return issues
