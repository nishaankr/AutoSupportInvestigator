"""The graph's state and every model it carries, from the ticket coming in to the final
`CaseResult` going out.

Working state and the final output live in one module on purpose: the output reuses the
state's models (evidence, confidence, classification), and a second copy would drift.
Anything written by parallel branches has a reducer; everything else has exactly one writer.
"""

from __future__ import annotations

import operator
import re
from datetime import datetime, timezone
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, computed_field, model_validator

CaseStatus = Literal["open", "investigating", "awaiting_user", "resolved", "escalated"]
Verdict = Literal["sufficient", "insufficient", "conflicting"]
NextAction = Literal["resolve", "escalate", "refine_retrieval", "ask_user"]
Priority = Literal["low", "medium", "high", "critical"]
AnswerClass = Literal["resolution", "clarification_request", "escalation"]
EscalationTrigger = Literal["rule", "evidence_exhausted", "verification_failed", "user_rejected"]
CapReason = Literal["no_substantive_support", "insufficient", "conflicting", "verification_failed"]

# The only citation syntax used in prose.
CASE_ID = r"HF-\d+|T-\d{8}-[0-9a-f]{6}"
CITATION = re.compile(rf"\[({CASE_ID})\]")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TicketInput(BaseModel):
    subject: str
    body: str
    customer_priority: Priority | None = None
    customer_tags: list[str] = Field(default_factory=list)
    submitted_at: datetime = Field(default_factory=_utcnow)


class Classification(BaseModel):
    queue: str
    type: str
    priority: Priority
    tags: list[str]
    rationale: str
    neighbor_agreement: float  # computed by code from retrieved_cases, not the model — see triage.py


class RetrievedCase(BaseModel):
    case_id: str
    source: Literal["dataset", "agent_resolved"]
    subject: str
    body_snippet: str
    answer_snippet: str
    queue: str | None = None
    type: str | None = None
    priority: str | None = None
    tags: list[str] = Field(default_factory=list)
    score: float
    similarity: float
    cluster_size: int = 1
    answer_class: AnswerClass | None = None
    retrieval_round: int
    query_label: str


class RetrievalQuery(BaseModel):
    text: str
    filters: dict = Field(default_factory=dict)
    k: int
    round: int
    label: str
    n_results: int


class Hypothesis(BaseModel):
    statement: str
    root_cause_category: str
    supporting_case_ids: list[str]
    contradicting_case_ids: list[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):  # model-authored
    case_id: str
    summary: str
    stance: Literal["supports", "contradicts", "neutral"]


class EvidenceEntry(EvidenceItem):  # enriched by code — graph/evidence.py
    source: Literal["dataset", "agent_resolved", "customer_history"]
    subject: str
    answer_class: AnswerClass | None
    cluster_size: int = Field(ge=1)
    similarity: float | None = None
    approach: str | None = None


class ApproachCluster(BaseModel):
    label: str
    case_ids: list[str]


class Findings(BaseModel):
    """What `investigate` submits to end a round, as the arguments of its `submit_findings` tool
    call: the hypothesis and evidence plus the judgement `assess_evidence` used to ask a second
    model for. One model call instead of two; `assess_evidence` turns it into
    a verdict and a route in Python."""

    hypothesis: str = Field(description="One or two sentences: what is wrong and what fixes it, or that no fix is on record.")
    root_cause_category: str = Field(description="Short label, e.g. 'configuration', 'billing', 'feature request'.")
    supporting_case_ids: list[str] = Field(default_factory=list)
    contradicting_case_ids: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(
        default_factory=list,
        description="EVERY retrieved case that supports or contradicts the hypothesis, with its stance — typically 2 to 6, "
                    "not only the single best match. A fix is only recommended when at least two cases support it.")
    clusters: list[ApproachCluster] = Field(
        default_factory=list,
        description="Group EVERY case marked relevant=yes (not only the ones you cite) by the approach their historical "
                    "answers took; each case in at most one cluster.")
    missing_slots: list[str] = Field(
        default_factory=list,
        description="At most 2 facts that are not in the ticket, profile or the customer's answers AND without which the fix "
                    "can't be chosen or applied. Not details historical agents merely asked for; a general request "
                    "(information, recommendations, pricing, how-to) needs none. Usually empty.")
    gap_is_retrievable: bool = Field(
        default=False,
        description="True only if what you already know is enough to write a better search (narrower queue, sharper terms). "
                    "False when the gap is a fact only the customer has.")
    history_contradicts: bool = Field(default=False, description="The customer's own history contradicts the dominant approach.")
    requires_human_action: str | None = Field(
        default=None, description="If the fix needs something an agent can't do (refund, account change, on-site visit), what; else null.")
    clarification_question: str | None = Field(
        default=None, description="If missing_slots is non-empty: ONE focused question to the customer asking for exactly those facts.")
    reason: str = Field(description="One or two sentences: why the evidence is or isn't enough.")


class EvidenceAssessment(BaseModel):
    verdict: Verdict
    relevant_count: int
    top_score: float
    clusters: list[ApproachCluster]
    dominant_share: float
    missing_slots: list[str] = Field(default_factory=list)
    gap_is_retrievable: bool
    escalation_rule_hit: str | None = None
    next_action: NextAction
    reason: str


class ToolCallRecord(BaseModel):
    name: str
    args: dict
    ok: bool
    duration_ms: int
    round: int


class ClarificationTurn(BaseModel):
    question: str
    answer: str
    asked_at: datetime = Field(default_factory=_utcnow)


class EscalationDraft(BaseModel):  # model-authored, inside DraftResponse
    target_queue: str
    reason: str
    handoff_summary: str


class DraftResponse(BaseModel):
    analysis: str
    resolution: str
    escalation: EscalationDraft | None = None


class Confidence(BaseModel):
    """Computed only by `verify`; every path to
    `persist_case` passes through `verify`, so `CaseResult.confidence` is required."""

    value: float = Field(ge=0, le=1)
    support: float
    agreement: float
    relevance: float
    penalty: float
    cap_reason: CapReason | None = None

    @computed_field
    @property
    def band(self) -> Literal["high", "medium", "low"]:
        return "high" if self.value >= 0.75 else "medium" if self.value >= 0.50 else "low"


class VerificationResult(BaseModel):  # in-graph state, written by `verify`
    passed: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    recommended_action: Literal["none", "re_reason", "re_retrieve"] = "none"


class VerificationOutcome(BaseModel):  # persisted form
    passed: bool
    attempts: int
    unresolved_issues: list[str] = Field(default_factory=list)


class EscalationBlock(BaseModel):
    required: bool
    trigger: EscalationTrigger | None = None
    rule: str | None = None
    target_queue: str | None = None
    reason: str | None = None
    handoff_summary: str | None = None


class CustomerMemory(BaseModel):
    customer_id: str
    facts: dict[str, str] = Field(default_factory=dict)
    flags: list[str] = Field(default_factory=list)
    tried_fixes: list[str] = Field(default_factory=list)
    preferences: dict[str, str] = Field(default_factory=dict)
    # "facts.<key>" / "preferences.<key>" -> ticket_id that last wrote it. Rendered by
    # `autosupport memory`, never put into a prompt.
    provenance: dict[str, str] = Field(default_factory=dict)


class CaseSummary(BaseModel):
    case_id: str
    status: CaseStatus
    subject: str
    queue: str | None
    resolution_snippet: str | None
    updated_at: datetime


class RunStats(BaseModel):
    retrieval_rounds: int
    tool_calls: int
    verify_attempts: int
    revisions: int


class CaseResult(BaseModel):
    ticket_id: str
    customer_id: str
    status: Literal["resolved", "escalated"]
    classification: Classification
    evidence: list[EvidenceEntry]
    analysis: str
    resolution: str
    escalation: EscalationBlock
    confidence: Confidence
    verification: VerificationOutcome
    acceptance: Literal["accepted", "rejected", "not_required"]
    clarifications: list[ClarificationTurn] = Field(default_factory=list)
    stats: RunStats
    errors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _escalation_matches_status(self) -> "CaseResult":
        if self.status == "escalated" and not self.escalation.required:
            raise ValueError("status is 'escalated' but escalation.required is False")
        if self.status == "resolved" and self.escalation.required:
            raise ValueError("status is 'resolved' but escalation.required is True")
        return self

    @model_validator(mode="after")
    def _escalation_fields_present_iff_required(self) -> "CaseResult":
        fields = (self.escalation.trigger, self.escalation.target_queue,
                  self.escalation.reason, self.escalation.handoff_summary)
        if self.escalation.required and any(f is None for f in fields):
            raise ValueError("escalation.required is True but a required escalation field is None")
        if not self.escalation.required and any(f is not None for f in fields):
            raise ValueError("escalation.required is False but an escalation field is set")
        return self

    @model_validator(mode="after")
    def _acceptance_implies_resolved(self) -> "CaseResult":
        if self.acceptance == "accepted" and self.status != "resolved":
            raise ValueError("acceptance is 'accepted' but status is not 'resolved'")
        return self

    @model_validator(mode="after")
    def _evidence_ids_unique_and_well_formed(self) -> "CaseResult":
        seen: set[str] = set()
        pattern = re.compile(rf"^(?:{CASE_ID})$")
        for entry in self.evidence:
            if not pattern.match(entry.case_id):
                raise ValueError(f"evidence case_id {entry.case_id!r} doesn't match {CASE_ID}")
            if entry.case_id in seen:
                raise ValueError(f"duplicate evidence case_id {entry.case_id!r}")
            seen.add(entry.case_id)
        return self


MAX_CASES_IN_STATE = 30


def merge_cases(left: list[RetrievedCase] | None, right: list[RetrievedCase] | None) -> list[RetrievedCase]:
    """Union by case_id, keep the copy with higher `similarity`, sort desc, cap size.
    Keyed on `similarity`, not the fused `score` — an RRF score is only meaningful relative
    to the other results of the *same* query, so comparing scores across different queries'
    results (which is exactly what merging retrieval rounds does) isn't a valid comparison.
    `similarity` is on one fixed scale regardless of which query or round found the case. Safe for concurrent writes from parallel Send branches."""
    by_id: dict[str, RetrievedCase] = {c.case_id: c for c in (left or [])}
    for c in right or []:
        prev = by_id.get(c.case_id)
        if prev is None or c.similarity > prev.similarity:
            by_id[c.case_id] = c
    return sorted(by_id.values(), key=lambda c: c.similarity, reverse=True)[:MAX_CASES_IN_STATE]


class AgentState(TypedDict, total=False):
    ticket_id: str
    customer_id: str
    thread_id: str
    ticket: TicketInput
    status: CaseStatus
    messages: Annotated[list[AnyMessage], add_messages]
    customer_profile: CustomerMemory | None
    customer_history: list[CaseSummary]
    classification: Classification | None
    active_skills: list[str]
    retrieved_cases: Annotated[list[RetrievedCase], merge_cases]
    retrieval_queries: Annotated[list[RetrievalQuery], operator.add]
    retrieval_round: int
    hypothesis: Hypothesis | None
    evidence: list[EvidenceEntry]
    findings: Findings | None  # written only by `investigate` (its submit_findings call)
    evidence_assessment: EvidenceAssessment | None
    tool_calls_this_round: int
    tool_log: Annotated[list[ToolCallRecord], operator.add]
    pending_question: str | None
    clarifications: Annotated[list[ClarificationTurn], operator.add]
    clarification_count: int
    decision: Literal["resolve", "escalate"] | None
    draft: DraftResponse | None
    # Written only by `escalate`, captured when it runs: a later verify pass would change what
    # the trigger precedence computes at persist time.
    escalation_trigger: EscalationTrigger | None
    confidence: Confidence | None
    verification: VerificationResult | None
    verify_attempts: int
    user_acceptance: Literal["pending", "accepted", "rejected"] | None
    user_feedback: str | None
    revision_count: int
    final_output: CaseResult | None
    errors: Annotated[list[str], operator.add]


class InputState(TypedDict):
    ticket_id: str  # generated by service.new_ticket() before invoking
    customer_id: str
    ticket: TicketInput


class OutputState(TypedDict):
    ticket_id: str
    status: CaseStatus
    final_output: CaseResult | None
    pending_question: str | None


# Every Pydantic type that can appear in a checkpointed `AgentState`, for the SqliteSaver
# serde allowlist (graph/build.py): a type missing here would fail to deserialise on resume.
CHECKPOINTED_MODELS: list[type[BaseModel]] = [
    TicketInput, Classification, RetrievedCase, RetrievalQuery, Hypothesis, EvidenceItem,
    EvidenceEntry, ApproachCluster, Findings, EvidenceAssessment, ToolCallRecord, ClarificationTurn,
    EscalationDraft, DraftResponse, Confidence, VerificationResult, VerificationOutcome,
    EscalationBlock, CustomerMemory, CaseSummary, RunStats, CaseResult,
]
