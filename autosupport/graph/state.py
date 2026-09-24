"""`AgentState` and every Pydantic sub-model it carries (`state-schema.md` §2-3,
`output-schema.md` §2-§3, §7). One module for both, because `state-schema.md` §3 explicitly
defers `CaseResult`/`Confidence`/etc. to `output-schema.md` to avoid a second copy drifting
out of sync — this file is that single copy.

Deviations from the doc sketches, tracked in `docs/project/decisions.md` and reflected in the
docs: `TicketInput.submitted_at`/`ClarificationTurn.asked_at` use a timezone-aware factory,
not the deprecated `datetime.utcnow`; `AgentState.escalation_trigger` is added (D15 F4).
"""

from __future__ import annotations

import operator
import re
from datetime import datetime, timezone
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, computed_field, model_validator

# ---------- enums ----------
CaseStatus = Literal["open", "investigating", "awaiting_user", "resolved", "escalated"]
Verdict = Literal["sufficient", "insufficient", "conflicting"]
NextAction = Literal["resolve", "escalate", "refine_retrieval", "ask_user"]
Priority = Literal["low", "medium", "high", "critical"]
AnswerClass = Literal["resolution", "clarification_request", "escalation"]  # rag-design.md §5
EscalationTrigger = Literal["rule", "evidence_exhausted", "verification_failed", "user_rejected"]
CapReason = Literal["no_substantive_support", "insufficient", "conflicting", "verification_failed"]

# output-schema.md §3.4 — the only citation syntax used in prose.
CASE_ID = r"HF-\d+|T-\d{8}-[0-9a-f]{6}"
CITATION = re.compile(rf"\[({CASE_ID})\]")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------- identity & input (state-schema.md §2.1) ----------
class TicketInput(BaseModel):
    subject: str
    body: str
    customer_priority: Priority | None = None
    customer_tags: list[str] = Field(default_factory=list)
    submitted_at: datetime = Field(default_factory=_utcnow)


# ---------- triage (state-schema.md §2.4) ----------
class Classification(BaseModel):
    queue: str
    type: str
    priority: Priority
    tags: list[str]
    rationale: str
    neighbor_agreement: float  # computed by code from retrieved_cases, not the model — see triage.py


# ---------- retrieval (state-schema.md §2.5) ----------
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


# ---------- investigation (state-schema.md §2.6, output-schema.md §3) ----------
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


# ---------- clarification (state-schema.md §2.7) ----------
class ClarificationTurn(BaseModel):
    question: str
    answer: str
    asked_at: datetime = Field(default_factory=_utcnow)


# ---------- decision, draft & verification (state-schema.md §2.8, output-schema.md §2.1/§7) ----------
class EscalationDraft(BaseModel):  # model-authored, inside DraftResponse
    target_queue: str
    reason: str
    handoff_summary: str


class DraftResponse(BaseModel):
    analysis: str
    resolution: str
    escalation: EscalationDraft | None = None


class Confidence(BaseModel):
    """output-schema.md §4.6. Computed only by `verify` (§4.5); every path to
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


class VerificationResult(BaseModel):  # in-graph state, written by `verify` (CP5)
    passed: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    recommended_action: Literal["none", "re_reason", "re_retrieve"] = "none"


class VerificationOutcome(BaseModel):  # persisted form, output-schema.md §2
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


# ---------- memory (state-schema.md §2.3, memory-design.md) ----------
class CustomerMemory(BaseModel):
    customer_id: str
    facts: dict[str, str] = Field(default_factory=dict)
    flags: list[str] = Field(default_factory=list)
    tried_fixes: list[str] = Field(default_factory=list)
    preferences: dict[str, str] = Field(default_factory=dict)


class CaseSummary(BaseModel):
    case_id: str
    status: CaseStatus
    subject: str
    queue: str | None
    resolution_snippet: str | None
    updated_at: datetime


# ---------- output (output-schema.md §2, §7) ----------
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


# ---------- reducers ----------
MAX_CASES_IN_STATE = 30


def merge_cases(left: list[RetrievedCase] | None, right: list[RetrievedCase] | None) -> list[RetrievedCase]:
    """Union by case_id, keep the copy with higher `similarity`, sort desc, cap size.
    Keyed on `similarity`, not the fused `score` — an RRF score is only meaningful relative
    to the other results of the *same* query, so comparing scores across different queries'
    results (which is exactly what merging retrieval rounds does) isn't a valid comparison.
    `similarity` is on one fixed scale regardless of which query or round found the case
    (rag-design.md §7/§11). Safe for concurrent writes from parallel Send branches."""
    by_id: dict[str, RetrievedCase] = {c.case_id: c for c in (left or [])}
    for c in right or []:
        prev = by_id.get(c.case_id)
        if prev is None or c.similarity > prev.similarity:
            by_id[c.case_id] = c
    return sorted(by_id.values(), key=lambda c: c.similarity, reverse=True)[:MAX_CASES_IN_STATE]


# ---------- graph state ----------
class AgentState(TypedDict, total=False):
    # identity & input
    ticket_id: str
    customer_id: str
    thread_id: str
    ticket: TicketInput
    status: CaseStatus
    # short-term conversation
    messages: Annotated[list[AnyMessage], add_messages]
    # memory
    customer_profile: CustomerMemory | None
    customer_history: list[CaseSummary]
    # triage
    classification: Classification | None
    active_skills: list[str]
    # retrieval
    retrieved_cases: Annotated[list[RetrievedCase], merge_cases]
    retrieval_queries: Annotated[list[RetrievalQuery], operator.add]
    retrieval_round: int
    # investigation
    hypothesis: Hypothesis | None
    evidence: list[EvidenceEntry]
    evidence_assessment: EvidenceAssessment | None
    tool_calls_this_round: int
    tool_log: Annotated[list[ToolCallRecord], operator.add]
    # clarification
    pending_question: str | None
    clarifications: Annotated[list[ClarificationTurn], operator.add]
    clarification_count: int
    # decision / draft / verification
    decision: Literal["resolve", "escalate"] | None
    draft: DraftResponse | None
    # Written only by `escalate`, captured when it runs: a later verify pass would change what
    # output-schema.md §2.1's precedence computes at persist time (decisions.md D15 F4).
    escalation_trigger: EscalationTrigger | None
    confidence: Confidence | None
    verification: VerificationResult | None
    verify_attempts: int
    # acceptance / output
    user_acceptance: Literal["pending", "accepted", "rejected"] | None
    user_feedback: str | None
    revision_count: int
    final_output: CaseResult | None
    errors: Annotated[list[str], operator.add]


# ---------- public I/O schemas ----------
class InputState(TypedDict):
    ticket_id: str  # generated by service.new_ticket() before invoking — see case-persistence.md §6
    customer_id: str
    ticket: TicketInput


class OutputState(TypedDict):
    ticket_id: str
    status: CaseStatus
    final_output: CaseResult | None
    pending_question: str | None


# Every Pydantic type that can appear in a checkpointed `AgentState`, for the SqliteSaver
# serde allowlist (graph/build.py) — see docs/design/state-schema.md §4 "Implementation
# notes" and the risk this resolves, recorded in docs/project/decisions.md's CP3 entry.
CHECKPOINTED_MODELS: list[type[BaseModel]] = [
    TicketInput, Classification, RetrievedCase, RetrievalQuery, Hypothesis, EvidenceItem,
    EvidenceEntry, ApproachCluster, EvidenceAssessment, ToolCallRecord, ClarificationTurn,
    EscalationDraft, DraftResponse, Confidence, VerificationResult, VerificationOutcome,
    EscalationBlock, CustomerMemory, CaseSummary, RunStats, CaseResult,
]
