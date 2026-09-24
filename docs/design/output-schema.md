# Output Schema — `CaseResult`, Confidence, Evidence

> **Status:** Draft v1 · the structured deliverable produced for every ticket.
> **Companion docs:** `state-schema.md` (where these models sit in state), `graph-design.md` (which nodes write them), `decisions.md` D12 (why confidence is computed).
> Built by `persist_case`, stored as JSON in `cases.final_output`, returned by `service.py`, rendered by `cli.py`, dumped verbatim by `--json`.

---

## 1. Principles

1. **Six required fields.** `classification`, `evidence`, `analysis`, `resolution`, `escalation` and `confidence` are the brief's output contract. Every other field is additional audit data.
2. **Every field has one author.** A field is written either by the model or by code, never both. Facts about the corpus or the run are computed by code: IDs, cluster sizes, answer classes, similarity, confidence, status and counters. The model writes prose and judgements: summaries, stance, analysis, the resolution and the escalation reason. The model can't inflate a number it never writes.
3. **Evidence is checkable.** Every case ID in the output resolves to a row in SQLite, and every ID cited in prose appears in the evidence list.
4. **One shape for both outcomes.** Resolved and escalated tickets produce the same `CaseResult`. `status` and `escalation.required` say which one it is.

---

## 2. `CaseResult`

| Field | Type | Author | Notes |
|---|---|---|---|
| `ticket_id` | `str` | code | `T-YYYYMMDD-<6 hex>` |
| `customer_id` | `str` | code | |
| `status` | `"resolved" \| "escalated"` | code | Terminal `CaseStatus`. `status == "escalated"` exactly when `escalation.required`. |
| `classification` | `Classification` | model | Written by `triage` and possibly revised by `investigate`. Unchanged from `state-schema.md` §3. |
| `evidence` | `list[EvidenceEntry]` | mixed | §3. Ordered, at most 10 entries. |
| `analysis` | `str` | model | Written for a support agent: the hypothesis, how the evidence agreed or conflicted, and what was ruled out. Cites case IDs inline (§3.4). |
| `resolution` | `str` | model | The customer-facing reply. For a resolved case, the steps, each citing the case it came from. For an escalated case, the holding reply saying what happens next. It never contains an invented fix. |
| `escalation` | `EscalationBlock` | mixed | §2.1 |
| `confidence` | `Confidence` | code | §4. |
| `verification` | `VerificationOutcome` | code | Written by `verify`: `passed`, `attempts`, `unresolved_issues`. `unresolved_issues` is non-empty only for an escalation persisted after the verify retries ran out (`graph-design.md` §4.2). |
| `acceptance` | `"accepted" \| "rejected" \| "not_required"` | code | `not_required` means `confirm_resolution` never ran. That covers every escalation that didn't follow a rejection, and resolutions run with `require_acceptance=false`. |
| `clarifications` | `list[ClarificationTurn]` | code | Copied from state: the questions and answers that shaped the result |
| `stats` | `RunStats` | code | `retrieval_rounds`, `tool_calls`, `verify_attempts`, `revisions`. Copied into the result because checkpoints can be wiped (`state-schema.md` §4). |
| `errors` | `list[str]` | code | `state.errors`, which are non-fatal |

### 2.1 `EscalationBlock`

| Field | Type | Author | Notes |
|---|---|---|---|
| `required` | `bool` | code | |
| `trigger` | `"rule" \| "evidence_exhausted" \| "verification_failed" \| "user_rejected" \| None` | code | Set by `escalate` from state (see precedence below) |
| `rule` | `str \| None` | code | `evidence_assessment.escalation_rule_hit` when `trigger == "rule"` |
| `target_queue` | `str \| None` | model | |
| `reason` | `str \| None` | model | One or two sentences for the human agent |
| `handoff_summary` | `str \| None` | model | What is known, what was tried and what is missing. Cites case IDs. |

The model only ever sees `EscalationDraft` (`target_queue`, `reason`, `handoff_summary`). `escalate` wraps that draft in the block. For a resolved case the block is `{"required": false}` with every other field `null`.

**Trigger precedence.** `escalate` checks these in order, and the first match wins:
1. `user_acceptance == "rejected"` → `user_rejected`
2. The latest verification failed and `verify_attempts ≥ max_verify_retries` → `verification_failed`
3. `evidence_assessment.escalation_rule_hit` is set → `rule`
4. Otherwise → `evidence_exhausted` (the retrieval and clarification budgets are spent)

---

## 3. Evidence entry format

### 3.1 `EvidenceEntry`

| Field | Type | Author | Notes |
|---|---|---|---|
| `case_id` | `str` | model picks it, code validates it | `HF-<row>` or `T-YYYYMMDD-<6 hex>`. It must exist in SQLite; unknown IDs are dropped during enrichment and logged to `errors`. |
| `source` | `"dataset" \| "agent_resolved" \| "customer_history"` | code | `dataset` for `HF-` IDs. A `T-` case is `agent_resolved` if it was resolved and indexed. Otherwise it's `customer_history`: an open, awaiting or escalated case reached through `get_customer_history`. |
| `subject` | `str` | code | Read from SQLite, never taken from the model |
| `summary` | `str` | model | One line of at most 200 chars: the problem → what resolved it (or → what was asked, or → where it was escalated) |
| `stance` | `"supports" \| "contradicts" \| "neutral"` | model | Relative to the current hypothesis |
| `answer_class` | `AnswerClass \| None` | code | For a dataset case, the label assigned at ingest. For a `T-` case, `resolved` gives `resolution`, `escalated` gives `escalation`, and any other status gives `None`. |
| `cluster_size` | `int ≥ 1` | code | For a dataset case, the cluster size from ingest. Always 1 for `T-` cases. |
| `similarity` | `float \| None` | code | Dense cosine between the case and the ticket query, taken from `retrieved_cases`. `None` if the case arrived only through `get_ticket_by_id` or customer history. |
| `approach` | `str \| None` | code | Filled by `persist_case`: the label of the `ApproachCluster` in the final `evidence_assessment` that contains this case, if any |

### 3.2 How an entry is built
1. `investigate`'s model emits `EvidenceItem` (`case_id`, `summary`, `stance`). This is its judgement.
2. Still inside `investigate`, code looks each `case_id` up in SQLite, adds `similarity` from `retrieved_cases`, and writes the enriched `EvidenceEntry` list into `state.evidence`. Unknown IDs are dropped and logged to `errors`.
3. `persist_case` fills in `approach` from the final assessment's clusters.

`verify` reads the enriched list, so it makes no database reads of its own.

### 3.3 Ordering and size
Entries are ordered `supports`, then `contradicts`, then `neutral`. Within each stance they sort by `cluster_size` descending, then `similarity` descending, with `None` last. The list holds at most 10 entries; enrichment keeps the first 10 after ordering.

### 3.4 Citation format in prose
Case IDs appear in square brackets: `[HF-10432]`, `[T-20260924-7f3a1c]`. Several IDs are written side by side: `[HF-10432][HF-877]`. No other citation syntax is used.

```python
CASE_ID = r"HF-\d+|T-\d{8}-[0-9a-f]{6}"
CITATION = re.compile(rf"\[({CASE_ID})\]")
```

The set of cited IDs is always taken from the text with this regex. There's no separate model-written list of cited IDs that could drift from the prose.

### 3.5 Grounding rules (checked by `verify`'s rule pass)
- **G1.** Every ID cited in `analysis`, `resolution` or `escalation.handoff_summary` appears in `evidence`.
- **G2.** When `status == "resolved"`, `resolution` cites at least one `supports` entry whose `answer_class == "resolution"`.
- **G3.** Every `contradicts` entry is cited in `analysis`, so conflicts are acknowledged and never silently dropped.

If any of these fails, `verification.passed = false` and the failure is listed as an issue with `recommended_action = "re_reason"`.

---

## 4. Confidence

### 4.1 What it means
**Confidence measures how strongly the historical evidence supports this result's analysis.** The meaning is the same for both outcomes. It is not the probability that the customer's issue is fixed, it is not the LangSmith quality score, and it is not the model's opinion of itself.

For escalations, this makes the number useful to the human who picks up the handoff:
- A rule-triggered escalation over a well-supported diagnosis can score **high** — provided the supporting evidence is itself `resolution`-class (§4.3 substantive is resolution-only, `rag-design.md` §5 Decision 3). "These cases also got escalated" is not a diagnosis and carries no weight; "these cases were all root-caused to the same misconfiguration, and policy requires a human to apply the fix" is, and does.
- An escalation because budgets ran out scores **low**. Nobody knows what's wrong yet.

### 4.2 Scale and bands
`value` is a float in [0, 1], rounded to 2 dp.

| Band | Range | Reads as |
|---|---|---|
| `high` | ≥ 0.75 | Several substantive cases agree, weighted by cluster size, and they closely match the ticket |
| `medium` | 0.50–0.74 | There is real support, but it's thin, partly contested or loosely matched |
| `low` | < 0.50 | The evidence is thin, conflicted or unverified. The resolution's wording must say so. |

`band` is a computed field derived from `value` and is never stored on its own.

### 4.3 Inputs
- The enriched `evidence` list (§3), plus `evidence_assessment.verdict`, `evidence_assessment.missing_slots` and whether verification passed. τ_rel comes from config.
- **Substantive** means `answer_class == resolution` — resolution-only, not `{resolution, escalation}` as an earlier draft had it. `rag-design.md` §5 (Decision 3) folded handoff answers ("we'll investigate and call you") into `escalation`, and a handoff carries no grounded fix — counting it as support would let "the historical neighbourhood mostly got handed off" read as evidence *for* a resolution, which is backwards. A `clarification_request` case shows someone asked a question, not what fixed the problem, and an open customer case (`answer_class = None`) has no answer yet. None of the three non-resolution kinds add support or contradiction weight.
- **Cluster weight** is `w = min(1 + log2(cluster_size), 5)`. A singleton weighs 1, a pair 2, a cluster of 4 weighs 3, a cluster of 8 weighs 4, and 16 or more weighs 5.
  - The weight grows sublinearly because the corpus is synthetic and templated. A cluster of 12 is stronger evidence than one case, but it isn't 12 independent confirmations (D3).
  - The cap stops a single large template cluster from carrying the score alone.
- `W_s` is the sum of `w` over substantive `supports` entries. `W_c` is the same sum over substantive `contradicts` entries.

### 4.4 Formula

```text
support   = 1 − exp(−W_s / 2.5)                     # 1 singleton → 0.33 · 3 singletons → 0.70 · W_s=6 → 0.91
agreement = W_s / (W_s + W_c)                        # 0 when both are 0
relevance = mean over the top-3 supporting entries by similarity (non-null) of
            clamp((sim − τ_rel) / (SIM_CEILING − τ_rel), 0, 1)
            # 0 if no supporting entry has a similarity

base      = support × (0.4 + 0.6·agreement) × (0.7 + 0.3·relevance)
penalty   = min(0.10 × len(missing_slots), 0.30)
cap       = lowest applicable of:
              no substantive support (W_s = 0)  → 0.20
              verdict == "insufficient"         → 0.45
              verdict == "conflicting"          → 0.60
              verification failed               → 0.30
value     = round(clamp(min(base − penalty, cap), 0, 1), 2)
```

**Why this shape**
- **The terms multiply** because support is necessary: agreement and similarity can't make up for missing support. Agreement and relevance have floors (0.4 and 0.7), so they can scale support down but never zero it.
- **Agreement is weighted by cluster size**, so 15 cases against 3 reads differently from 15 against 14 (D3).
- **Relevance uses dense cosine** on the τ_rel scale. That signal is independent of the model's stance judgements, so it checks whether the cases marked "supports" actually match the ticket.
- **Caps tie the number to the verdict.** An insufficient verdict can never read as medium, a conflicting one can never read as high, and a failed verification can never rise above low.
- **The missing-slot penalty** covers facts the approach depends on that nobody supplied.

**Constants.** These live as module constants in `autosupport/graph/confidence.py`: `K = 2.5`, `W_MAX = 5`, `SIM_CEILING = 0.92`, the two factor floors, the penalty step and ceiling, the four caps, and the band edges.
- They aren't in config because they define the scale. Changing one changes the meaning of every stored value, so a change should go through code review, not an environment variable.
- τ_rel does come from config, because `assess_evidence` shares it.
- `SIM_CEILING = 0.92`, measured, not a starting guess: it equals the clustering threshold T (`rag-design.md` §4/§9), so a retrieved case at or above it is — by the same standard used to canonicalise the corpus — a near-duplicate of the ticket. `τ_rel = 0.76` is the measured random-pair p95 (`rag-design.md` §9); the two together define a 0.16-wide relevance band, both anchored to the real `bge-small` similarity distribution rather than assumed.

### 4.5 Where it is computed
Confidence is computed **only in `verify`**, once per verification attempt:
1. Compute the value from the evidence, without the verification cap, and derive its band.
2. The fast-tier check receives that band and flags wording that overclaims it, such as "this will fix it" at `low`. This is what `graph-design.md` means by confidence calibration.
3. Once the verification result is in, apply the `verification failed` cap if it applies, and write `state.confidence`.

Every path to `confirm_resolution` and `persist_case` passes through `verify`, so no other node writes confidence. `resolve` and `escalate` never produce a number. Accepting or rejecting a resolution doesn't change it, and a revised draft gets a fresh value at its own `verify`.

### 4.6 `Confidence` model
`value` and `band` (computed), plus the components that produced them: `support`, `agreement`, `relevance` (each rounded to 3 dp), `penalty`, and `cap_reason ∈ {no_substantive_support, insufficient, conflicting, verification_failed} | None`. The components are persisted so that `show` and the evals can explain any score.

### 4.7 Worked examples (τ_rel = 0.76, SIM_CEILING = 0.92 — `rag-design.md` §9; substantive = resolution-only, §4.3)

Similarities below are realistic for this corpus, not illustrative round numbers: measured 3rd-best dense similarity to an actual canonical has p10 = 0.770, p50 = 0.829 (`rag-design.md` §9), so "supporting" similarities in the high-0.7s to high-0.8s are the typical case, not an edge case.

| # | Scenario | Supporting cluster sizes (resolution-class) | Contradicting | Top similarities | Other | support | agreement factor | relevance factor | **value** | band |
|---|---|---|---|---|---|---|---|---|---|---|
| A | Strong resolve | 8, 3, 1 | — | .91 .87 .83 | — | 0.952 | 1.000 | 0.906 | **0.86** | high |
| B | Contested resolve | 15, 1 | 3 | .85 .80 | — | 0.906 | 0.817 | 0.822 | **0.60** | medium (capped: conflicting) |
| C | Thin resolve | 1, 1, 1 | — | .80 .78 .77 | — | 0.699 | 1.000 | 0.744 | **0.52** | medium |
| D | Rule-hit escalation, well-diagnosed | 4, 2, 1 | — | .88 .84 .81 | — | 0.909 | 1.000 | 0.856 | **0.78** | high |
| E | Escalation, budgets spent | *(evidence is escalation-class only — not substantive)* | — | .77 | 2 missing slots, verdict insufficient | 0.000 | 0.400 | 0.719 | **0.00** | low (capped: no_substantive_support) |

**Example E is the sharpest illustration of resolution-only substantive**: even though a case was retrieved at reasonable similarity (.77, just above τ_rel), it's `escalation`-class — "this also got escalated" is not a diagnosis — so `W_s = 0` and the score floors at exactly 0.00, not a small positive number. An earlier draft of this table (before Decision 3) let escalation-class evidence count as substantive and scored this same shape of scenario at 0.05; the corrected version is a better fit for §4.1's stated meaning: this escalation genuinely has *no* grounded diagnosis behind it, and the number should say so plainly.

### 4.8 Calibration hook
`evaluation-design.md` (CP7) buckets runs by band and compares the buckets against the groundedness evaluator. High-band results should score higher than medium, and medium higher than low. If they don't, the constants are wrong: change them here and record why.

---

## 5. Validation (model validators on `CaseResult`)
The validators check structure only, at the database-write boundary:
- `status == "escalated"` if and only if `escalation.required`.
- If `escalation.required`, then `trigger`, `target_queue`, `reason` and `handoff_summary` are all non-null. If it isn't required, all of them are `None`.
- `acceptance == "accepted"` implies `status == "resolved"`.
- Evidence `case_id`s are unique and match `CASE_ID`.

The grounding rules in §3.5 belong to `verify`, not to the validators. A flagged escalation must still persist with its unresolved issues listed; a validator would crash `persist_case` instead.

---

## 6. Persistence and rendering
- `persist_case` builds the `CaseResult`, writes `cases.final_output = model_dump_json()` and sets `cases.status = status`.
- `service.show()` loads it back with `model_validate_json`. `band` appears in the dump and is ignored on load.
- `index_case` indexes a case only when `status == "resolved"` and `acceptance == "accepted"`. As a result, eval runs (`require_acceptance=false`) never add to the index.

---

## 7. Code sketch (lives in `autosupport/graph/state.py`; formula in `graph/confidence.py`)

```python
AnswerClass = Literal["resolution", "clarification_request", "escalation"]
EscalationTrigger = Literal["rule", "evidence_exhausted", "verification_failed", "user_rejected"]
CapReason = Literal["no_substantive_support", "insufficient", "conflicting", "verification_failed"]


class EvidenceItem(BaseModel):            # model-authored (state-schema §3)
    case_id: str
    summary: str
    stance: Literal["supports", "contradicts", "neutral"]


class EvidenceEntry(EvidenceItem):        # enriched by code in investigate
    source: Literal["dataset", "agent_resolved", "customer_history"]
    subject: str
    answer_class: AnswerClass | None
    cluster_size: int = Field(ge=1)
    similarity: float | None = None
    approach: str | None = None


class Confidence(BaseModel):
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


class EscalationDraft(BaseModel):         # model-authored, inside DraftResponse
    target_queue: str
    reason: str
    handoff_summary: str


class EscalationBlock(BaseModel):
    required: bool
    trigger: EscalationTrigger | None = None
    rule: str | None = None
    target_queue: str | None = None
    reason: str | None = None
    handoff_summary: str | None = None


class VerificationOutcome(BaseModel):
    passed: bool
    attempts: int
    unresolved_issues: list[str] = []


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
    clarifications: list[ClarificationTurn] = []
    stats: RunStats
    errors: list[str] = []
    # model_validator enforcing §5


# graph/confidence.py
def compute_confidence(
    evidence: list[EvidenceEntry],
    assessment: EvidenceAssessment,
    verification_passed: bool | None,   # None = before the check (§4.5 step 1)
    tau_rel: float,
) -> Confidence: ...
```

---

## 8. Example (illustrative; matches worked example A)

```json
{
  "ticket_id": "T-20260924-7f3a1c",
  "customer_id": "C-1042",
  "status": "resolved",
  "classification": {"queue": "Technical Support", "type": "Incident", "priority": "high",
                     "tags": ["network", "nas"], "rationale": "…", "neighbor_agreement": 0.8},
  "evidence": [
    {"case_id": "HF-10432", "summary": "QNAP shares unreachable after firmware update → re-enabled SMB2",
     "stance": "supports", "source": "dataset", "subject": "NAS shares not visible after update",
     "answer_class": "resolution", "cluster_size": 8, "similarity": 0.91, "approach": "re-enable SMB2"},
    {"case_id": "HF-877", "summary": "SMB clients dropped after NAS update → protocol range reset",
     "stance": "supports", "source": "dataset", "subject": "…",
     "answer_class": "resolution", "cluster_size": 3, "similarity": 0.87, "approach": "re-enable SMB2"},
    {"case_id": "T-20260911-a41c09", "summary": "Same customer, same NAS → fixed by SMB2 re-enable",
     "stance": "supports", "source": "agent_resolved", "subject": "…",
     "answer_class": "resolution", "cluster_size": 1, "similarity": 0.83, "approach": "re-enable SMB2"}
  ],
  "analysis": "The firmware update resets the minimum SMB version; eleven historical cases [HF-10432][HF-877] and this customer's own earlier ticket [T-20260911-a41c09] were resolved by re-enabling SMB2.",
  "resolution": "1. Open Control Panel → Network & File Services → Win/Mac/NFS [HF-10432]. 2. …",
  "escalation": {"required": false, "trigger": null, "rule": null,
                 "target_queue": null, "reason": null, "handoff_summary": null},
  "confidence": {"value": 0.86, "band": "high", "support": 0.952, "agreement": 1.0,
                 "relevance": 0.687, "penalty": 0.0, "cap_reason": null},
  "verification": {"passed": true, "attempts": 1, "unresolved_issues": []},
  "acceptance": "accepted",
  "clarifications": [],
  "stats": {"retrieval_rounds": 1, "tool_calls": 3, "verify_attempts": 1, "revisions": 0},
  "errors": []
}
```

An escalated result has the same shape. Worked example E, for instance, has `"status": "escalated"`, `"escalation": {"required": true, "trigger": "evidence_exhausted", …}`, `"confidence": {"value": 0.00, "band": "low", "cap_reason": "no_substantive_support", …}` and `"acceptance": "not_required"`.

---

## 9. Open — all resolved in `rag-design.md`'s full rewrite
- ~~Calibrate `SIM_CEILING` alongside τ_rel.~~ → `τ_rel = 0.76` (measured random-pair p95), `SIM_CEILING = 0.92` (equal to the clustering threshold T) — `rag-design.md` §9.
- ~~Decide how `similarity` is filled for results found only by the lexical arm.~~ → filled for **every** fused candidate, dense-only or lexical-only, as cosine(ticket embedding, candidate embedding fetched from Chroma) — never `None` (`rag-design.md` §7). `RetrievedCase.similarity` (`state-schema.md`) is a required field, not optional.
- ~~`merge_cases` keeps the maximum fused `score` across queries...~~ → switched: `merge_cases` now keys on `similarity`, which is on one fixed scale regardless of which query found the case; `score` (the fused RRF value) stays ranking-only within a single query's own results (`state-schema.md`, `rag-design.md` §11).
