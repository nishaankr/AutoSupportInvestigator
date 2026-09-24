# Evaluation Design — offline LangSmith evaluation

> **Status:** v1 · CP7. **Code:** `evals/` (`select_examples.py`, `dataset.py`,
> `evaluators.py`, `run.py`); `autosupport eval` → `service.run_eval`.
> **Companion docs:** `output-schema.md` §3.5 (the in-graph `verify` rules — *not* this),
> `rag-design.md` (retrieval being measured), `case-persistence.md` §5.1 (why eval runs
> don't grow the corpus).

---

## 1. Evaluation is not `verify`

| | `verify` (graph node) | Offline evaluation (this doc) |
|---|---|---|
| When | Inside every run, before `persist_case` | After a batch of runs, in LangSmith |
| Purpose | Gate *this* draft: loop back or escalate | Measure the *system* across a dataset |
| Sees | Retrieved snippets, the draft | Full case records, gold labels, the whole final state |
| Effect | Changes the run's path and confidence | None on any run — scores only |
| Model | Fast tier | A separately configured judge (`AUTOSUPPORT_JUDGE_MODEL`), plus code-only metrics |
| Names in code | `verify`, `VerificationResult`, `verification` | `evals/`, `*_evaluator`, `EvalSummary`, feedback keys below |

Evaluator code never imports `graph/verification.py`, and no feedback key contains "verif".
A draft can pass `verify` and still score low on `response_groundedness`: `verify` only
checks against snippets, and it grades the draft it was given, not whether the right
cases were retrieved.

---

## 2. Dataset

**Source.** The five example tickets from the brief (`evals/brief_examples.json`, added
when the brief text is available: no gold labels, so only the label-free evaluators score
them) plus **15 held-out corpus tickets** (`evals/examples.jsonl`, generated once by
`evals/select_examples.py`, committed).

**Held-out, not in-corpus.** An eval ticket that is itself indexed retrieves its own answer
at similarity 1.0 and measures nothing. Candidates are drawn from the English subset
**minus every row in `dataset_tickets`**. A candidate is **rejected if its nearest indexed
neighbour has cosine ≥ 0.92** (the clustering threshold `T`, `rag-design.md` §4): above that
it is a near-duplicate, and retrieval would be finding a copy. `ingest/load.py` drops every
`examples.jsonl` id from future ingests, so the holdout survives a full-corpus ingest.

**Stratification.** 5 examples per heuristic `answer_class` of the gold answer
(`resolution`, `escalation`, `clarification_request`; residue skipped, no LLM call),
choosing a different queue where possible. Subject present, body 120–700 characters. The
seed is fixed.

**Each example:**
- **inputs:** `example_id` (= `HF-<row>`), `subject`, `body`
- **reference outputs:** `queue`, `type`, `priority`, `answer_class`, `answer`: the
  dataset's own labels and historical answer, which are free ground truth

**Sync.** `evals/dataset.py` creates the LangSmith dataset (default `autosupport-eval`) if
it is missing, and adds any example whose `example_id` isn't already in it.

---

## 3. The run

- **One experiment per `autosupport eval`**, with `run_id = <UTC yyyymmddHHMMSS>`.
- **Per example:**
  - `thread_id = f"eval:{run_id}:{example_id}"`
  - `customer_id = f"EVAL-{run_id}-{example_id}"`, a fresh customer, so no memory leaks
    between examples or between runs
  - `require_acceptance = false`
- **Sequential** (`max_concurrency=1`). The runs share one SQLite checkpointer connection
  and SQLite store.
- **No simulated customer.** A run that pauses at `ask_user` is recorded as
  `asked_clarification`. There is no simulated customer to answer it, so it is not resumed.
- **Nothing is indexed.** `require_acceptance=false` means `acceptance = not_required`, so
  eval runs never index into the corpus they are measured against (case-persistence §5.1).
- **Trace metadata.** Every graph invocation (eval and normal) carries `ticket_id`,
  `customer_id` and `thread_id` metadata, so a score links to its trace.
- **Target output.** The target returns a compact view of the final graph state:
  - status and outcome
  - classification
  - top retrieved cases
  - `tool_log`
  - resolution/analysis and cited ids
  - confidence
  - pending question

---

## 4. Evaluators

| Feedback key | Kind | What it measures | Score |
|---|---|---|---|
| `response_groundedness` | LLM judge (`AUTOSUPPORT_JUDGE_MODEL`) | Each factual claim in the customer-facing resolution is checked against the **full** text of the cases it cites, fetched from SQLite rather than snippets | supported claims / all claims. **Resolved tickets only**: an escalation's holding reply is a code template with no claims (D19), and a judge scored those 0 |
| `retrieval_relevance` | LLM judge (`AUTOSUPPORT_JUDGE_MODEL`) | Each of the top 5 retrieved cases by similarity is graded `relevant` (same problem, its answer directly helps), `partial` (same kind of request, but a different product or platform) or `irrelevant`. The judge is shown the ticket's real historical answer as a reference. Graded rather than binary, because a binary judge scored exact-request, different-platform neighbours as 0 during calibration | mean of 1 / 0.5 / 0 over the 5 |
| `tool_usage_correctness` | Code | Over `tool_log`: (a) every call succeeded; (b) no identical call repeated; (c) every case id passed to `get_ticket_by_id` exists, i.e. no invented ids | mean of the checks that apply (1.0 when no tools were called) |
| `classification_accuracy` | Code | `triage`'s queue/type/priority vs the dataset labels | mean of 3 exact matches. Also `classification_queue`/`_type`/`_priority`. Skipped without labels |
| `outcome_appropriateness` | Code | REQUIREMENTS §8 "overall agent behaviour": gold `resolution` → resolved, `escalation` → escalated, `clarification_request` → asked a clarification | 1 / 0. Skipped without labels |

The judges use `.with_structured_output(..., method="json_schema")` (D13) and must return
a per-item verdict (per claim, per case). The score is computed in code from those
verdicts, never taken as a model-reported number (same principle as D12).

**Known limits.**
- **Noisy classification labels.** The dataset's queue labels are noisy (CP6 observed one
  Redis/Rails question filed under "Human Resources"), so `classification_accuracy` has a
  ceiling well below 1.
- **Judge bias.** The groundedness judge is the same model family as the drafter
  (main tier), so it may be biased towards its own phrasing.
- **Small sample.** 15 examples: differences of a few points between runs are noise.

---

## 5. Reading the results

`autosupport eval` prints each key's mean plus a per-example table, and links to the
experiment. Which layer each low score points at:

| Weak key | First place to look |
|---|---|
| `retrieval_relevance` | Retrieval parameters (`τ_rel`, RRF k, MMR λ, query text), `rag-design.md` |
| `response_groundedness` | The `customer_response` skill / `resolve` prompt, or evidence passed to it |
| `tool_usage_correctness` | The `investigation` skill / tool docstrings |
| `classification_accuracy` | The `triage` skill, or the label noise ceiling |
| `outcome_appropriateness` | `assess_evidence` thresholds and escalation rules, and the confidence formula only insofar as it gates `confirm_resolution` |
