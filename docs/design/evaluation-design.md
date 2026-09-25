# Evaluation Design — offline LangSmith evaluation

> **Status:** v3 · rebuilt at CP8 as a five-pattern dataset (decisions.md D23; earlier results
> in D17–D21). **Code:** `evals/` (`select_examples.py`, `dataset.py`, `evaluators.py`,
> `run.py`); `autosupport eval` → `service.run_eval`.
> **Companion docs:** `output-schema.md` §3.5 (the in-graph `verify` rules — *not* this),
> `rag-design.md` (retrieval being measured), `case-persistence.md` §5.1 (the index policy).

---

## 1. Evaluation is not `verify`

| | `verify` (graph node) | Offline evaluation (this doc) |
|---|---|---|
| When | Inside every run, before `persist_case` | After a batch of runs, in LangSmith |
| Purpose | Gate *this* draft: loop back or escalate | Measure the *system* across a dataset |
| Sees | Retrieved snippets, the draft | Full case records, gold labels, the whole final state |
| Effect | Changes the run's path and confidence | None on any run — scores only |
| Model | Fast tier | Code, plus one judge (`AUTOSUPPORT_JUDGE_MODEL`) for groundedness |
| Names in code | `verify`, `VerificationResult`, `verification` | `evals/`, feedback keys below |

Evaluator code never imports `graph/verification.py`, and no feedback key contains "verif".

---

## 2. Dataset: five real tickets, one per behaviour pattern

There is no `brief.md`, so the dataset is built from the ingested corpus itself
(`evals/select_examples.py` → `evals/examples.jsonl`, committed). This section is its
documentation: which tickets were chosen and why each fits its pattern.

**No example retrieves itself.** A ticket that is itself a retrievable canonical would find
its own answer at similarity 1.0 and the eval would measure nothing. So every input is a
**cluster member**, a real ticket merged into a canonical at ingest and never searchable
itself, or, for one ticket, a row outside the index with no indexed near-duplicate. The
member's canonical, or the named neighbours, are the **known-good evidence** retrieval is
checked against. All inputs are excluded from future ingests (`holdout_ids`).

**Labels** (queue, type, priority, tags) are the dataset's own. Measured on the 2,000-row index
the eval runs against (relevant = cosine ≥ 0.76):

| Example | Ticket | Known-good evidence | Why it fits the pattern | Expected behaviour |
|---|---|---|---|---|
| `clean_resolution` | **HF-9322**, "MongoDB 4.4 Integration Solutions" (Product Support / Request / medium) | HF-2958 (its canonical) | 8 of its 9 relevant neighbours are resolution-class, the strongest consistent resolution neighbourhood in the index. *Caveat:* in the 2,000-row index every resolution cluster has size 2, so "high `cluster_size`" isn't available; strength comes from neighbourhood agreement instead | resolved |
| `clarification_needed` | **HF-49750**, "Assistance with Drop Issue" (Customer Service / Incident / medium) | HF-12458 | A 12-word report ("unusual decrease in engagement metrics… possibly related to recent algorithm updates"): no product, metric or timeframe. 9 of its 10 relevant neighbours were answered with clarification requests | asks one question and pauses |
| `conflicting_evidence` | **HF-61377**, "Investment Tools Compatible with Evernote" (resolution) | HF-54771, HF-50033, HF-43855 | Its relevant resolution neighbours solve the same-looking request ("which analytics tools for investment optimisation") differently: HF-43855 names concrete tools (Tableau, Power BI, Python), HF-50033 (Xero) defers to a call-back, HF-54771 lists Evernote-specific tools. The corpus is synthetic and has no sharply contradictory fixes; this is the largest measured divergence between near-identical requests' answers | an assessment reaches `conflicting` |
| `escalation_worthy` | **HF-7725**, "Problem with Financial Tools Integration" (Product Support / Incident / **high**; tags Bug, Performance, **Outage, Disruption**) | HF-8283 | A financial firm's tool integrations fail after an upgrade; they already rebooted, cleared caches and checked the API connections. The historical answer escalated | escalated |
| `memory_and_new_case` | **HF-3745** then **HF-59267**, same synthesized customer | the first ticket's own `T-` id, and HF-2958 | Ticket 1 asks for options to integrate MongoDB 4.4 into a scalable SaaS project-management platform. It sits on the HF-2958 neighbourhood, which resolves reliably, and "MongoDB 4.4" is a fact the memory policy keeps. It is resolved and **accepted**, so it's indexed. Ticket 2 asks for documentation on the same integration (cosine 0.920 to ticket 1) and is not in the index. *The first pick, HF-22907 (Docker Django security), escalated in the dry run: the draft added security detail no cited case contained, and `verify` rejected it twice.* | ticket 2 loads the remembered facts, and retrieves ticket 1 as `agent_resolved` |

**The escalation example exposes a rule gap on purpose.** `critical_high_stakes` requires
priority `critical`, but the dataset's priorities stop at `high`. So on this corpus, HF-7725 can
only escalate through the other rules (escalated neighbourhood, a person-only action, or the
investigator's own flag). The example measures whether the agent gets there anyway.

**Each example in `examples.jsonl`:** `example_id` (the pattern name), `inputs` (`subject`,
`body`, and `first_ticket` for the memory example), `reference` (the ticket id, the reason
above, the dataset labels, `known_good`, `allowed_tools`, `expect`), `holdout_ids`.

---

## 3. The run

- **Tracing is off by default** (`LANGSMITH_TRACING=false`). `autosupport eval` switches it on
  for itself only; `scripts/smoke_langsmith.py` is the only other place that does.
- **Per example:** `thread_id = f"eval:{run_id}:{example_id}"`, and a fresh customer
  `EVAL-{run_id}-{example_id}`, so no memory leaks between examples or between runs. The
  acceptance step is off.
- **The memory example runs two tickets in one target call:**
  1. Ticket 1 runs with acceptance on and is accepted, so it is indexed like any accepted
     resolution.
  2. Ticket 2 runs, and the target records whether memory was loaded and whether ticket 1 was
     retrieved as `agent_resolved`.
  3. Ticket 1 is then **un-indexed** (Chroma, FTS5, `indexed_at`), so an eval run leaves the
     corpus exactly as it found it.
- **Sequential** (`max_concurrency=1`): one SQLite checkpointer connection and store.
- **Trace budget:** each example is one pipeline trace, since the memory example's two
  tickets nest inside one target call. The groundedness judge adds one trace per *resolved*
  example. Evaluator tracing is disabled, so the four code evaluators add none. **Estimate:
  5 pipeline traces + ≤ 5 judge traces ≤ 10.**

---

## 4. Evaluators

Four of the five are plain code; only groundedness calls a model.

| Feedback key | Kind | What it measures | Score |
|---|---|---|---|
| `classification_accuracy` | Code | `triage`'s queue, type and priority (exact) and tags (Jaccard overlap) vs the dataset's labels | mean of the four; each also reported separately |
| `tool_usage_correctness` | Code | Every call in `tool_log` against the example's expected tool set: only allowed tools (escalating a ticket that should resolve or ask is wrong), every call succeeded, no identical repeat, no invented case id | share of checks passed (1.0 with no calls) |
| `retrieval_relevance` | Code | Share of the known-good case ids found in the top 10 retrieved cases | found / known-good |
| `response_groundedness` | LLM judge — **the only evaluator that traces** | Each factual claim of a resolution checked against the full text of the cases it cites | supported / all claims; resolutions only |
| `pattern_behaviour` | Code | Did the agent do what the pattern calls for (`expect`: outcome, verdict, memory loaded, first ticket retrieved)? Covers REQUIREMENTS §8 "overall behaviour" | share of expectations met |

The judge returns a per-claim verdict and code computes the score, never a model-reported
number (D12).

**Known limits.** Five examples are a behaviour check, not a statistical measurement: one
flipped outcome moves a mean by 0.2. The judge is the same model as the drafter with the
default models, so it may favour its own phrasing. Queue and priority labels are noisy, which
caps classification.

---

## 5. Reading the results

| Weak key | First place to look |
|---|---|
| `pattern_behaviour` | `assess_evidence` rules (`graph/assessment.py`), the escalation rules, the thin-ticket rule |
| `retrieval_relevance` | Retrieval parameters (`τ_rel`, RRF k, MMR λ), query text, the index policy (memory example) |
| `response_groundedness` | The `customer_response` skill / `resolve` prompt, or the evidence passed to it |
| `tool_usage_correctness` | The `investigation` skill / tool docstrings |
| `classification_accuracy` | The `triage` vote, or the label-noise ceiling |

---

## 6. Results (first run, 2026-09-25, experiment `autosupport-20260925004516`)

Default models (GPT-OSS 120B for both tiers and the judge), on the 2,000-row index.

| Example | Outcome | Pattern met | Retrieval | Tools | Groundedness |
|---|---|---|---|---|---|
| clean_resolution | resolved | yes | 1.0 | 1.0 | 0.8 (8/10 claims) |
| clarification_needed | asked one question | yes | 1.0 | 1.0 | — |
| conflicting_evidence | escalated | **no** — verdict was `insufficient`, not `conflicting` | 1.0 | 1.0 | — |
| escalation_worthy | escalated | yes | 1.0 | 1.0 | — |
| memory_and_new_case | resolved | yes: memory loaded, ticket 1 retrieved as `agent_resolved` (0.920) | 1.0 | 1.0 | 1.0 (12/12) |

Classification accuracy was 0.74. Type and priority were 1.0; queue was 0.4 and tags 0.57, the
known label noise.

**Traces: 7** (5 pipeline, 2 judge, one per resolved example), matching the estimate of
5 + (number resolved).

**Reading these honestly.**
- **Retrieval 1.0 is partly guaranteed by construction.** A cluster member's canonical is by
  definition a near-copy (cosine ≥ 0.92), so for the four member-based examples, finding it
  mostly confirms that search returns near-duplicates. The memory example is the non-trivial
  retrieval check: ticket 2 isn't in the index, and it found a case that didn't exist until
  ticket 1 was accepted.
- **The conflicting pattern is a genuine miss.** The investigator put the divergent answers in
  one approach cluster, so no rival fixes existed for the verdict to see. Either the corpus
  lacks real contradictions, or clustering by approach is too coarse; one example can't tell
  which.
- **Five examples check behaviour, not a statistic.** The dry runs showed the same example
  flip between runs: the memory example resolved in one run and escalated in another.
