# Decision Record

> Every non-obvious choice, with the reasoning behind it and what was rejected.
> This exists so the reasoning survives into implementation and into the README's
> design discussion. Do not reverse a decision here without saying so explicitly.

---

## D1 — Deterministic ticket IDs, assigned at ingest

The source dataset has no ID column. Dataset rows get `HF-<row_index>`; agent-created
tickets get `T-YYYYMMDD-<6 hex>`.

**Why:** the required output schema demands `Evidence: IDs/summaries of historical cases
used`. Without stable IDs that field cannot be populated at all.

**What they are not for:** user-facing lookup of archive tickets, which has no value here.
`get_ticket_by_id` earns its place differently — see D7.

---

## D2 — Customer identity begins at runtime, not in the archive

The historical corpus stays anonymous. No synthetic customer population is assigned over it.
`customer_id` arrives with each newly submitted ticket and lives in the `cases` and
`customers` tables. Customer history and long-term memory read only from there.

**Why:** randomly assigning customers across 28K historical tickets would manufacture
relationships that don't exist, which the agent would then reason over as though they were
real. Fabricated context is worse than absent context.

**Supported by the brief:** "New/open cases should be used as customer history" — the
assignment itself locates customer history in the growing case DB, not the archive.

**Rejected:** random customer assignment; deriving pseudo-customers by clustering on
queue/tags/business type.

**Limitation to state in the README:** memory and history demos exercise only
agent-created tickets.

---

## D3 — Canonicalisation with frequency counts

Near-duplicate clusters collapse to one canonical case carrying `cluster_size`. Only
canonicals are embedded and retrievable by similarity; members stay in SQLite, reachable
through `canonical_of`.

**Why:** the dataset is synthetic and heavily templated — near-identical tickets appear
under different subjects. Uncontrolled, top-k returns the same case five times wearing
different hats, which silently defeats the brief's "compare evidence across multiple cases
and reason about conflicts" requirement.

**Why counts rather than plain dedup:** collapsing with a count converts the liability into
an evidence-strength signal. "Twelve historical cases resolved this way" is a materially
stronger grounding claim than one case. It feeds the confidence score, and it gives conflict
detection a weight dimension — clusters of 15 vs 3 disagreeing is a different situation from
15 vs 14.

**Method, as measured (`rag-design.md` §4):** star (leader) clustering over a k-NN graph, not
connected components — a full-corpus measurement found connected components collapses the
corpus into a handful of giant clusters (74.6% of records in one component at the chosen
threshold), because a similarity graph chains transitively through intermediate near-matches.
Star clustering requires every member to be similar *to the canonical itself*, which doesn't
chain. A candidate only joins when, beyond cosine similarity ≥ T = 0.92, a **guard** also
holds: matching `answer_class`, answer-text similarity ≥ 0.85, and no conflicting named
entity. The guard exists because measurement found near-duplicate *problems* routinely get
different *answers* (30% of pairs at similarity 0.90–0.92 have answer-similarity < 0.80) —
without it, `cluster_size` would certify agreement that never happened, exactly the failure
this decision exists to prevent, just relocated from exact- to near-duplicate detection.
Full thresholds, the sweep that produced them, and the argued-and-resolved case against T=0.92
are in `rag-design.md` §4.

**Rejected:** exact-hash dedup alone (misses templated variation — used only for the small
set of *true* exact duplicates, a dataset-generation merge artifact, not the general case);
MinHash/LSH (viable, but embedding-threshold clustering reuses machinery already present);
retrieval-time MMR *alone* (treats the symptom, discards the signal). MMR is still used, on
top of this.

---

## D4 — Hybrid retrieval: dense + BM25, fused with RRF

Chroma supplies the dense arm; a SQLite FTS5 virtual table supplies BM25. Results are fused
with Reciprocal Rank Fusion, then diversified with MMR. RRF and MMR are written in-repo.

**Why, corrected after measurement:** an earlier draft of this decision attributed the
flattened dense similarity to greeting boilerplate ("Dear Customer Support Team, I hope this
message reaches you well"). Measured: only 17.9% of bodies actually open with a greeting —
concentrated almost entirely in one small dataset-version slice (92% of `version=51`'s 551
rows, 10–22% elsewhere) — and stripping greetings barely moves the random-pair similarity
distribution. **The real cause is corpus-wide style homogeneity**: every ticket is synthetic,
single-domain (IT/software support) and generated in the same narrow register, which flattens
embeddings independent of any specific boilerplate phrase. The conclusion is unchanged: the
discriminating signal is entities — `QNAP NAS`, `Aruba 2530`, `macOS 15`, `Crucial MX500`,
`Kubernetes` — which lexical search captures precisely and embeddings blur. Measured on the
full corpus (`rag-design.md` §7): dense-only entity@10 is 0.202 against BM25's 0.394, and a
lexical-only entity hit survives RRF fusion into the top 10 at 59% (k=10).

**Why FTS5 specifically:** it ships inside SQLite, so the lexical arm costs no new
dependency and no new store, and it indexes the same rows the tools already query.

**Rejected:** dense-only (loses the entity signal); `rank_bm25` (a dependency wrapping what
the standard library already provides, holding a second in-memory copy of the corpus with no
persistence).

---

## D5 — `answer_class` assigned at ingest, by heuristics with a validated sample

Every historical answer is typed `resolution`, `clarification_request` or `escalation`.
Method: a **sentence-level** regex/heuristic first pass (not whole-answer pattern matching —
an answer routinely mixes a real fix with an unrelated trailing sentence, and sentence-level
scoring lets the strongest signal win instead of the two cancelling out); precision measured
against a hand-labelled sample of 200, stratified by predicted class from a held-out test
pool; a fast-tier LLM applied only to the ~8–10% residue the heuristic can't confidently
call. Handoff answers ("we'll investigate and call you") are classed as `escalation`, not a
fourth category — matching `graph-design.md`'s escalation rule, which already reads
"resolved by escalation *or handoff*."

**Why the field exists:** a large share of the dataset's `answer` values are not resolutions
at all — they are requests for more information ("please specify the model and firmware
version"). Grounding a resolution on five retrieved cases that all say "send us your logs"
produces confident nonsense. Conversely, a retrieved neighbourhood dominated by
`clarification_request` is the strongest available signal that *this* ticket needs the
clarification route — it tells the agent what to ask and that asking is correct. Measured:
only ~7–12% of historical answers are genuine resolutions (`rag-design.md` §5.3), so
grounding is scarce by construction, and the `resolution_only` second-pass retrieval variant
(`rag-design.md` §10) exists specifically to compensate.

**Why this method:** running a model over 28K rows is slow and expensive for a labelling
task that patterns handle well. The sampled precision figure is reported in the README as
measured methodology rather than an unverified claim — which is itself worth more than a
marginally better classifier. **Labelled by Claude, in this session — not a human hand-label**
— disclosed as such rather than implied otherwise, since D5's whole point is that the
precision figure is honest methodology.

**Measured precision** (`rag-design.md` §5.4, full detail and the labelled fixture at
`tests/fixtures/answer_class_validation.csv`): resolution 0.839 (n=31), escalation 0.840
(n=50), clarification_request 0.885 (n=61). Clarification clears the 0.85 bar set going in;
resolution and escalation land at it within measurement noise (95% CIs comfortably contain
0.85) rather than strictly above the point estimate — reported as-is rather than pushed
further, because continued pattern-fitting against the same fixed 200 rows showed the
signature of overfitting the sample (one fix swinging resolution precision up 14 points while
costing escalation precision a comparable amount), not genuine improvement.

**Rejected:** LLM over the full corpus (constraint 5 forecloses this outright, and the
partial-concession/revision trigger for reopening that constraint is recorded in
`rag-design.md` §5.7); heuristics with no validation step; whole-answer (not sentence-level)
pattern matching, an earlier draft's approach that this rewrite replaced once validation
against real answers showed it conflating a real fix with an unrelated trailing sentence.

---

## D6 — English subset only

Filter `language == "en"` at load. Mandated by the brief. The source file also merges two
dataset generations via the `version` column; measured: `language == "en"` alone (no version
filter — all four version values carry English rows against an identical schema) gives
28,261 rows, matching the brief's "~28K." After exact-duplicate dedup (a dataset-generation
merge artifact — every one of 4,460 duplicate pairs is one `version=None` row paired with an
otherwise-identical `version=400` row) the working corpus is **23,801 distinct records**.
Full detail in `rag-design.md` §1.

---

## D7 — Retrieval returns summaries; full cases are fetched by tool

Search returns `ticket_id` + subject + `answer_class` + tags + snippet + `cluster_size`.
The agent scans those, decides which are genuinely on-point, and calls `get_ticket_by_id`
for the full body and answer of just those.

**Why:** small-to-big retrieval keeps the context window and checkpoints small, and it gives
`get_ticket_by_id` a real reason to exist — a genuine model decision rather than a tool
added to satisfy a count. It also serves the runtime case: newly persisted cases are looked
up the same way.

---

## D8 — Deterministic lifecycle wrapped around an agentic core

Fixed workflow edges for intake, persistence, verification and indexing; a ReAct tool loop
for investigation; corrective-RAG grading for evidence; reflection for pre-commit checking;
`interrupt()` for human input.

**Why:** the brief requires that certain steps *always* happen — every ticket persisted as an
open case, every final case persisted and indexed, an evidence check before output. Those
cannot be left to model discretion. Everything genuinely discretionary — which tools, which
queries, when evidence suffices — is left to the model.

**Rejected:** `create_react_agent` alone (can't guarantee persistence, verification or
indexing, and hides the graph design, which is itself assessed); supervisor with sub-agents
(one domain, one ticket at a time — coordination cost with no added capability);
plan-and-execute (tickets are short; a written plan adds a model call without changing what
gets retrieved).

---

## D9 — SQLite checkpointer, not in-memory

`SqliteSaver` → `data/checkpoints.sqlite`, a separate file from the application DB.

**Why:** the clarification demo is `autosupport new` → **process exits** → `autosupport
resume`. An in-memory checkpoint would not survive that, and surviving it is the entire
point of the interrupt/resume requirement. Separate files mean threads can be reset without
destroying resolved cases or customer memory.

---

## D10 — CLI, with `service.py` as a UI-agnostic seam

Typer CLI. `service.py` returns Pydantic objects; `cli.py` is a pure renderer; every command
supports `--json`.

**Why:** the brief accepts a CLI or an API and explicitly discourages spending time on UI.
A CLI also demonstrates interrupt/resume across separate processes more clearly than an API
would. The seam means a FastAPI layer and a web frontend can be added later as roughly one
route per command, with no changes to the graph.

---

## D11 — Two model tiers, and the verifier differs from the drafter

Main tier for investigation, resolution and escalation drafting. Fast tier for triage,
evidence grading, query rewriting, verification and memory extraction.

**Why:** the fast-tier nodes produce structured output against fixed schemas and some run
once per loop iteration, so they dominate call volume while benefiting least from capability.
Running `verify` on a different tier from the drafter is a cheap partial guard against a
model agreeing with its own output.

---

## D12 — Confidence is computed, not model-reported

`confidence` is produced by a deterministic formula in `graph/confidence.py`, run once per
attempt inside `verify`. `resolve` and `escalate` never emit a number; the drafting model
only ever writes prose. Full formula, inputs and worked examples in `output-schema.md` §4.

**Why:** an LLM self-reported confidence is uncalibrated and can't be falsified — nothing
stops the model from writing 0.9 next to a single weak match. A formula over the evidence
can be inspected, reproduced outside the graph, and calibrated against the LangSmith
groundedness evaluator at CP7 (`output-schema.md` §4.8). It also gives `cluster_size` a
concrete downstream use, which is what D3 promised when canonicalisation was chosen.

**Rejected:**
- *LLM self-report* — the model already originates every other number that could be gamed
  (similarity scores, cluster sizes); confidence is the one number that should audit those,
  not restate the model's own opinion of itself.
- *Formula + bounded LLM adjustment* — adds a second, harder-to-reproduce source of variance
  for a marginal gain, and complicates the calibration hook: a bad calibration run couldn't
  tell whether the formula or the adjustment needed fixing.

---

## D13 — CP3 vertical slice: scoped deviations and a structured-output finding

CP3 (`intake → (load_memory ∥ retrieve_initial) → triage → resolve → persist_case → END`)
cuts the full graph-design.md graph down to five nodes. Four places where the full-design
docs don't (yet) fit that shape were flagged and decided before writing code, not resolved
silently:

- **Q1 — `confidence`/`verification`.** Both are `Optional` on `CaseResult` until CP5.
  There's no `verify` node yet, and `output-schema.md` §4.5 computes confidence only inside
  it. `graph/confidence.py` is written and unit-tested now (against the §4.7 worked examples)
  even though nothing calls it until CP5 wires `verify` in.
- **Q2 — `evidence` without `investigate`.** `resolve`'s structured output includes the
  `EvidenceItem` list directly; `graph/evidence.py::enrich()` does the same code-side
  enrichment `investigate` will call at CP4. `enrich()` itself isn't CP3-specific.
- **Q3 — `ticket_id` vs `thread_id`.** `service.new_ticket()` generates `ticket_id` and
  builds `thread_id` *before* invoking the graph — the thread has to exist to invoke it at
  all. `intake` only mirrors `config.thread_id` into state. This also fixed a self-
  contradiction between `state-schema.md` §2.1 ("`intake` assigns `ticket_id`") and §4.2
  ("the CLI builds `thread_id`") that predated CP3. Full detail: `case-persistence.md` §6.
- **Q4 — `cases`/`customers` schemas.** Written up front as `case-persistence.md` and
  `memory-design.md`, reviewed before any code, rather than inferred implicitly while
  writing `intake`/`load_memory`/`persist_case`.

Full list, including the smaller flagged items (F1–F11: `EvidenceEntry.cluster_size`,
`datetime.utcnow` deprecation, `neighbor_agreement` computed by code not the model, etc.),
is in the CP3 plan this checkpoint was built from.

**Structured-output reliability finding, applied everywhere `.with_structured_output` is
called:** `main_llm()`/`fast_llm()` (`claude-sonnet-5`, `claude-haiku-4-5`) both run with
reasoning enabled by default. `langchain-anthropic`'s default `method="function_calling"`
doesn't force the tool call when reasoning is on (its own docstring says so), and reproduced
directly against the live API: on a nontrivial prompt, the model would reliably return a
malformed or incomplete tool call — a required field missing, or one field's text bleeding
into another. Claude's native structured-output feature, `method="json_schema"`, doesn't
depend on forced tool choice and was reliable in the same repro, with the exact same schema
and prompt, every time. Every `.with_structured_output(...)` call in the graph passes
`method="json_schema"` (`triage.py`, `resolve.py`, and every LLM node CP4+ adds). This isn't
a deviation from CLAUDE.md's "every structured LLM step uses `.with_structured_output`" rule
— it's the same call, with the one keyword argument that makes it actually reliable for
these two models.

---

## D14 — CP4: ReAct loop, tools and skills — scoped choices

- **`route_after_investigate` exits to `resolve`, not `assess_evidence`** (graph-design.md
  §4.2 names `assess_evidence`, which is CP5). Same tool-vs-stop logic; CP5 swaps the target.
- **`investigate` uses two LLM shapes.** The tool-deciding turn is `bind_tools` (a ReAct turn
  isn't a fixed-schema step, so CLAUDE.md's `.with_structured_output` rule doesn't apply to
  it); once the model stops calling tools, one structured call (`json_schema`, D13) extracts
  `hypothesis` and `evidence`, enriched by `graph/evidence.enrich`. `resolve` reverts to
  prose only, as D13 Q2 planned.
- **Custom `tools` node, not prebuilt `ToolNode`**, so it can write `tool_log`,
  `tool_calls_this_round` and merge `search_similar_tickets` hits into `retrieved_cases`.
- **`get_customer_history` closes over `customer_id`** — never a model-supplied argument.
- **`escalate_ticket` is advisory at CP4** (logged, shown to the model); routing to an
  escalation outcome is CP5's `assess_evidence`/`escalate`.
- **Skills:** `triage` (fixed), `investigation` (fixed), `customer_response` (fixed),
  `escalation` (optional, added to `active_skills` by `triage`; fixed for CP5's `escalate`).
  `load_skill` has no fallback — a missing file raises, so deleting one visibly changes
  behaviour.
- **Bug found in live verification:** `persist_case` hardcoded `stats.tool_calls = 0`; now
  `len(tool_log)` (regression test in `tests/test_persist_case.py`).
- **`autosupport.tools` imports are aliased** (`... as _search_similar_tickets`): a plain
  import rebinds the package attribute from the submodule to the function, breaking dotted-path
  monkeypatching.

---

## D15 — CP5: loops, interrupts, verification — deviations from graph-design.md

Everything in graph-design.md §4-§8 is built as written except the following, each found
while planning or verifying and recorded rather than silently absorbed.

- **F1 — `similarity` is re-anchored to the ticket.** `rag.queries.search` scores similarity
  against its *own query*; for a hypothesis rewrite or a tool search that isn't the ticket, yet
  `merge_cases` dedups on it and `assess_evidence` thresholds it against τ_rel. Every node that
  merges results (`retrieve_variant`, `tools`) recomputes it against the ticket embedding
  (`dense.similarities_to`, `graph/retrieval.py`). This also corrects CP4's `tools` merge.
- **F2 — `recursion_limit` 60 → 100.** Measured worst case is 69 node executions in one
  invocation (graph-design.md §6); 60 would have fired before a loop counter routed to `escalate`.
- **F3 — the "conflicting: top-queue share < 0.5" test is recomputed over the current
  *relevant* cases**, not triage's frozen round-1 `neighbor_agreement`, which refinement could
  otherwise never move.
- **F4 — new state key `escalation_trigger`**, written only by `escalate`.
- **F5 — a clarification-led dominant cluster is `insufficient`.** If ≥60% (weighted) of the
  dominant cluster's answers are `clarification_request`, it says what to ask, not how to fix
  (D5). Escalation-led dominant clusters are the `dominant_cluster_escalated` rule.
- **F6 — tool calls in one message run sequentially** inside the `tools` node (D14), not as
  parallel branches; the only genuinely parallel writers are `load_memory ∥ retrieve_initial`
  (disjoint keys) and the `retrieve_variant` Sends (`retrieved_cases` = `merge_cases`,
  `retrieval_queries`/`errors` = `operator.add`).
- **F7 — `route_after_confirm` uses `revision_count <= max_revisions`.** `confirm_resolution`
  increments the counter on the rejection it is handling, so graph-design.md's literal `<`
  would allow zero revised drafts with `max_revisions = 1`.
- **F8 — `gap_is_retrievable` is defined narrowly in the assess prompt.** A live run showed the
  model marking a thin ticket's gap "retrievable", so the run burned both refinement rounds
  *before* asking, leaving no round for the customer's answer to drive V1
  (`clarification_keywords`). The prompt now says a gap that is a fact only the customer has is
  not retrievable, matching graph-design.md §5's own "missing slot" definition. Verified: the
  thin-ticket run asks at `retrieval_round = 1` and V1 fires after the answer.
- **`escalate_ticket` (CP4's advisory flag) now feeds `escalation_rule_hit`** as
  `action_beyond_agent`, as does the model's `requires_human_action` judgement.
- **Retrieval filters reach both arms.** `rag/lexical.search` gained a whitelisted `where` and
  forced `phrases`, and `rag.queries.search` passes `where` to the lexical arm too — before,
  V2/V4's filter only constrained the dense arm and the lexical arm leaked unfiltered hits.
- **Bugs found in live verification:** `dense.similarities_to` raised `DuplicateIDError` when
  two searches in one turn shared a hit (deduped, regression test); `--json > file` crashed on
  a model-emitted `→` under cp1252 after the graph had finished (CLI output is now always
  UTF-8); `escalate` did not show the customer's clarification answers to the model (fixed).

---

## D16 — CP6: memory write policy and corpus growth

- **The memory write policy is code, not prompt.** `memory-design.md` §4 states rules W1–W6.
  The model only proposes items; `graph/memory.apply_update` keeps an item only if (W1) its
  verbatim `quote` is found in what the *customer* wrote (ticket, clarification answers,
  rejection feedback), (W2) its key is in a closed vocabulary (6 fact keys, 3 preference
  keys, plus tried fixes), and (W3) neither value nor quote matches the secrets/contact-PII
  pattern. `repeat_unresolved` (W6) is a count over `cases`, recomputed every write. This
  replaces the earlier draft's free-form `dict[str,str]` extraction, whose "keep both
  conflicting preferences" rule was impossible on a dict and whose caps were needed only
  because the keys were unbounded.
- **Index policy: accepted resolutions only** (`case-persistence.md` §5.1). `not_required`
  (eval runs) is excluded as well as escalated/rejected, so the agent's unconfirmed answers
  never become evidence for later tickets and evals don't mutate their own corpus.
- **Agent cases resolve through the same retrieval path.** `rag.queries._fetch_rows`
  previously read only `dataset_tickets` and silently dropped any other hit; it now resolves
  `T-` ids from `cases`, and `source` flows through `SearchResult` → `RetrievedCase`.
- **Re-ingest preserves learned cases.** `rebuild_fts` now deletes only `source='dataset'`
  rows (it used to wipe the whole FTS table on *every* ingest), and `--rebuild` re-indexes
  `cases` rows with `indexed_at` set.
- **One prompt rendering of memory** (`graph/memory.profile_block`) for `triage`,
  `investigate` and `resolve`. `resolve` previously saw no profile at all, so a stated
  preference or tried fix could not shape the customer-facing draft.

---

## D17 — CP7: offline evaluation

- **Offline evaluation ≠ `verify`** (`evaluation-design.md` §1). The code lives in `evals/`,
  feedback keys never say "verif", and no evaluator changes a run.
- **Held-out examples.** The 15 corpus examples come from outside `dataset_tickets`, and
  none has an indexed near-duplicate (cosine ≥ 0.92). `ingest/load.load_ingest_subset` drops
  them from every future ingest, filtered after the `--limit` sample so existing samples
  don't shift. Without this, the agent would retrieve an eval ticket's own answer.
- **The brief's five example tickets are not in the repo.** `evals/brief_examples.json` is
  the slot. `dataset.sync` adds them without gold labels, and the label-based evaluators
  skip them.
- **A fifth evaluator, `outcome_appropriateness`.** CP7 lists four; REQUIREMENTS §8 also
  requires "overall agent behaviour". It is code-only: the gold `answer_class` implies the
  expected outcome.
- **Graded retrieval relevance.** The binary judge scored exact-request neighbours on a
  different platform as irrelevant, so the judge grades relevant / partial / irrelevant
  (1 / 0.5 / 0).
- **Trace metadata.** `service._run_config` adds `ticket_id`, `customer_id` and `thread_id`
  to every invocation, not only eval runs.
- **Failed runs are skipped, not scored.** The first experiment hit an Anthropic
  credit-balance error on its last 4 examples. A target that raises reaches the evaluators
  as `{"output": None}`, which had been scored as 0.0 retrieval and 1.0 tool use. Every
  evaluator now skips a run without an `outcome`.

---

## D18 — Cost: prompt budget, prompt caching, and fixing over-asking

Measured on the first eval experiment (LangSmith token counts): **$1.91 for 15 tickets**.
`investigate` (Sonnet) accounted for 81% of it: 56 calls, 532K input tokens, and 49K
output tokens, a large share of them adaptive-thinking tokens. Two tickets that went
through 3 retrieval rounds accounted for 47% of the `investigate` cost.

- **Prompt budget** (`graph/retrieval.py`):
  - `investigate` shows at most 12 graph-retrieved cases, with 300-character snippets
    (was up to 30 cases × 600 + 600 characters).
  - `search_similar_tickets` returns 300-character snippets.
  - `assess_evidence` keeps every relevant case (its shares are weighted over all of them)
    but with 300-character snippets.
  - Full text is one `get_ticket_by_id` call away, which is the D7 small-to-big design.
- **Prompt caching.** `investigate`'s ReAct call sends `cache_control={"type":
  "ephemeral"}`, i.e. automatic caching up to the last block. Tool-search hits are no
  longer rendered into the system block; they are already in their tool result. So the
  system prefix stays fixed for a whole round and every ReAct turn after the first reads
  it from cache.
- **Projection:** replaying the 11 valid tickets' checkpoints against the call sequence
  in their traces gives `investigate` input of 532K → 369K tokens after trimming → about
  147K effective after caching, **not yet measured** (Anthropic credit ran out).
- **Over-asking (CP7's worst evaluator, `outcome_appropriateness` 0.36).** Two changes:
  - The assess prompt limits `missing_slots` to facts that block choosing or applying a
    fix: at most 2, and not merely because historical agents asked for them.
  - `next_action_for` escalates on a matched rule before refining or asking
    (graph-design §4.2 step 0).

---

## D19 — Fewer model calls: Python for deterministic work, one call per investigation round

**Problem, measured on the first eval experiment:** 106 agent LLM calls for 12 tickets, about
9 per ticket. Several were a model doing what code can do, or re-reading what another call had
just read:

| Before | Calls in the eval | Why it existed |
|---|---|---|
| `triage` (Haiku) | 15 | Classify queue/type/priority |
| `investigate` conclusion (Sonnet) | ~18 | A second full-context call after the last ReAct turn, only to extract JSON |
| `assess_evidence` (Haiku) | 18 | Re-read the same cases to cluster them and find missing facts |
| `refine_retrieval` rewrite (Haiku) | 6 | Turn the hypothesis into a search query |
| `escalate` (Sonnet) | 4 | Write the handoff note and holding reply |
| `update_memory` (Haiku) | 3 | Extract durable facts, called for every ticket |

**What changed (chosen by the user from an audit of every call site):**
- **`triage` → Python.** A similarity × cluster-weighted vote over the retrieved neighbours'
  own dataset labels. The labels are the ground truth `classification_accuracy` scores
  against, so the vote is measurable, and it is free. The `escalation` skill is selected by
  rule (escalation-class share ≥ 0.4, or a high-stakes term). `skills/triage.md` was removed.
- **`investigate` ends with a `submit_findings` tool call.** Its arguments (`Findings`) carry
  hypothesis, evidence, clusters, missing facts, the clarification question and the
  human-action flag. This removes both the separate conclusion call and the `assess_evidence`
  model: the investigator judges the evidence it just gathered, in the same turn.
  - The arguments are schema-typed and Pydantic-validated. This deviates from CLAUDE.md's
    "structured steps use `.with_structured_output`": the structure arrives as a tool call.
  - If the model answers in prose or the budget is spent, one forced `submit_findings` call
    runs with thinking disabled, because Anthropic forbids forced tool choice with thinking.
- **`assess_evidence` → Python only.** It computes verdict, rule and route from `findings`
  plus code metrics.
- **`refine_retrieval` V3 → Python.** The hypothesis statement is the query; its and the
  evidence's entities (the `ENTITY` pattern) become forced BM25 phrases.
- **`escalate` → template.** Reason from the trigger/rule, handoff from hypothesis, cited
  evidence, customer answers and missing facts, and a holding reply that honours a
  remembered contact channel. It can only cite `evidence` entries, so it cannot invent a
  case. `verify` checks it with code rules G1–G3 only, since no model wrote a claim; the LLM
  claim check still runs on every `resolve` draft.
- **`update_memory` gated.** A regex for write-policy candidates (versions, OS names,
  deployment/plan words, "already tried…", stated preferences) runs first; the extraction
  call happens only on a match. W6 flags are still recomputed every time.
- **`resolve` → fast tier.** This relaxes D11: drafter and checker are now the same model,
  and the separation rests on the role (a separate adversarial prompt) plus the
  model-independent code rules.

**Result:**
- **Calls per ticket:** typical ticket from ~9 LLM calls to ~3 (the investigate tool turns, one
  `submit_findings`, one `resolve` draft). Every other node is Python.
- **Same 12 tickets:** 106 agent calls → ~41 (projected from the recorded call sequences).
- **Cost:** combined with D18, projected at roughly a third of the original agent cost. **Not
  yet measured**, because Anthropic credit ran out; the next eval run records the real
  numbers here.

---

## D20 — Provider-switchable models: Groq GPT-OSS alongside Claude

**Why:** after D18/D19 cut the number of calls, per-token price is the remaining lever. On
Groq, GPT-OSS 20B costs $0.075 in / $0.30 out per million tokens, against Claude Haiku 4.5
at $1 / $5 and Sonnet 5 at $2 / $10: about 13–33× cheaper.

**Locked stack:** this adds a second provider. The user proposed and approved it on
2026-09-25. Claude stays the default; Groq is opt-in per tier.

**What changed:**
- **Model strings pick the provider.** `AUTOSUPPORT_MAIN_MODEL` / `_FAST_MODEL` accept
  `anthropic:…` or `groq:…`.
- **Keys are required only for providers in use.** `config.py` fails fast with the missing
  variable's name.
- **New dependency: `langchain-groq`.** It is the official provider package for
  `init_chat_model("groq:…")`, adding 2 packages; `httpx` and `pydantic` were already present.
- **A separate eval judge.** `AUTOSUPPORT_JUDGE_MODEL` grades the offline eval, so swapping
  a tier under test never changes who grades it.
- **Provider differences live in `llm.py` helpers, so nodes stay provider-agnostic:**
  - Anthropic-only request options: `cache_control` and `thinking`.
  - `structured()`: strict JSON schema on Groq.
  - `must_call_a_tool()`: `tool_choice="required"` on Groq.

**Measured on GPT-OSS 20B before wiring it in:**
- **Plain calls, `DraftOutput` and `VerificationJudgement`:** all worked. The checker caught
  a "this will definitely fix it" overclaim at a `low` band.
- **`MemoryUpdate`:** returned the schema's own shape until `strict=True`, then correct.
  Hence `structured()`.
- **Free ReAct turn:** with `tool_choice="auto"`, one call reasoned until the token limit and
  called nothing. With `"required"`: 8/8 tool calls. Hence `must_call_a_tool()`.
- **Forced `submit_findings`:** valid `Findings` with sensible stances and clusters.
- **Reasoning effort:** default averaged ~840 output tokens per turn; "low" averaged ~50 but
  always searched first. Default is kept: about $0.0003 per turn.

**Free tier too small (resolved).** Groq's free tier allows 8,000 tokens/min and 1,000
requests/day per model, and a single `investigate` request measured 9,657 tokens. The
account moved to the Dev tier: 250K tokens/min, 500K requests/day.

**Found in the first live Groq ticket** (end to end in 35s):
- **`tool_use_failed`.** With tool use required, GPT-OSS 20B sometimes writes the `Findings`
  JSON as text, and Groq rejects the response with a 400. A failed free turn now falls into
  the forced-`submit_findings` path, which retries once (`llm.is_tool_use_failure`). The text
  is never parsed; the retry returns a validated tool call.
- **It asked the customer for their S3 access key and secret key**, and listed 4 missing
  details against the prompt's "at most 2". Code now enforces both
  (`assessment.askable_slots`): secret-type details are dropped, the list is capped at 2,
  and the question is rebuilt from what's left whenever anything was removed. A support
  agent must never ask for credentials, whichever model is behind it.

**Found in the eval runs:**
- **A tool that wasn't offered.** In the forced fallback, GPT-OSS once called
  `search_similar_tickets` although only `submit_findings` was offered. The fallback now makes
  3 attempts. If none returns valid findings, the round degrades to explicit "no findings" with
  a recorded error, and the ticket escalates to a person instead of crashing (graph test
  added).
- **20B vs 120B for `investigate`.** GPT-OSS 20B marked all evidence `neutral` even with 18
  relevant cases at 0.889 similarity, so nothing could reach "sufficient" (outcome 0.21).
  120B gives usable stances: all 15 tickets ran, outcome 0.33 before D21.

**Adopted defaults** (user decision, 2026-09-25):
- `investigate`: `groq:openai/gpt-oss-120b`
- `resolve`, `verify`, memory: `groq:openai/gpt-oss-20b`
- eval judge: `groq:openai/gpt-oss-120b`
- Claude remains a one-line switch in `.env`.

**Result — first run (Claude, original pipeline) vs final run (Groq + D18–D21), same 15
held-out tickets:**

| | Claude, original | Groq + D18–D21 |
|---|---|---|
| Eval run cost, judge included | $1.91 | **$0.061** (31× cheaper) |
| Agent cost per ticket | ~$0.15 | **$0.0033** (~45× cheaper) |
| LLM calls per ticket | ~9 | **3.9** |
| Time per ticket | ~100 s | ~5–13 s |
| `outcome_appropriateness` | 0.36 | **0.47** |
| Tickets resolved | 0 | 2 |
| `retrieval_relevance` | 0.63 | 0.65 |
| `tool_usage_correctness` | 1.00 | 0.91 |
| `classification_accuracy` | 0.52 (LLM) | 0.49 (Python vote, D19) |

The LLM-judged scores come from different judges (Sonnet 5 vs GPT-OSS 120B), so they are
indicative only. Calls, cost, tool usage, classification and outcome are measured the same
way in both columns.

---

## D21 — "Conflicting" means competing fixes, not noisy labels

**Problem, measured in the eval:** no ticket resolved in any run, on any model (0/11 Claude,
0/14 GPT-OSS 20B, 0/15 GPT-OSS 120B). On 3 of the 5 resolution-class tickets the dominant
cluster was 100% `resolution`-class, with 3–4 supporting cases, yet the verdict was
`conflicting` for two reasons:
- **Queue agreement.** The relevant cases' queue labels agreed only 20–33% of the time, and
  the dataset's queue labels are noisy (`classification_queue` scores 0.27–0.50 against them).
- **Non-answers counted as rivals.** Clusters of "asked for more info" or "escalated" answers
  counted as rival approaches, which dragged the real fix's share below 0.6.

**Change** (`assessment.verdict_for`, graph-design §5):
- A cluster *proposes a fix* if at least half its weight is resolution-class.
- `conflicting` = two or more fix-proposing clusters with the largest under 0.6 of their
  weight, or the customer's history contradicts the approach.
- No fix-proposing cluster at all = `insufficient`.
- The queue-agreement test now only triggers retrieval variant V4.

**Result:**
- **Simulated first:** replaying the 15 stored checkpoints, first decisions matching the
  historical outcome went from 5/15 to 8/15, with no escalation- or clarification-class
  ticket getting worse.
- **Then measured:** see D20's table. Outcome was 0.47, and the first resolutions appeared.
  Which resolution tickets resolve varies between runs (HF-54906 resolved in one run and
  escalated in the next), so single-run differences on 15 tickets are noise-sized.

---

---

## Open, pending data

All items previously listed here are resolved, with measured numbers, in `rag-design.md`'s
full rewrite:
- ~~Clustering method and threshold (D3)~~ → **star (leader) clustering**, not connected
  components (measured: connected components collapses 74.6% of the corpus into a handful
  of giant clusters via transitive chaining). Threshold **T = 0.92** with a same-answer-
  class/answer-similarity/entity-conflict **guard**, `T_a = 0.85` (`rag-design.md` §4).
- ~~Embedding-text composition vs filterable metadata~~ → `subject + body` (index text,
  normalised — `rag-design.md` §2) only; queue/type/priority/answer_class/cluster_size/tags
  are Chroma metadata only, never concatenated into the embedded text (`rag-design.md` §3).
- ~~Which `version` value yields the ~28K English subset (D6)~~ → none: `language == 'en'`
  alone yields 28,261 rows, 23,801 after exact-duplicate dedup; `version` is provenance
  metadata, not a filter (`rag-design.md` §1).
- ~~Over-fetch size, RRF constant, MMR λ (D4)~~ → over-fetch 50/arm, RRF **k = 10** (not the
  literature default of 60 — measured to structurally suppress lexical-only hits at this
  corpus's arm-overlap rate), MMR **λ = 0.7** (`rag-design.md` §7–§8).
- ~~Second-pass targeted retrieval query construction~~ → four variants
  (`clarification_keywords`, `resolution_only`, `hypothesis_rewrite`, `queue_filtered`),
  triggers and reasons in `rag-design.md` §10.
- ~~`τ_rel` and `SIM_CEILING` recalibration~~ → `τ_rel = 0.76` (measured random-pair p95,
  replacing the old 0.55 default which sat *below* the random-pair median), `SIM_CEILING =
  0.92` (equal to the clustering threshold T — `rag-design.md` §9).
