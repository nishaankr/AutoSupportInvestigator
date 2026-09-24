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
