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

**Rejected:** exact-hash dedup alone (misses templated variation); MinHash/LSH (viable, but
embedding-threshold clustering reuses machinery already present); retrieval-time MMR *alone*
(treats the symptom, discards the signal). MMR is still used, on top of this.

---

## D4 — Hybrid retrieval: dense + BM25, fused with RRF

Chroma supplies the dense arm; a SQLite FTS5 virtual table supplies BM25. Results are fused
with Reciprocal Rank Fusion, then diversified with MMR. RRF and MMR are written in-repo.

**Why:** every ticket in this corpus shares the same generated scaffolding ("Dear Customer
Support Team, I hope this message reaches you well"). That boilerplate dominates the
embedding and flattens dense similarity across unrelated tickets. The actual discriminating
signal is entities — `QNAP NAS`, `Aruba 2530`, `macOS 15`, `Crucial MX500`, `Kubernetes` —
which lexical search captures precisely and embeddings blur.

**Why FTS5 specifically:** it ships inside SQLite, so the lexical arm costs no new
dependency and no new store, and it indexes the same rows the tools already query.

**Rejected:** dense-only (loses the entity signal); `rank_bm25` (a dependency wrapping what
the standard library already provides, holding a second in-memory copy of the corpus with no
persistence).

---

## D5 — `answer_class` assigned at ingest, by heuristics with a validated sample

Every historical answer is typed `resolution`, `clarification_request` or `escalation`.
Method: regex/heuristic first pass; precision measured against a hand-labelled random sample
of 200; an LLM applied only to the ambiguous residue.

**Why the field exists:** a large share of the dataset's `answer` values are not resolutions
at all — they are requests for more information ("please specify the model and firmware
version"). Grounding a resolution on five retrieved cases that all say "send us your logs"
produces confident nonsense. Conversely, a retrieved neighbourhood dominated by
`clarification_request` is the strongest available signal that *this* ticket needs the
clarification route — it tells the agent what to ask and that asking is correct.

**Why this method:** running a model over 28K rows is slow and expensive for a labelling
task that patterns handle well. The sampled precision figure is reported in the README as
measured methodology rather than an unverified claim — which is itself worth more than a
marginally better classifier.

**Rejected:** LLM over the full corpus; heuristics with no validation step.

---

## D6 — English subset only

Filter `language == "en"` at load. Mandated by the brief. The source file also merges two
dataset generations via the `version` column; confirm which combination yields ~28K English
rows and record it in `rag-design.md`.

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

## Open, pending data

Resolved by profiling the real dataset at CP1 prep — see `rag-design.md`:
- ~~Clustering method and threshold (D3)~~ → connected components, cosine ≥ 0.92
  (`rag-design.md` §2.3), grounded in a measured random-pair similarity ceiling
  (p99.9 = 0.900) vs. a qualitatively-confirmed near-duplicate band (0.90–0.97).
- ~~Embedding-text composition vs filterable metadata~~ → dense embed text is
  `subject + body` only; tags/queue/type/priority/answer_class/cluster_size are Chroma
  metadata only, never concatenated into the embedded text (`rag-design.md` §3).
- ~~Which `version` value yields the ~28K English subset (D6)~~ → none: `language == 'en'`
  alone yields 28,261 rows; `version` is provenance metadata, not a filter
  (`rag-design.md` §1.1).

Still open, blocked on CP2 (retrieval-layer build, not ingest):
- Over-fetch size, RRF constant, MMR λ (D4)
- Second-pass targeted retrieval query construction
- `τ_rel` and `SIM_CEILING` recalibration — the measured random-pair baseline (median
  cosine 0.585, p99.9 0.900) sits *above* the current `τ_rel = 0.55` default from
  `graph-design.md` §5, meaning most random unrelated pairs would currently register as
  "relevant." This needs fixing once CP2's fused (RRF) ranking exists to test against,
  not against raw dense cosine alone (`rag-design.md` §5).
