# Build Plan — Checkpoints

> **Budget:** ~16 hours, one day. **Status:** not started.
> Each checkpoint is a git tag. Work them in order. Do not start CP*n+1* until CP*n*'s
> verification command passes.

## Ground rules

1. **Verify before advancing.** Every checkpoint has a command that must produce the stated
   result. Run it. Don't claim done on inspection alone.
2. **Tag on green.** `git tag cp<N>` when verification passes. If a later checkpoint goes
   badly wrong, `git reset --hard cp<N>` rather than debugging forward.
3. **CP3 is the gate.** If hour 7 arrives without a working vertical slice, stop adding
   features and debug. Everything after CP3 is additive on a system that already runs.
4. **Iterate on `--limit 500`.** Run the full 28K ingest once, in a background terminal.

---

## CP0 — Skeleton · 0.5h

**Build:** repo tree per `CLAUDE.md`, `pyproject.toml`, `config.py` (pydantic-settings),
`.env.example`, `.gitignore`, `llm.py` with both tiers via `init_chat_model`, empty module
files, Typer app with all commands stubbed to `raise NotImplementedError`.

**Also:** a `scripts/smoke_llm.py` that sends one trivial message to each tier and prints the
response. Model-string mistakes are the single most common way to lose 30 minutes on day one.

**Done:** `autosupport --help` lists every command. `python scripts/smoke_llm.py` returns
text from both the main and fast tiers. `python -c "from autosupport.config import settings; print(settings)"`
prints resolved config and raises clearly if a key is missing.

**Tag:** `cp0`

---

## CP1 — Ingestion · 2.0h

Implements `architecture.md` §4.1's six stages. Read `rag-design.md` first — if it doesn't
yet specify the clustering threshold, the embed-text composition, or the `answer_class`
heuristics, **write those into it before coding.**

**Build:** `ingest/load.py` (HF → English filter → local parquet snapshot),
`ingest/classify.py` (regex/heuristic pass → 200-row sample report → LLM on residue only),
`ingest/cluster.py` (normalise → near-duplicate clustering → canonical selection →
`cluster_id` / `cluster_size` / `is_canonical` / `canonical_of`), `ingest/index.py`
(SQLite write → FTS5 build → embed canonicals → Chroma upsert), `store/` repositories.

**Watch:** Chroma metadata values must be scalars — tags go in as a joined string plus one
boolean key per tag. Only canonicals get embedded.

**Done:** `autosupport ingest --limit 500` completes and prints row counts, the
`answer_class` distribution, cluster-size distribution, and the sampled classifier precision.
Then kick off the full run in a second terminal and continue to CP2 while it works.

**Tag:** `cp1`

---

## CP2 — Retrieval layer · 1.5h

**Build:** `rag/embedder.py`, `rag/dense.py` (Chroma similarity + metadata filters),
`rag/lexical.py` (FTS5 `bm25()` query), `rag/fusion.py` (RRF + MMR, written in-repo),
`rag/queries.py` (query construction, filter building). Plus a temporary
`autosupport search "..." [--k N] [--queue Q]` command that prints fused results with both
arms' ranks, the final RRF score, `cluster_size` and `answer_class`.

**Done:** search on a distinctive entity phrase (e.g. a hardware model name) returns results
where BM25 visibly contributes rank that dense retrieval alone misses. Results are diverse —
no five near-identical rows. Record the chosen RRF constant and MMR λ in `rag-design.md`.

**Tag:** `cp2`

---

## CP3 — Vertical slice · 1.5h  ← **the gate**

**Build:** `graph/state.py` exactly as `state-schema.md` specifies. Then the minimum path:
`intake → (load_memory ∥ retrieve_initial) → triage → resolve → persist_case → END`.
No tools, no loops, no interrupts, no verify. `service.new_ticket()` and `cli.py` rendering.
Compile with `SqliteSaver`.

**Done:** `autosupport new --customer C-1 --subject "..." --body "..."` writes an open case,
runs through, and prints a schema-valid `CaseResult` with real cited case IDs.
`autosupport show <id>` retrieves it. `--json` produces parseable output.

**Tag:** `cp3`

---

## CP4 — Tools and skills · 2.0h

Read `tools-and-skills.md` first; write it if it doesn't exist yet.

**Build:** five tools — `search_similar_tickets`, `get_ticket_by_id`, `get_customer_history`,
`compute_queue_stats`, `escalate_ticket`. Four skills as `skills/*.md` — `triage`,
`investigation`, `escalation`, `customer_response` — plus `skills.py` loading them by name.
Replace the straight-through path with the ReAct loop: `investigate ⇄ tools`, bounded by
`max_tool_calls_per_round`.

**Watch:** the model chooses tools; nothing is hardcoded into a fixed sequence. Skills load
per node — never concatenate all of them into one system prompt.

**Done:** a run's `tool_log` shows at least two different tools called on the model's own
initiative. `state["active_skills"]` reflects what `triage` selected. Removing a skill file
changes behaviour, proving skills are actually loaded rather than inlined.

**Tag:** `cp4`

---

## CP5 — Loops, interrupts, verification · 2.5h

The heaviest checkpoint. Follow `graph-design.md` §4–§8 exactly.

**Build:** `assess_evidence` (rule metrics weighted by `cluster_size`, plus a fast-tier
judgement on conflicts and missing slots — it also generates `pending_question`),
`refine_retrieval` with `Send` fan-out to `retrieve_variant`, `verify` (fast tier, different
from the drafter), `escalate`, `ask_user` and `confirm_resolution` interrupts, and every
router in `routers.py` with its counter and limit.

**Watch:** interrupt nodes contain no side effects — the question is generated upstream.
Every key written by parallel branches needs a reducer. Every loop routes to `escalate` on
exhaustion.

**Done — this is a four-part check:**
1. A thin ticket triggers `ask_user`; **the process exits**; `autosupport resume <id> --answer "..."`
   picks up from the checkpoint and completes. (Process exit is the point — it's what proves
   `SqliteSaver` over `InMemorySaver`.)
2. `autosupport list --awaiting` shows the paused ticket while it's paused.
3. A run reaches `confirm_resolution`; `--reject "feedback"` loops back and produces a
   different draft.
4. Forcing all counters to their limits terminates in `escalate`, never a recursion error.

**Tag:** `cp5`

---

## CP6 — Memory and case growth · 1.5h

Read `memory-design.md` and `case-persistence.md`; write them if absent. The write policy —
what is worth remembering and what is discarded — is explicitly graded, so make it a stated
policy in the doc, not an implicit prompt.

**Build:** `load_memory` (profile + open/recent cases), `update_memory` (fast tier, applying
the write policy, harvesting durable facts from `clarifications`), `index_case` (embed the
accepted resolution, upsert with `source="agent_resolved"`, add to FTS5). Both run in
parallel after `persist_case`. Escalated and rejected cases are not indexed.

**Done — two parts:**
1. Submit ticket A for `C-1`, resolve and accept it. Submit ticket B for `C-1` on the same
   theme. `autosupport memory C-1` shows facts harvested from A, and B's trace shows them
   loaded and used.
2. B's retrieved cases include ticket A with `source="agent_resolved"`. This is the
   continuously-growing-knowledge requirement and it must be demonstrably true.

**Tag:** `cp6`

---

## CP7 — LangSmith evaluation · 1.5h

Read `evaluation-design.md`; write it if absent. Keep the naming distinct from the `verify`
node — these are different things and a reviewer will check that you know it.

**Build:** an eval dataset of ~15 examples (reuse the example tickets from `docs/project/brief.md` — *as built: that file was never provided, so the dataset is 15 held-out corpus tickets with a `evals/brief_examples.json` slot; decisions.md D17*),
and four evaluators: response quality/groundedness, retrieval relevance, tool-usage
correctness, classification accuracy against the dataset's own `queue`/`type`/`priority`
labels — that last one is free ground truth, use it. Run with `require_acceptance=false` and
`thread_id = f"eval:{run_id}:{example_id}"`.

**Done:** `autosupport eval` completes and scores appear in the LangSmith UI. Traces carry
`ticket_id`, `customer_id` and `thread_id` metadata.

**Tag:** `cp7`

---

## CP8 — Demo, README, docs · 2.0h

**Build:** `scripts/demo.py` covering all four required scenarios in one scripted run —
normal resolution, clarification + resume, long-term memory reuse, retrieval of a newly
resolved case. Then the README: one-line statement, locked stack table, repo tree,
prerequisites table, setup, commands, architecture summary, graph diagram, state/memory
strategy, RAG approach, tool/skill design, **limitations**.

**Limitations must include:** no customer IDs in the source data; heuristic `answer_class`
with its measured precision; threshold-based clustering; canonical-only retrievability;
single-process SQLite/Chroma; local embedding quality trade-off; model-based self-check.
The brief asks for limitations explicitly — a thin section here is a cheap loss.

**Finally:** reconcile every `docs/*.md` against what was actually built. Anywhere the code
diverged from the design, update the doc and note why.

**Done:** `autosupport demo` runs clean start to finish. A fresh clone plus the README's
setup steps reaches a working `autosupport new`.

**Tag:** `cp8`

---

## Buffer · 1.0h

Reserved. If CP5 overruns, it comes from here.

---

## Parallel track (do these while something else runs)

- **During CP0:** write `docs/project/brief.md` — client context, the problem, what the agent must do,
  five example tickets, the trust contract, constraints, out-of-scope, definition of done.
  Those five example tickets become the CP7 eval dataset and the CP8 demo script. Writing
  them once serves three checkpoints.
  *Status: not done — the brief text was never provided. Demo (CP8) and eval (CP7) use
  corpus-derived tickets instead; both have a slot for the brief's tickets.*
- **During CP2:** the full ingest runs in a second terminal.
- **After each checkpoint:** commit, tag, and note anything that diverged from the docs.

## Drop order, if hour 13 arrives at CP5

1. `Send` fan-out in `refine_retrieval` — go sequential (explicitly optional in the brief).
2. Collapse the two model tiers to one, keeping the config seam.
3. Loop E / `revision_count` — accept-only, no reject-and-revise.

Do not drop: either interrupt, the corrective-retrieval loop, the evals, or the limitations
section. Those are graded.
