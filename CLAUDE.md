# AutoSupport — Agent Instructions

Autonomous Support Investigation Agent. A local LangGraph agent that receives a support
ticket, investigates it against a corpus of ~28K historical tickets, decides whether it has
enough information, and produces a grounded resolution or an escalation.

**Read `docs/checkpoints.md` before starting work.** It defines the build order and what
"done" means for each stage. Do not skip ahead.

---

## Stack (locked)

| Layer | Choice |
|---|---|
| Language | Python 3.11+ |
| Orchestration | LangGraph 1.x |
| Checkpointer | `SqliteSaver` → `data/checkpoints.sqlite` |
| Vector store (dense) | Chroma, embedded `PersistentClient` → `data/chroma/` |
| Lexical index (sparse) | SQLite FTS5 virtual table |
| Structured store | SQLite → `data/autosupport.sqlite` |
| Embeddings | local `sentence-transformers`, default `BAAI/bge-small-en-v1.5` |
| LLM (main tier) | `anthropic:claude-sonnet-5` via `init_chat_model` |
| LLM (fast tier) | `anthropic:claude-haiku-4-5` via `init_chat_model` |
| Interface | Typer CLI |
| Observability + evals | LangSmith |
| Config | `pydantic-settings` reading `.env` |
| Packaging | `pyproject.toml`, console script `autosupport` |

**The stack is locked.** Do not propose or substitute alternatives without a stated reason
and explicit approval. If something in the stack appears not to work, say so and stop —
do not silently swap it.

---

## Non-negotiable design constraints

These were decided deliberately. Do not revisit them mid-implementation.

1. **Deterministic IDs.** Dataset tickets are `HF-<row_index>`, assigned at ingest. Agent
   tickets are `T-YYYYMMDD-<6 hex>`. IDs exist so evidence can be cited, not for user lookup.
2. **No synthetic customers over the historical corpus.** The archive is anonymous.
   `customer_id` originates at runtime with each submitted ticket. Customer history and
   long-term memory read only from the `cases` and `customers` tables, never from the dataset.
3. **Canonicalisation with frequency counts.** Near-duplicate clusters collapse to one
   canonical case carrying `cluster_size`. Only canonicals are embedded and retrievable by
   similarity. Members stay in SQLite, reachable via `canonical_of`.
4. **Hybrid retrieval.** Dense (Chroma) + BM25 (FTS5), fused with Reciprocal Rank Fusion,
   then MMR for diversity. Write RRF and MMR in-repo. Do not add a BM25 library.
5. **`answer_class` at ingest.** Every historical answer is typed as `resolution`,
   `clarification_request` or `escalation`. Heuristics first, validated on a 200-row hand
   sample, LLM only on the ambiguous residue. Never run a model over all 28K rows.
6. **English subset only.** Filter `language == "en"` at load.

---

## Repo layout

```
AutoSupport/
├── CLAUDE.md                     # this file
├── README.md
├── pyproject.toml
├── .env.example                  # committed, no secrets
├── docs/                         # design docs — read before implementing a layer
├── skills/                       # triage.md, investigation.md, escalation.md, customer_response.md
├── autosupport/
│   ├── cli.py                    # Typer commands — rendering only
│   ├── service.py                # the ONLY thing cli.py calls
│   ├── config.py                 # pydantic-settings
│   ├── llm.py                    # init_chat_model for both tiers
│   ├── graph/                    # state.py, nodes/, routers.py, build.py
│   ├── tools/                    # @tool definitions
│   ├── skills.py                 # skill loader
│   ├── rag/                      # embedder, dense, lexical, fusion, queries
│   ├── store/                    # sqlite repositories
│   └── ingest/                   # load, classify, cluster, index
├── evals/                        # LangSmith datasets + evaluators
├── scripts/demo.py
├── tests/
└── data/                         # runtime, gitignored
```

Design docs, by layer: `architecture.md`, `graph-design.md`, `state-schema.md`,
`rag-design.md`, `tools-and-skills.md`, `memory-design.md`, `case-persistence.md`,
`evaluation-design.md`, `output-schema.md`. **Read the relevant doc before implementing
that layer.** Where a doc and this file disagree, this file wins; flag the conflict.

---

## Dependency policy

**Default: write it yourself.** Reach for a library only when the alternative is genuinely
hard to get right. Before adding any runtime dependency, state:

1. What it does that we can't write in under 30 lines of clear code.
2. How often it gets used.
3. Its transitive footprint.

OK: the declared stack, plus things that are hard to get right (DB drivers, LLM SDKs, parsers).
Not OK: helper libraries wrapping stdlib, frameworks where a function would do, "nicer API"
layers over a dependency we already have.

RRF, MMR and the clustering logic are written in-repo. That is deliberate.

---

## Configuration

`config.py` is the single source of truth for environment. **Never call `os.getenv` in
application code. Never call `load_dotenv` anywhere.** Fail fast at startup if required
config is missing — no silent fallbacks.

Secrets live in `.env`, which is gitignored. `.env.example` is committed with empty values.
`data/` is gitignored — it holds user tickets and customer memory.

---

## Architectural rules

- **`service.py` is the UI seam.** It returns Pydantic objects, never formatted strings.
  `cli.py` is a pure renderer over those objects. Every CLI command takes `--json` and dumps
  `model_dump_json()`. A graph node must never import from `cli.py` and must never print.
  A web frontend will be added later against this same layer.
- **State is working memory; SQLite is the system of record.** Retrieved cases carry
  snippets, not full text. Full bodies are fetched on demand via `get_ticket_by_id`.
- **Every state key written by parallel branches has a reducer.** Without one LangGraph
  raises `InvalidUpdateError`.
- **Interrupt nodes have no side effects.** No DB writes, no tool calls, no LLM calls before
  `interrupt()`. Resuming re-executes the node from the top. The clarification question is
  generated in `assess_evidence` and stored in `pending_question`.
- **Every loop has a counter in state and a limit in config**, and routes to `escalate` on
  exhaustion. The graph must always terminate.
- **Every structured LLM step uses `.with_structured_output(PydanticModel)`.** Never parse
  free text.
- **`verify` is the in-graph evidence check. It is not the LangSmith evaluation.** Keep the
  two concepts separately named in code and in docs.

---

## Code style

- Small, obvious functions. A 15-line function with clear names beats a three-class abstraction.
- No premature abstraction. Extract on the third caller, not a hypothetical one.
- No error handling for cases that can't happen. Validate at boundaries only: CLI input,
  external APIs, DB writes, dataset parsing.
- No backwards-compat shims, no speculative feature flags.
- Comments explain *why* when non-obvious, never *what*. Remove stale TODOs.
- Keep files focused and small.

---

## Commands

```bash
autosupport ingest [--limit N] [--rebuild]      # dataset → sqlite + fts5 + chroma
autosupport new --customer C --subject S --body B
autosupport resume <ticket_id> --answer "..."
autosupport resume <ticket_id> --accept | --reject "feedback"
autosupport show <ticket_id>
autosupport list [--customer C] [--awaiting]
autosupport memory <customer_id>
autosupport eval [--dataset NAME]
autosupport demo
```

---

## Working agreement

- Work one checkpoint at a time. Stop at each checkpoint boundary and report.
- Run the verification command for a checkpoint before claiming it is done.
- If a checkpoint's verification fails, fix it before moving on. Do not accumulate debt.
- Prefer `--limit 500` for iteration; run full ingest once, in the background.
