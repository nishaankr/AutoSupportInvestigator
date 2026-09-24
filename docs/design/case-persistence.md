# Case Persistence — the `cases` table

> **Status:** v2 · CP6. The system of record for every ticket the agent has seen, and (§5)
> the policy for which of them grow the retrievable corpus.
> **Companion docs:** `state-schema.md` (in-memory `AgentState`), `output-schema.md`
> (`CaseResult`, the JSON stored in `final_output`), `graph-design.md` (node order),
> `memory-design.md` (the sibling `customers` table).
> **Code:** `store/cases.py`, `ingest/agent_index.py` (§5), `graph/nodes/index_case.py`.

---

## 1. Why SQLite, why one table

Decided in `architecture.md` §2.2. `cases` is the only table that changes shape over a
ticket's life (open → … → resolved/escalated), so it's the one table with real write
concurrency concerns (a resume can race a `list`). Everything else here just answers: what
are the columns, who writes each one, and what reads depend on them.

---

## 2. DDL

```sql
CREATE TABLE IF NOT EXISTS cases (
    ticket_id       TEXT PRIMARY KEY,              -- T-YYYYMMDD-<6 hex>
    customer_id     TEXT NOT NULL,
    thread_id       TEXT NOT NULL UNIQUE,           -- "{customer_id}:{ticket_id}"
    status          TEXT NOT NULL,                  -- CaseStatus
    subject         TEXT NOT NULL,
    body            TEXT NOT NULL,
    submitted_at    TEXT NOT NULL,                  -- ISO 8601, from TicketInput.submitted_at
    classification  TEXT,                           -- Classification, JSON; NULL before triage
    pending_question TEXT,                          -- mirrors state.pending_question; NULL unless awaiting_user
    final_output    TEXT,                           -- CaseResult, JSON; NULL until persist_case
    indexed_at      TEXT,                           -- set by index_case (CP6); NULL until then
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cases_customer_updated ON cases(customer_id, updated_at);
CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status);
```

**Column notes**
- `ticket_id` is the primary key, not `thread_id`. The CLI, `get_ticket_by_id` and every
  citation in `CaseResult` prose address a case by `ticket_id`; `thread_id` exists only for
  the checkpointer. `thread_id` still gets its own `UNIQUE` index because `resume` looks up
  by ticket_id and needs the thread_id it maps to, in one row, with no ambiguity.
- `subject`/`body` are stored raw (not the index/normalised form `rag-design.md` uses for the
  dataset) — they're `TicketInput.subject`/`.body` verbatim, since this is the record of what
  the customer actually sent, and CP6's `index_case` normalises separately when it writes to
  `dataset_tickets_fts`.
- `classification` and `final_output` are whole Pydantic models dumped with
  `model_dump_json()`. Storing them as JSON blobs (rather than exploding into columns) matches
  `output-schema.md` §6 ("`service.show()` loads it back with `model_validate_json`") and
  avoids a second schema that has to track every Pydantic field change. `status` is
  denormalised out of `final_output` into its own column because it's the one field every
  list/filter query needs and JSON extraction for every row would be slower and harder to
  index.
- `pending_question` is denormalised out of state for the same reason `status` is: `list
  --awaiting` (CP5) needs to show it without replaying the checkpoint.
- No `escalation_rule_hit`, `verify_attempts`, etc. as columns — those live inside
  `final_output` once persisted, and before that they're only in the checkpoint. `cases`
  stores the two things every query path needs cheaply (`status`, `pending_question`) and the
  two big documents (`classification` as an early preview, `final_output` as the final
  record). Everything else is either in `final_output` or intentionally checkpoint-only.

---

## 3. Writers, by node

| Node | Writes | Notes |
|---|---|---|
| `intake` | `INSERT`: `ticket_id, customer_id, thread_id, status='open', subject, body, submitted_at, created_at, updated_at` (`classification`, `pending_question`, `final_output`, `indexed_at` all `NULL`) | The only `INSERT`. Runs once per ticket, before any interrupt (state-schema §4), so it can never run twice for the same `ticket_id` — see idempotency below. |
| `triage` | `UPDATE cases SET classification=?, status='investigating', updated_at=? WHERE ticket_id=?` | |
| `assess_evidence` / `verify` | `UPDATE cases SET status='awaiting_user', pending_question=?, updated_at=?` | The node that routes *into* an interrupt writes it — interrupt nodes have no side effects (graph-design.md §7.4). |
| `service.resume_ticket` | `UPDATE cases SET status='investigating', pending_question=NULL, updated_at=?` | Before invoking `Command(resume=…)`, not inside the interrupt node. |
| `persist_case` | `UPDATE cases SET final_output=?, status=?, updated_at=? WHERE ticket_id=?` | `status` is `final_output.status` ("resolved" or "escalated"), so the row and the JSON always agree — see the validator note below. |
| `index_case` | `UPDATE cases SET indexed_at=? WHERE ticket_id=?` | Only for accepted, resolved cases (§5). |

No node ever writes two different values to the same column in one run — each row above owns
its own `UPDATE`, so there's no reducer question here the way there is for `AgentState`; SQLite
serialises writes to the same row.

### Status transitions

```
open ──(triage)──► investigating ──(→ ask_user / confirm_resolution)──► awaiting_user
                         ▲                                                   │
                         └──────────────────(service.resume_ticket)──────────┘
investigating ──(persist_case, decision=resolve, verify passed)──► resolved ──(index_case, if accepted)──► indexed_at set
investigating ──(persist_case, decision=escalate)──► escalated
```

`persist_case` is the only writer of a terminal status, and
it writes exactly `final_output.status`, so `cases.status` and the persisted `CaseResult` can
never disagree — this is the DB-level echo of the `CaseResult` model_validator in
`output-schema.md` §5 (`status == "escalated"` iff `escalation.required`).

### Idempotency

`intake` is the only node that inserts. `state-schema.md` §5 describes two entry paths for a
thread: a brand-new ticket, and a follow-up on an already-closed ticket that reopens the same
`ticket_id`/`thread_id`. Both go through `intake`, so `intake`'s SQL is
`INSERT ... ON CONFLICT(ticket_id) DO UPDATE SET status='open', updated_at=? ` rather than a
bare `INSERT` — this is what makes "sees that `ticket_id` is already set" (state-schema §5)
concrete at the DB layer: the existing row's `subject`/`body`/`created_at` are preserved, and
only `status`/`updated_at` (and, at CP5+, cleared loop-related columns) reset. A resume
(answering a clarification or the accept/reject flow) never calls `intake` at all — it invokes
the graph on the existing `thread_id`, so it never touches this INSERT path.

---

## 4. Reads

- **`resume <ticket_id>`** (CP5): `SELECT thread_id, status FROM cases WHERE ticket_id=?`.
  If `status` isn't `awaiting_user`, the CLI reports there's nothing to resume rather than
  calling the graph. The thread_id feeds `graph.invoke(Command(resume=...), {"configurable":
  {"thread_id": row.thread_id}})`.
- **`show <ticket_id>`**: `SELECT * FROM cases WHERE ticket_id=?`. If `final_output` is
  `NULL` (still in progress), the CLI renders `status` and `pending_question` only — there's
  no `CaseResult` yet to load. If it's set, `service.show()` does
  `CaseResult.model_validate_json(row.final_output)` (output-schema §6) and returns it
  alongside `status`.
- **`list [--customer C] [--awaiting]`**: `SELECT ticket_id, customer_id, status, subject,
  updated_at FROM cases WHERE (customer_id=? OR ? IS NULL) AND (status='awaiting_user' OR NOT
  ?) ORDER BY updated_at DESC`. Both indexes above exist to serve this query's two filters
  without a full scan as `cases` grows.
- **`get_customer_history`** (tool, CP4) and `load_memory`'s history read (CP3): see
  `memory-design.md` §3 — same table, a `customer_id` + `status` query, joined conceptually
  with `customers` but no SQL join since they're separate lookups (`memory-design.md` explains
  why).
- **`compute_queue_stats`** (tool, CP4) reads `classification` JSON across `cases` plus
  `dataset_tickets.queue` — out of scope here, covered when `tools-and-skills.md` is written.

---

## 5. How an agent-resolved case reaches FTS5 / Chroma (`index_case`)

`index_case` runs after `persist_case`, in parallel with `update_memory`. It writes no graph
state, only Chroma, FTS5 and `cases.indexed_at`.

### 5.1 The index policy: what grows the corpus

**Indexed iff `final_output.status == "resolved"` and `acceptance == "accepted"`.**

| Outcome | Indexed? | Why |
|---|---|---|
| Resolved, customer **accepted** | Yes | A verified draft (`accepted` is only reachable after `verify` passed) that a human confirmed. The only outcome that is evidence a fix works |
| Escalated (any trigger) | No | No attested fix. The holding reply isn't a resolution |
| Resolved after a rejection, then accepted | Yes | The *accepted* revision is what gets indexed |
| Rejected, then escalated | No | Escalated |
| Resolved, `acceptance == "not_required"` (eval runs, `require_acceptance=false`) | No | Unconfirmed. Indexing the agent's own unconfirmed answers would let a wrong fix be retrieved as evidence for the next ticket and reinforce itself. It would also make eval runs mutate the corpus they are measured against |

What is indexed about an agent case is its **problem text and accepted resolution only** —
never `customer_id`, customer facts, or clarification answers. The corpus is shared across
customers; the profile is not (`memory-design.md` §1).

### 5.2 Mechanics (`ingest/agent_index.py::index_agent_case`)

- **Chroma:** embed `ticket_query_text(subject, body)` — the same normalise+embed-text rule as
  the dataset (`rag-design.md` §3), so the case lives in the same vector space — and upsert
  into `support_cases` with `id=ticket_id`, metadata `{source: "agent_resolved", queue, type,
  priority, answer_class: "resolution", cluster_size: 1, version: -1, tags + tag slugs}`: the
  shape dataset rows carry, so dense search and `where` filters need no source-specific path.
- **FTS5:** one row in `dataset_tickets_fts` with `case_id=ticket_id`,
  `source='agent_resolved'`, normalised subject/body, `answer` = normalised
  `final_output.resolution`, `tags` = `classification.tags`, and the same `UNINDEXED`
  `queue`/`type`/`answer_class` columns.
- **`cases.indexed_at`** is set last. The node first checks `indexed_at IS NULL`, so a case is
  indexed at most once and the step is re-runnable.
- **Retrieval:** `rag/queries.search()` resolves `HF-` ids from `dataset_tickets` and `T-` ids
  from `cases` (queue/type/priority/tags from `classification`, answer from
  `final_output.resolution`), and every `SearchResult`/`RetrievedCase` carries `source`.
  Citations of it enrich to `EvidenceEntry.source == "agent_resolved"` (output-schema.md
  §3.1). Agent cases never enter `dataset_tickets`, which stays a mirror of the HF ingest.

### 5.3 Surviving re-ingest

The corpus grows at runtime, so ingest must not silently drop what was learned:
- `dataset_tickets.rebuild_fts` deletes and re-inserts only `source = 'dataset'` rows.
- `ingest --rebuild` drops the FTS5 table and the Chroma collection, so it finishes by
  re-indexing every `cases` row with `indexed_at` set (`force=True`).

---

## 6. `ticket_id` generation (resolves state-schema/architecture Q3)

`service.new_ticket()` generates the ID before the graph runs — not `intake` — because
`thread_id` must exist to invoke the graph at all:

```python
def _new_ticket_id() -> str:
    return f"T-{datetime.now(timezone.utc):%Y%m%d}-{secrets.token_hex(3)}"
```

- 6 hex chars = 24 bits ≈ 16.7M values per day, which makes a collision astronomically
  unlikely for a local demo's ticket volume, but `intake`'s `INSERT ... ON CONFLICT` (§3 above)
  turns a collision into "reopen this ticket" behaviour rather than a crash — that's an
  accepted, harmless failure mode rather than something worth a retry loop.
- `service` builds `thread_id = f"{customer_id}:{ticket_id}"`, passes both into
  `InputState`, and `intake` only validates the format and mirrors `thread_id` into
  `state["thread_id"]` — it does not invent either value. `state-schema.md` §2.1/§3/§5 and
  `graph-design.md`'s node-1 row are updated to say this (done alongside this doc, in the same
  pass as the code, since they currently contradict each other on this exact point).

---

## 7. What's deliberately not here

| Item | Where it actually lives | Why not `cases` |
|---|---|---|
| Full `AgentState` (evidence, hypothesis, tool_log, …) | `checkpoints.sqlite`, per-thread | Working memory, not the system of record — `state-schema.md` §1. Wiped independently of `cases`. |
| `customer_profile`, long-term facts | `customers` table | `memory-design.md` — outlives any single case. |
| Retrieved case text | `dataset_tickets` / Chroma | `cases` never duplicates corpus content. |
