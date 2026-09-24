# Case Persistence — the `cases` table

> **Status:** Draft v1 · the system of record for every ticket the agent has seen.
> **Companion docs:** `state-schema.md` (in-memory `AgentState`), `output-schema.md`
> (`CaseResult`, the JSON stored in `final_output`), `graph-design.md` (node order),
> `memory-design.md` (the sibling `customers` table).
> **Scope:** CP3 needs `cases` to exist and to support `intake`/`persist_case`. Everything
> here is built at CP3 except the `list --awaiting` query and `index_case`'s write, which
> land when `ask_user` (CP5) and `index_case` (CP6) exist — the columns are reserved now so
> the schema doesn't migrate later.

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
| `ask_user` (CP5) | `UPDATE cases SET status='awaiting_user', pending_question=?, updated_at=?` on interrupt; `UPDATE cases SET status='investigating', pending_question=NULL, updated_at=?` on resume | Not built at CP3; the columns exist now so this is additive later. |
| `persist_case` | `UPDATE cases SET final_output=?, status=?, updated_at=? WHERE ticket_id=?` | `status` is `final_output.status` ("resolved" or "escalated"), so the row and the JSON always agree — see the validator note below. |
| `index_case` (CP6) | `UPDATE cases SET indexed_at=? WHERE ticket_id=?` | Only for accepted, resolved cases. |

No node ever writes two different values to the same column in one run — each row above owns
its own `UPDATE`, so there's no reducer question here the way there is for `AgentState`; SQLite
serialises writes to the same row.

### Status transitions

```
open ──(triage)──► investigating ──(ask_user interrupt, CP5)──► awaiting_user
                         ▲                                            │
                         └────────────────(ask_user resume)───────────┘
investigating ──(persist_case, decision=resolve, verify passed)──► resolved
investigating ──(persist_case, decision=escalate)──► escalated
```

At CP3 (no `assess_evidence`/`ask_user`/`verify`) the only path a ticket takes is
`open → investigating → resolved`. `persist_case` is the only writer of a terminal status, and
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

## 5. How an agent-resolved case reaches FTS5 / Chroma (CP6)

This is `index_case`'s job, not CP3's, but the shape is fixed now so `cases`' columns don't
need to change later.

- **Trigger:** `final_output.status == "resolved"` and `acceptance == "accepted"`
  (`output-schema.md` §6). Escalated or rejected cases are never indexed.
- **Chroma:** `index_case` embeds `subject + ". " + body` (the same embed-text rule as the
  dataset, `rag-design.md` §3) and upserts into the single `support_cases` collection with
  `id=ticket_id`, `metadata={"source": "agent_resolved", "queue": ..., "type": ...,
  "priority": ..., "answer_class": "resolution", "cluster_size": 1, ...}` — the same metadata
  shape dataset rows carry, so `rag/dense.py` and `rag/queries.py` don't need a source-specific
  code path (`architecture.md` §2.1: "told apart by metadata").
- **FTS5:** a row is inserted into `dataset_tickets_fts` (`rag-design.md` §6 — "a plain,
  separately-populated FTS5 table is the only workable shape", built to receive rows from both
  `dataset_tickets` and `cases`) with `case_id=ticket_id`, `source='agent_resolved'`,
  `subject`/`body` = the ticket's normalised text, `answer` = the resolution text from
  `final_output.resolution`, `tags` = the joined `classification.tags`, and the same
  `queue`/`type`/`answer_class` `UNINDEXED` columns dataset rows carry.
- **`cases.indexed_at`** is then set to the current timestamp, so a case is indexed at most
  once — `index_case` first checks `indexed_at IS NULL` before doing either write, which also
  makes the whole step safely re-runnable.
- Once indexed, the case is retrievable by `rag/queries.search()` exactly like a dataset
  canonical: `EvidenceEntry.source == "agent_resolved"` for citations of it
  (`output-schema.md` §3.1), and it never re-enters `dataset_tickets` — that table stays a
  read-only mirror of the HF ingest (`architecture.md` §2.2).

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
