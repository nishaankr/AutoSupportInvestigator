# Memory Design — the `customers` table and the write policy

> **Status:** Draft v1 · long-term, cross-ticket memory.
> **Companion docs:** `state-schema.md` §2.3 (`CustomerMemory`, `customer_history` in state),
> `case-persistence.md` (the sibling `cases` table, read by `load_memory`),
> `graph-design.md` node 16 (`update_memory`).
> **Scope:** CP3 builds only the read side (`load_memory`, returning `None`/`[]` for every
> customer since nothing has written yet). The write side (`update_memory`) is CP6. Both are
> specified in full here so the schema doesn't migrate between CP3 and CP6.

---

## 1. Principle (CLAUDE.md constraint 2)

**No synthetic customers over the historical corpus.** The dataset is anonymous; it has no
`customer_id`. `customer_id` originates only from a submitted ticket, at runtime. `customers`
and `cases` are the only sources `load_memory` reads — never `dataset_tickets`. A customer who
has never submitted a ticket simply has no row, and `load_memory` returns `customer_profile =
None`, `customer_history = []`. This is expected for every customer's first ticket, not an
error state.

---

## 2. DDL

```sql
CREATE TABLE IF NOT EXISTS customers (
    customer_id   TEXT PRIMARY KEY,
    profile       TEXT NOT NULL,   -- CustomerMemory, JSON, minus customer_id (redundant with the key)
    updated_at    TEXT NOT NULL
);
```

One row per customer, one JSON blob. `CustomerMemory` (`state-schema.md` §3) is small and has
no query pattern that needs its own columns — nothing ever does `WHERE facts->>'plan' = ...`.
If that changes later (e.g. an eval wants to bucket customers by plan), it's a column added
then, not now.

`CustomerMemory` recap, for reference:
```python
class CustomerMemory(BaseModel):
    customer_id: str
    facts: dict[str, str] = {}       # e.g. {"product": "QNAP TS-464", "plan": "enterprise", "os": "Windows 11"}
    flags: list[str] = []            # e.g. ["vip", "repeat_unresolved"]
    tried_fixes: list[str] = []
    preferences: dict[str, str] = {}
```

---

## 3. Reading (`load_memory`, CP3)

```python
def load_memory(customer_id: str) -> tuple[CustomerMemory | None, list[CaseSummary]]:
    profile = customers.get(customer_id)                     # None if no row
    history = cases.history_for(customer_id, exclude_ticket_id=state["ticket_id"], limit=5)
    return profile, history
```

- **`customer_profile`**: `SELECT profile FROM customers WHERE customer_id=?`, parsed with
  `CustomerMemory.model_validate_json`. `None` if no row — this is the common case until CP6
  writes anything, and stays the common case for a customer's first-ever ticket after CP6
  ships too.
- **`customer_history`**: "the customer's open cases plus their last N=5 resolved or
  escalated cases" (`state-schema.md` §2.3), read from `cases`:
  ```sql
  SELECT ticket_id, status, subject, final_output FROM cases
  WHERE customer_id = ? AND ticket_id != ? AND status != 'open'
  ORDER BY updated_at DESC LIMIT 5
  UNION ALL
  SELECT ticket_id, status, subject, final_output FROM cases
  WHERE customer_id = ? AND ticket_id != ? AND status IN ('open', 'investigating', 'awaiting_user')
  ORDER BY updated_at DESC
  ```
  (implemented as two queries in `store/cases.py::history_for`, not literally one UNION, so the
  "last 5 terminal" cap doesn't accidentally apply to the open-cases half). Each row becomes a
  `CaseSummary(case_id, status, subject, queue=classification.queue if present else None,
  resolution_snippet=final_output.resolution[:200] if resolved else None, updated_at)`.
- **`exclude_ticket_id`** is required, not optional: `intake` writes the current ticket's
  `cases` row (`status='open'`) *before* `load_memory` runs (they're parallel branches off
  `intake`, but `intake` itself completes first — `graph-design.md`'s fan-out is
  `intake → {load_memory, retrieve_initial}`), so without the exclusion every ticket would see
  itself as an "open case" in its own history. This is state-schema's §2.3 note read literally:
  "the customer's open cases" means the *other* open cases.
- **No join** between `customers` and `cases`: they answer different questions (a profile vs.
  a case list) and are read with two independent, simple queries. A join would only save one
  round-trip to the same SQLite file in-process — not worth the extra query complexity for two
  small tables.

---

## 4. Writing (`update_memory`, CP6)

Runs in parallel with `index_case`, after `persist_case` (`graph-design.md` node 16). Fast
tier, structured output, once per finished ticket (resolved or escalated — memory is written
regardless of outcome, since even an escalation teaches the agent what was tried and what
didn't work).

### 4.1 What's worth remembering (the graded write policy)

The model is shown the final `CaseResult` plus the ticket's `clarifications`, and asked to
extract only what would help a *future, different* ticket from the same customer. The prompt
draws this line explicitly, because it's the part that's easy to get wrong in both directions
(remembering too little defeats the point of long-term memory; remembering too much turns
`facts` into a transcript):

| Keep (durable, cross-ticket) | Discard (ticket-local) |
|---|---|
| A product/model/OS/plan the customer stated, likely to recur ("QNAP TS-464", "Windows 11", "enterprise plan") | The specific error message or timestamp of *this* incident |
| A fix that was tried and didn't work, so the next investigation doesn't re-suggest it | A fix that *did* work — that's provenance already captured by `agent_resolved` citing this ticket; repeating it in `facts` would duplicate what retrieval already surfaces |
| A stated preference ("prefers email", "not comfortable with CLI steps") | A one-off scheduling detail ("said they'd be traveling next week") |
| A flag the policy below derives (`vip`, `repeat_unresolved`) | Anything not stated or clearly implied — no inference chains, no guessing a plan tier from writing style |

The rule of thumb given to the model: **would this change how you'd handle their *next*
ticket, on a different subject?** If the answer is no, it's ticket-local and stays only in
that ticket's own `cases.final_output`, which is still reachable via `customer_history`.

### 4.2 Extraction schema (structured output)

```python
class MemoryUpdate(BaseModel):
    new_facts: dict[str, str] = {}          # merged into profile.facts, overwriting same keys
    new_tried_fixes: list[str] = []          # appended, deduplicated
    new_preferences: dict[str, str] = {}     # merged into profile.preferences
    reasoning: str                           # one line, why each item qualifies — logged, not stored in CustomerMemory
```

`reasoning` exists for the same reason confidence's components are persisted
(`output-schema.md` §4.6): so a low-effort or wrong extraction is diagnosable later, without
re-running the model. It's written to `tool_log`/`errors`-style logging, not into
`CustomerMemory` itself, which stays lean.

**Provenance.** Every `CustomerMemory` write also updates two bookkeeping fields the model
never sees or sets:
```python
facts_provenance: dict[str, str] = {}   # fact key -> the ticket_id that last wrote it
```
This isn't part of the `CustomerMemory` shown to the model or to `load_memory`'s callers —
it's a second column-free field folded into the same JSON blob, used only by `autosupport
memory <customer_id>` when rendering (so the CLI can show "plan: enterprise (from
T-20260910-...)" ) and by conflict resolution below.

### 4.3 Conflict / overwrite rule

`new_facts` **overwrites** the existing key. Facts are meant to be current state ("plan:
enterprise"), not a history — if it changes, the old value is wrong and keeping it around
would let a stale fact get cited in a future resolution. `facts_provenance` is what lets the
overwrite be explained if it ever looks wrong (e.g. a plan downgrade misclassified as a typo).

`new_tried_fixes` and `new_preferences.values()` for a repeated key are **appended**/merged
rather than overwritten — a tried fix stays tried even if it's tried again, and a stated
preference conflicting with an earlier one is unusual enough (per the model's `reasoning`) that
CP6 keeps both rather than silently dropping one; it's a UI/prompt-formatting problem (show
the most recent), not a data-loss problem to solve by overwriting.

### 4.4 `repeat_unresolved` flag rule

Set (idempotently — the extraction step checks before adding, since flags are a plain list, not
a set with unique semantics enforced by the DB) when: this customer has ≥ 2 `cases` rows with
`status = 'escalated'` in the last 90 days, **or** the current ticket cites the same
`classification.queue` + overlapping `tags` as a still-open or previously-escalated case in
`customer_history`. The rule lives in code (a plain SQL count + a tag-overlap check on the
history already loaded), not in the model's judgement — same reasoning as `neighbor_agreement`
in `state-schema.md`/CP3's plan: a countable corpus fact is computed by code, not asked of an
LLM. The flag itself is what `graph-design.md` §6's `escalation_rule_hit` reads
(`customer_profile.flags contains ... "repeat_unresolved"`).

`vip` is out of scope for this build — there's no source of truth for it (no billing system),
so it's never set automatically. The field stays in the model for forward-compatibility, and
`update_memory` never writes it.

### 4.5 Size caps

- `facts`: at most 20 keys. If an extraction would exceed it, the oldest-provenance key not
  reconfirmed by this ticket is dropped, keeping the extraction rather than rejecting it.
- `tried_fixes`: at most 15 entries, FIFO-trimmed the same way.
- `preferences`: at most 10 keys, same overwrite rule as facts (§4.3).
- `flags`: unbounded (there are only ever two possible values today, so this can't grow
  unboundedly in practice; the cap would only matter if more flag types are added later).

These caps exist so a customer's profile can be safely concatenated into a prompt at
`triage`/`resolve` without a token-budget surprise after dozens of tickets — the same "keep it
lean" principle `state-schema.md` §1.2 applies to `AgentState`, applied here to
`CustomerMemory`.

### 4.6 Upsert

```sql
INSERT INTO customers (customer_id, profile, updated_at) VALUES (?, ?, ?)
ON CONFLICT(customer_id) DO UPDATE SET profile=excluded.profile, updated_at=excluded.updated_at
```
`update_memory` reads the current row (or starts from an empty `CustomerMemory`), applies §4.3
and §4.5, and writes the merged result back in one statement.

---

## 5. What CP3 actually builds

Only `store/customers.py::get(customer_id) -> CustomerMemory | None` and the `history_for`
query in `store/cases.py` (case-persistence.md §4), wired into `load_memory`. No writer exists
yet, so every profile read at CP3 returns `None` and every history list reflects only prior
`cases` rows (there won't be any, in a fresh demo run, until a second ticket is submitted for
the same customer — which is exactly CP6's done criteria, not CP3's).

---

## 6. What's deliberately not here

| Item | Where it actually lives | Why not `customers` |
|---|---|---|
| The ticket-by-ticket record | `cases.final_output` | `customers` is the *distilled* cross-ticket summary, not a duplicate archive |
| Dataset-derived customer info | Nowhere — doesn't exist | Constraint 2: the archive is anonymous |
| Embeddings / retrievability of a customer's history | Not retrievable via `rag/queries.search()` at all | `customer_history` is loaded directly by `load_memory`, not retrieved by similarity — it's a small, complete list, not a corpus to search |
