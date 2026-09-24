# Memory Design — the `customers` table and the write policy

> **Status:** v2 · CP6 (read side CP3, write side CP6).
> **Companion docs:** `state-schema.md` §2.3 (`CustomerMemory`, `customer_history` in state),
> `case-persistence.md` (the sibling `cases` table, read by `load_memory`; §5 is the *index*
> policy, which is a different decision from the *memory* policy here),
> `graph-design.md` node 16 (`update_memory`).
> **Code:** `graph/memory.py` (the policy, pure functions), `graph/nodes/update_memory.py`,
> `graph/nodes/load_memory.py`, `store/customers.py`.

---

## 1. Principle (CLAUDE.md constraint 2)

**No synthetic customers over the historical corpus.** The dataset is anonymous; it has no
`customer_id`. `customer_id` originates only from a submitted ticket, at runtime. `customers`
and `cases` are the only sources `load_memory` reads — never `dataset_tickets`. A customer who
has never submitted a ticket simply has no row, and `load_memory` returns `customer_profile =
None`, `customer_history = []`. This is expected for every customer's first ticket, not an
error state.

Two kinds of cross-ticket memory, deliberately separate:

| | What | Written by | Read by |
|---|---|---|---|
| **Profile** (`customers`) | Distilled durable facts about *this customer* | `update_memory` (§4) | `load_memory` → `triage`, `investigate`, `assess_evidence`, `resolve` |
| **History** (`cases`) | The customer's own earlier tickets, verbatim outcomes | `persist_case` | `load_memory`, `get_customer_history` tool |

(The third kind of growth — resolved cases becoming retrievable *for every customer* — is
`index_case`, `case-persistence.md` §5. It is knowledge, not memory, and never carries a
`customer_id`.)

---

## 2. DDL

```sql
CREATE TABLE IF NOT EXISTS customers (
    customer_id   TEXT PRIMARY KEY,
    profile       TEXT NOT NULL,   -- CustomerMemory, JSON
    updated_at    TEXT NOT NULL
);
```

One row per customer, one JSON blob. Nothing ever queries inside the profile, so it has no
columns of its own.

```python
class CustomerMemory(BaseModel):
    customer_id: str
    facts: dict[str, str] = {}        # keys from FACT_KEYS only (§4 W2)
    flags: list[str] = []             # computed by code (§4 W6)
    tried_fixes: list[str] = []       # FIFO, max 15
    preferences: dict[str, str] = {}  # keys from PREFERENCE_KEYS only
    provenance: dict[str, str] = {}   # "facts.<key>" / "preferences.<key>" -> ticket_id that last wrote it
```

`provenance` is bookkeeping: `autosupport memory` renders it ("os: Windows 11 (from
T-20260924-…)") so any remembered fact can be traced to the ticket that stated it. It is never
put into a prompt.

---

## 3. Reading (`load_memory`)

Runs in parallel with `retrieve_initial`, straight after `intake`.

- **`customer_profile`**: `SELECT profile FROM customers WHERE customer_id=?` →
  `CustomerMemory`, or `None` if no row.
- **`customer_history`**: the customer's *other* open cases plus their last 5 terminal
  (resolved/escalated) cases, from `cases` (`store/cases.py::history_for`, two queries so the
  "last 5" cap doesn't apply to open cases). Each becomes a `CaseSummary(case_id, status,
  subject, queue, resolution_snippet, updated_at)`. The current ticket is excluded — `intake`
  has already inserted its row.
- **Used**: `graph/memory.py::profile_block` renders profile + history once, identically, into
  the prompts of `triage`, `investigate` and `resolve` (so `resolve` can honour a preference
  and avoid re-suggesting a tried fix); `assess_evidence` reads facts/history for
  `history_contradicts`, and `flags` for the `repeat_unresolved` escalation rule.

---

## 4. Writing (`update_memory`) — the write policy

Runs once per finished ticket, in parallel with `index_case`, after `persist_case`. Fast tier,
`.with_structured_output(MemoryUpdate, method="json_schema")`, then **every rule below is
enforced in code** (`graph/memory.py::apply_update`) — the prompt states the rules too, but
the model's output is only a *proposal*; what is stored is what survives the code.

**The test for every item:** *would this change how we handle this customer's **next** ticket,
on a different subject?* If not, it is ticket-local — it stays in that ticket's own
`cases.final_output`, still reachable through `customer_history`, and is not written to the
profile.

### W1 — Only what the customer said

Source text is exactly what the customer wrote in this ticket: subject, body, answers to
clarification questions, and rejection feedback. Each proposed item must carry a verbatim
`quote` from that text; **an item whose quote is not found in it is dropped** (whitespace- and
case-normalised substring match). By construction this excludes: historical dataset cases, the
agent's own analysis/resolution, other customers' text, and anything inferred ("sounds like an
enterprise customer").

### W2 — Only durable kinds (closed vocabulary)

| Kind | Allowed keys | Example |
|---|---|---|
| `facts` | `product`, `os`, `software_version`, `plan`, `deployment`, `integration` | `os: Windows 11` |
| `preferences` | `contact_channel`, `technical_level`, `language` | `technical_level: not comfortable with command line` |
| `tried_fixes` | free text, one fix per entry | `restarted the sync service` |

The keys are `Literal` enums in the output schema, so a key outside the list cannot be
emitted. A **tried fix** is either a fix the customer says they already tried, or a fix the
agent proposed that the customer **rejected** as not working (from their rejection feedback) —
so the next investigation doesn't suggest it again.

### W3 — Explicitly discarded

| Discarded | Why |
|---|---|
| This incident's specifics: error codes/messages, timestamps, stack traces, order/invoice/ticket numbers | True only of this incident; a stale error code misleads the next investigation |
| The fix that **worked** | Already captured twice: the case is indexed as `agent_resolved` (retrievable by similarity) and is in `customer_history`. Copying it into `facts` would duplicate and drift |
| One-off circumstances ("travelling next week", "deadline Friday") | Not durable |
| Anything not stated by the customer (inference, guesses from tone) | W1 |
| Secrets and contact PII: passwords, API keys/tokens, e-mail addresses, phone numbers, card/account numbers (long digit runs) | Never stored. Code drops any value **or quote** matching the sensitive-data pattern, whatever the model proposed |

### W4 — Written for every outcome

Resolved and escalated tickets both update memory. Escalations are where the most useful
memory comes from: what was tried and rejected. (Contrast `index_case`, which only indexes
accepted resolutions — `case-persistence.md` §5.)

### W5 — Merge rules

- `facts`, `preferences`: **overwrite by key** — they describe current state (a customer who
  upgraded OS is no longer on the old one; keeping both would let the stale one be used).
  `provenance["facts.<key>"] = ticket_id` records who last wrote it.
- `tried_fixes`: **append**, dedupe case-insensitively, keep the most recent 15 (FIFO). The
  closed key vocabulary already bounds `facts` (6) and `preferences` (3), so the profile stays
  small enough to inline into every prompt after any number of tickets.

### W6 — Flags are computed, never extracted

- `repeat_unresolved`: set iff the customer has **≥ 2 `escalated` cases in the last 90 days**
  (including this one). Recomputed on every write, so it also clears once they age out. It
  feeds `assess_evidence`'s escalation rule. A countable fact is computed by code, not asked
  of a model.
- `vip`: never set — no source of truth exists (no billing system). Kept in the model only
  because `assess_evidence` already reads it.

### Output schema

```python
class RememberedFact(BaseModel):       key: FactKey;        value: str; quote: str
class RememberedPreference(BaseModel): key: PreferenceKey;  value: str; quote: str
class RememberedFix(BaseModel):        fix: str;            quote: str

class MemoryUpdate(BaseModel):
    facts: list[RememberedFact] = []
    preferences: list[RememberedPreference] = []
    tried_fixes: list[RememberedFix] = []
    reasoning: str        # why each item qualifies; visible in the LangSmith trace, not stored
```

### Upsert

```sql
INSERT INTO customers (customer_id, profile, updated_at) VALUES (?, ?, ?)
ON CONFLICT(customer_id) DO UPDATE SET profile=excluded.profile, updated_at=excluded.updated_at
```
`update_memory` reads the current row (or starts empty), applies W1–W6, writes once. It
writes no graph state (external write only), so running in parallel with `index_case` needs
no reducer.

---

## 5. What's deliberately not here

| Item | Where it actually lives | Why not `customers` |
|---|---|---|
| The ticket-by-ticket record | `cases.final_output` | `customers` is the distilled summary, not an archive |
| Dataset-derived customer info | Nowhere | Constraint 2: the archive is anonymous |
| Similarity search over a customer's history | Not supported | History is a small complete list, loaded directly |
| Resolved cases as knowledge for all customers | Chroma + FTS5 via `index_case` | Knowledge, not memory — `case-persistence.md` §5 |
