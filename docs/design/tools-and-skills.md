# Tools and Skills

> **Status:** Draft v1 · CP4. **Companion docs:** `graph-design.md` §3 (node table), `rag-design.md`
> (retrieval internals the tools wrap), `case-persistence.md`/`memory-design.md` (tables read).
> Tools are `@tool`-decorated functions in `autosupport/tools/`; `build_tools(customer_id,
> ticket_id)` assembles the bound set for one graph run. Skills are `skills/*.md`, loaded by name via
> `skills.py`'s `load_skill(name)` (cached, raises `FileNotFoundError` if missing — no silent
> fallback, so deleting a skill file visibly changes behaviour rather than nothing).

## 1. Tools

| Tool | Signature | Returns | Notes |
|---|---|---|---|
| `search_similar_tickets` | `(query: str, k: int = 8, queue: str \| None = None) -> list[dict]` | One dict per hit: `case_id, subject, queue, type, priority, answer_class, cluster_size, similarity, score, body_snippet, answer_snippet, tags`. | Runs the full hybrid pipeline (`rag/queries.search`). Targeted, mid-investigation retrieval — distinct from `retrieve_initial`'s one broad pass. Its hits are also merged into `state.retrieved_cases` (state-schema.md §2.5 lists `tools` as a writer). |
| `get_ticket_by_id` | `(case_id: str) -> dict` | `{case_id, source, subject, body, answer, answer_class, cluster_size, queue, type, priority, tags}`, or `{"error": "not found"}`. | Small-to-big retrieval (decisions.md D7): search returns snippets, this returns the full record — dataset (`HF-`) or an agent-resolved/customer-history case (`T-`). |
| `get_customer_history` | `(limit: int = 5) -> list[dict]` | `{case_id, status, subject, queue, resolution_snippet, updated_at}` per case. | `customer_id` is bound by closure at tool-build time, never a model-supplied argument — a ticket can only ever see its own customer's history. Same query as `memory-design.md` §3. |
| `compute_queue_stats` | `(queue: str \| None = None) -> dict` | `{queue, distinct_cases, by_type, by_priority, by_answer_class}` — counts over canonical `dataset_tickets` rows, one queue or the whole corpus. | Corpus-wide pattern context (e.g. "how common is this queue/type"), not per-ticket evidence. |
| `escalate_ticket` | `(reason: str, target_queue: str \| None = None) -> dict` | `{"acknowledged": true, "reason": ..., "target_queue": ...}` | **CP4 scope note:** advisory only — it lets the model formally flag "this needs a human" mid-investigation, logged in `tool_log` and visible to the model on its next turn. It does not itself route the graph to an escalation outcome; `assess_evidence`/`escalate` (CP5) own that decision. |

Every tool call is wrapped by the `tools` node (not LangGraph's prebuilt `ToolNode`, so it can
also write `tool_log` and merge `search_similar_tickets` hits into `retrieved_cases`): a failed
call becomes `{"error": ...}` with `ToolCallRecord.ok = false` rather than crashing the graph.

## 2. Node bindings

| Node | Tools bound | Skill(s) loaded |
|---|---|---|
| `triage` | — | `triage.md` (fixed) |
| `investigate` | all five, via `build_tools(customer_id, ticket_id)` | `investigation.md` (fixed) + `escalation.md` when `triage` put `"escalation"` in `active_skills` |
| `resolve` | — | `customer_response.md` (fixed) |
| `escalate` (CP5) | — | `escalation.md` (fixed) |

`active_skills` is the only *dynamic* selection: `triage` decides, from the ticket and its
retrieved neighbours, whether this looks escalation-bound (e.g. a refund, an account/legal
matter, an outage) and if so adds `"escalation"` — `investigate` then layers that skill's
guidance on top of its own fixed one. Nothing else is optional; skills are never concatenated
into one prompt (CLAUDE.md).

## 3. Skills

- **`triage.md`** — classify `queue`/`type`/`priority`/`tags` from the ticket, the retrieved
  neighbours' metadata and any customer profile; write one or two sentences of rationale a
  human reviewer can check; decide whether the `escalation` skill should also be active for
  this ticket's investigation.
- **`investigation.md`** — the ReAct loop's operating instructions: form a hypothesis, use the
  five tools to gather and check evidence (not guess), prefer `resolution`-class cases as
  grounding, and stop calling tools once the evidence is enough to state a hypothesis with
  supporting/contradicting case IDs — don't call a tool "just in case."
  This is the loop `graph-design.md` §1 requires to make real decisions rather than a fixed
  sequence, per `REQUIREMENTS.md` §3's "Important implementation rule."
- **`escalation.md`** — how to write a target queue, a one/two-sentence reason and a handoff
  summary a human agent can act on without re-reading the whole ticket; when a case actually
  warrants escalation (CP5's `escalate` node) versus just flagging risk mid-investigation
  (`escalate_ticket`, CP4).
- **`customer_response.md`** — draft the customer-facing resolution grounded only in cited
  evidence, cite every case relied on as `[case_id]`, and say plainly when the evidence
  doesn't support a confident fix rather than inventing one (output-schema.md §1 principle 2).
