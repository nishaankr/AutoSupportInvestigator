# Tools and Skills

> **Status:** v2 · reconciled at CP8. **Companion docs:** `graph-design.md` §3 (node table), `rag-design.md`
> (retrieval internals the tools wrap), `case-persistence.md`/`memory-design.md` (tables read).
> Tools are `@tool`-decorated functions in `autosupport/tools/`; `build_tools(customer_id,
> ticket_id)` assembles the bound set for one graph run. Skills are `skills/*.md`, loaded by name via
> `skills.py`'s `load_skill(name)` (cached, raises `FileNotFoundError` if missing — no silent
> fallback, so deleting a skill file visibly changes behaviour rather than nothing).

## 1. Tools

| Tool | Signature | Returns | Notes |
|---|---|---|---|
| `search_similar_tickets` | `(query: str, k: int = 8, queue: str \| None = None) -> list[dict]` | One dict per hit (`source` is `dataset` or `agent_resolved`, case-persistence.md §5): `case_id, source, subject, queue, type, priority, answer_class, cluster_size, similarity, score, body_snippet, answer_snippet, tags`. | Runs the full hybrid pipeline (`rag/queries.search`). Targeted, mid-investigation retrieval — distinct from `retrieve_initial`'s one broad pass. Its hits are also merged into `state.retrieved_cases` (state-schema.md §2.5 lists `tools` as a writer). |
| `get_ticket_by_id` | `(case_id: str) -> dict` | `{case_id, source, subject, body, answer, answer_class, cluster_size, queue, type, priority, tags}`, or `{"error": "not found"}`. | Small-to-big retrieval (decisions.md D7): search returns snippets, this returns the full record — dataset (`HF-`) or an agent-resolved/customer-history case (`T-`). |
| `get_customer_history` | `(limit: int = 5) -> list[dict]` | `{case_id, status, subject, queue, resolution_snippet, updated_at}` per case. | `customer_id` is bound by closure at tool-build time, never a model-supplied argument — a ticket can only ever see its own customer's history. Same query as `memory-design.md` §3. |
| `compute_queue_stats` | `(queue: str \| None = None) -> dict` | `{queue, distinct_cases, by_type, by_priority, by_answer_class}` — counts over canonical `dataset_tickets` rows, one queue or the whole corpus. | Corpus-wide pattern context (e.g. "how common is this queue/type"), not per-ticket evidence. |
| `escalate_ticket` | `(reason: str, target_queue: str \| None = None) -> dict` | `{"acknowledged": true, "reason": ..., "target_queue": ...}` | Advisory: it lets the model flag "this needs a human" mid-investigation, logged in `tool_log`. It does not itself route the graph; from CP5 a successful call feeds `assess_evidence`'s `escalation_rule_hit` (`action_beyond_agent`), and `assess_evidence`/`escalate` own the decision. |

Every tool call is wrapped by the `tools` node (not LangGraph's prebuilt `ToolNode`, so it can
also write `tool_log` and merge `search_similar_tickets` hits into `retrieved_cases`): a failed
call becomes `{"error": ...}` with `ToolCallRecord.ok = false` rather than crashing the graph.

## 2. Node bindings

| Node | Tools bound | Skill(s) loaded |
|---|---|---|
| `investigate` | all five, via `build_tools(customer_id, ticket_id)`, plus `submit_findings` (ends the round; its arguments are the `Findings` schema, D19) | `investigation.md` (fixed) + `escalation.md` when `triage` put `"escalation"` in `active_skills` |
| `resolve` | — | `customer_response.md` (fixed) |

`triage` and `escalate` make no model call since D19 (a label vote and a template), so they
load no skill; `skills/triage.md` was removed rather than left unused.

`active_skills` is the only *dynamic* selection: `triage` decides by rule, from the ticket and
its retrieved neighbours, whether this looks escalation-bound (escalation-class answers
dominate, or a high-stakes term such as a breach, outage or legal matter) and if so adds
`"escalation"` — `investigate` then layers that skill's guidance on top of its own fixed one.
Nothing else is optional; skills are never concatenated into one prompt (CLAUDE.md).

## 3. Skills

- **`investigation.md`** — the ReAct loop's operating instructions: form a hypothesis, use the
  five tools to gather and check evidence (not guess), prefer `resolution`-class cases as
  grounding, and stop calling tools once the evidence is enough to state a hypothesis with
  supporting/contradicting case IDs — don't call a tool "just in case."
  This is the loop `graph-design.md` §1 requires to make real decisions rather than a fixed
  sequence, per `REQUIREMENTS.md` §3's "Important implementation rule." Since D19 it also
  says how to end a round with `submit_findings` — cite every supporting/contradicting case,
  cluster every case marked `relevant=yes`, and list a missing fact only if it blocks the fix.
- **`escalation.md`** — layered onto `investigation` when `triage` flags the ticket as
  escalation-bound: decide early whether a person is needed (and say what in
  `requires_human_action` / `escalate_ticket`), and make the findings a usable handoff, since
  the `escalate` template is assembled from them. *v1 had this skill drive a model-written
  handoff in the `escalate` node; D19 made that node a Python template.*
- **`customer_response.md`** — draft the customer-facing resolution grounded only in cited
  evidence, cite every case relied on as `[case_id]`, and say plainly when the evidence
  doesn't support a confident fix rather than inventing one (output-schema.md §1 principle 2).
  The reply is sent as written: no placeholders like `[Your Name]`, no invented document or
  portal names (D22, found in the CP8 demo).

**Skill count:** three skills are in use — the REQUIREMENTS minimum. v1 had a fourth,
`triage.md`; `triage` stopped calling a model in D19, and the file was removed rather than
left unused.
