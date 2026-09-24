# Requirements & Deliverables Checklist

> **Purpose:** This file is the implementation source-of-truth. Follow it while building the project so that no required behavior, tool, artifact, or demo path is missed.
>
> **Scope:** Derived from the provided assessment only. Do **not** introduce external product/domain knowledge beyond the specified ticket dataset.

---

## 1. Project Objective

Build a **locally runnable, LangGraph-based AI Support Investigation Agent** that can:

- investigate new support tickets using historical cases;
- use tools and reusable skills;
- maintain short-term and long-term memory;
- ask the user for clarification when evidence is insufficient;
- resume the same investigation after clarification;
- produce a grounded resolution or escalation;
- evaluate its own result/evidence;
- continuously grow its trusted case knowledge from newly resolved cases.

### Domain knowledge restriction — mandatory

The **only domain knowledge source** for the assessment is the provided English subset (~28K records) of:

`Tobi-Bueck/customer-support-tickets`

Do **not** use external product documentation or any other domain knowledge source.

---

# 2. Mandatory Technology / Tooling Requirements

## Required

- [ ] **Python**
- [ ] **LangGraph** as the workflow/orchestration layer
- [ ] **LangSmith evaluators** and evaluation datasets for basic evaluation coverage
- [ ] A **vector database** for historical-ticket retrieval
  - Technology is our choice.
- [ ] A **local structured database** for storing new/open/final cases and customer-level memory
  - Technology is our choice.
- [ ] Dependency file, e.g. `requirements.txt` or `pyproject.toml`
- [ ] Secrets/API keys kept **outside source control**

## Allowed / optional

- [ ] A **cloud LLM may be used**.
- [ ] CLI **or** API is sufficient.
- [ ] Frontend is **not mandatory**.
- [ ] LangGraph parallel execution is a **plus**, not a requirement.

---

# 3. LangGraph Requirements — Mandatory

The graph must include all of the following:

- [ ] **Multiple nodes**
- [ ] **Conditional routing**
- [ ] At least one **meaningful loop/retry path**
- [ ] **Interrupt/resume** when more user information is required
- [ ] A clearly defined **graph state**
- [ ] A coherent agentic architecture appropriate for support investigation

### Required behavior of the loop

A valid important loop is:

`Evaluate / Evidence Check -> unsupported or insufficient -> retrieve/investigate again`

The agent may also perform targeted/follow-up retrieval after forming a hypothesis.

### Important implementation rule

Do **not** reduce the graph to a completely fixed sequence of tool calls. The model must make meaningful decisions about what to do next.

---

# 4. Case-Based RAG Requirements — Mandatory

- [ ] Index historical tickets in a vector database.
- [ ] Retrieve **multiple relevant historical cases** for a new ticket.
- [ ] Use ticket text/content in retrieval.
- [ ] Use useful metadata when appropriate, including:
  - queue
  - type
  - priority
  - tags
- [ ] Compare evidence across multiple historical cases.
- [ ] Handle conflicting historical evidence.
- [ ] Detect insufficient evidence.
- [ ] Support follow-up / targeted retrieval after an initial hypothesis when useful.

### Explicitly prohibited shortcut

Do **not** simply retrieve the nearest ticket and return its historical answer.

The final answer must be grounded in the retrieved evidence and tool results.

---

# 5. Memory Requirements — Mandatory

## Short-term memory

- [ ] Preserve ticket/conversation context across multiple turns in the **same thread**.
- [ ] The clarification/resume flow must retain enough state to continue the original investigation.

## Long-term memory

- [ ] Persist useful **customer-level information** across separate conversations.
- [ ] Decide selectively what is worth remembering.
- [ ] Avoid storing unnecessary information.
- [ ] Demonstrate reuse of long-term memory in the final demo.

---

# 6. Tools — Minimum 4 Meaningful Tools

Implement **at least four meaningful tools**.

The assessment gives these examples:

1. Search similar historical tickets
2. Get a ticket by ID
3. Retrieve customer history
4. Compute queue/tag statistics
5. Update ticket classification
6. Escalate a ticket

## Non-negotiable tool behavior

- [ ] At least **4 meaningful tools** exist.
- [ ] The model decides **when** a tool is needed.
- [ ] Tool execution is **not fully hard-coded** into a fixed sequence.

## Recommended minimum tool set for this project

To satisfy the requirement clearly, implement at least:

- [ ] `search_similar_tickets(...)`
- [ ] `get_ticket_by_id(...)`
- [ ] `get_customer_history(...)`
- [ ] `compute_case_statistics(...)`
- [ ] `update_classification(...)` *(recommended additional tool)*
- [ ] `escalate_ticket(...)` *(recommended additional tool)*

Using 5–6 solid tools gives more margin than implementing exactly four.

---

# 7. Reusable Skills / Instruction Modules — Minimum 3

Implement **at least three reusable skills/instruction modules**.

Examples from the assessment:

- Ticket triage
- Incident investigation
- Escalation
- Customer response

## Mandatory skill behavior

- [ ] At least **3 reusable skills** exist.
- [ ] Skills are selected/loaded **dynamically**.
- [ ] Do **not** place every skill into one giant system prompt.

## Recommended project skill set

- [ ] `ticket_triage`
- [ ] `incident_investigation`
- [ ] `escalation_decision`
- [ ] `customer_response`

Four skills are preferable if implementation remains clean.

---

# 8. Evaluators — Mandatory

Implement basic evaluation coverage using **LangSmith evaluators and evaluation datasets**.

Evaluation must cover:

- [ ] Response quality
- [ ] Retrieval relevance
- [ ] Tool usage
- [ ] Overall agent behavior

The runtime workflow must also contain an evaluation/evidence-check step before the case is finalized.

If the result is unsupported:

`Evaluate -> Retrieve / Investigate Again`

---

# 9. Continuously Growing Case Knowledge — Mandatory

Every new customer query must be stored in a **local structured database**.

## On ticket creation

- [ ] Persist the incoming ticket as an **open case**.
- [ ] Open/new cases are available as customer history.

## After accepted/final resolution

Persist:

- [ ] final classification;
- [ ] final resolution;
- [ ] final case/result as appropriate.

Then:

- [ ] make the resolved case available for future retrieval;
- [ ] index the trusted resolved case into the searchable case knowledge.

### Required lifecycle

```text
New Ticket
    -> Retrieve History
    -> Investigate / Ask User
    -> Retrieve History (if needed)
    -> Resolve or Escalate
    -> Evaluate
    -> Persist Final Case
    -> Index for Future Retrieval
```

---

# 10. Expected End-to-End Agent Behavior

The implementation should demonstrate this full path:

1. [ ] Receive a new support ticket.
2. [ ] Persist it as an open case.
3. [ ] Load relevant short-term state.
4. [ ] Load customer long-term memory.
5. [ ] Classify/triage using the ticket and historical evidence.
6. [ ] Retrieve multiple relevant historical cases.
7. [ ] Inspect prior resolutions/evidence.
8. [ ] Select relevant skill(s).
9. [ ] Use tools as needed to investigate.
10. [ ] If evidence is insufficient or conflicting, ask a **focused clarification question**.
11. [ ] Interrupt execution while waiting for the answer.
12. [ ] Resume the same graph/thread after the user responds.
13. [ ] Perform additional retrieval/investigation if needed.
14. [ ] Produce a grounded resolution **or** escalation.
15. [ ] Run evaluation/evidence-checking.
16. [ ] If unsupported, loop back to retrieval/investigation.
17. [ ] Persist the final result.
18. [ ] Index the resolved case for future retrieval.

---

# 11. Minimum Output Schema — Mandatory

For **each completed case**, return a structured result containing **at minimum**:

```json
{
  "classification": {
    "queue": "...",
    "type": "...",
    "priority": "...",
    "tags": []
  },
  "evidence": [
    {
      "ticket_id": "...",
      "summary": "..."
    }
  ],
  "analysis": "Concise reasoning based on retrieved evidence and tool results.",
  "resolution": "Recommended final response/action.",
  "escalation": {
    "required": false,
    "reason": "..."
  },
  "confidence": 0.0
}
```

### Required fields

- [ ] **Classification**
  - queue
  - type
  - priority
  - relevant tags
- [ ] **Evidence**
  - historical case IDs and/or summaries
- [ ] **Analysis**
  - concise reasoning based on retrieved evidence + tool results
- [ ] **Resolution**
  - recommended final response/action
- [ ] **Escalation**
  - whether escalation is required
  - why
- [ ] **Confidence**
  - confidence score for the resolution

Field naming can differ, but all required information must be present.

---

# 12. Required Deliverables

## A. Runnable source code

- [ ] Complete runnable source code
- [ ] Runs **locally**
- [ ] Dependency file included
- [ ] Clear setup instructions
- [ ] Secrets/API keys are excluded from source control
- [ ] `.env.example` is recommended, but real `.env` / keys must not be committed

## B. README

The README must cover:

- [ ] Architecture
- [ ] LangGraph graph design
- [ ] State design
- [ ] Memory strategy
- [ ] RAG approach
- [ ] Tool design
- [ ] Skill design
- [ ] Limitations
- [ ] Setup/run instructions

Useful additions, although not explicitly required:

- project structure;
- data ingestion/indexing instructions;
- environment variables;
- demo commands;
- evaluation commands;
- graph diagram.

## C. Dataset ingestion/indexing command or script

Provide a **small script or command** that:

- [ ] ingests the provided historical ticket dataset;
- [ ] indexes it for retrieval.

Example naming:

```text
scripts/ingest.py
```

or

```bash
python -m app.ingest
```

## D. Demonstration

The demo must explicitly cover **all four** of these scenarios:

### Demo 1 — Normal resolution
- [ ] New ticket enters the graph.
- [ ] Relevant historical cases are retrieved.
- [ ] Agent resolves the ticket normally.

### Demo 2 — Clarification + resume
- [ ] Agent determines that information is insufficient/conflicting.
- [ ] Agent asks a focused clarification question.
- [ ] Graph execution interrupts.
- [ ] User provides the missing information.
- [ ] Graph resumes and reaches a result.

### Demo 3 — Long-term memory reuse
- [ ] Information learned about a customer in an earlier conversation is persisted.
- [ ] A separate later conversation for that customer retrieves and uses appropriate long-term memory.

### Demo 4 — Newly resolved case retrieval
- [ ] Resolve a new case.
- [ ] Persist the final classification/resolution.
- [ ] Index the resolved case.
- [ ] Submit another relevant ticket.
- [ ] Demonstrate that the newly resolved case is now retrievable as historical evidence.

---

# 13. Recommended Project Structure

This structure is not mandated by the assessment, but maps cleanly to every requirement:

```text
project/
├── README.md
├── ASSESSMENT_REQUIREMENTS.md
├── requirements.txt              # or pyproject.toml
├── .env.example
├── .gitignore
│
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── schemas.py
│   ├── graph.py
│   ├── state.py
│   │
│   ├── nodes/
│   │   ├── intake.py
│   │   ├── triage.py
│   │   ├── retrieve.py
│   │   ├── investigate.py
│   │   ├── clarify.py
│   │   ├── resolve.py
│   │   ├── evaluate.py
│   │   └── persist.py
│   │
│   ├── tools/
│   │   ├── search_cases.py
│   │   ├── get_ticket.py
│   │   ├── customer_history.py
│   │   ├── statistics.py
│   │   ├── classification.py
│   │   └── escalation.py
│   │
│   ├── skills/
│   │   ├── triage.md
│   │   ├── investigation.md
│   │   ├── escalation.md
│   │   └── customer_response.md
│   │
│   ├── rag/
│   │   ├── embeddings.py
│   │   ├── index.py
│   │   └── retriever.py
│   │
│   ├── memory/
│   │   ├── short_term.py
│   │   └── long_term.py
│   │
│   ├── storage/
│   │   ├── cases.py
│   │   └── database.py
│   │
│   └── evaluation/
│       ├── evaluators.py
│       └── datasets.py
│
├── scripts/
│   ├── ingest.py
│   └── demo.py
│
└── tests/
    ├── test_retrieval.py
    ├── test_tools.py
    ├── test_memory.py
    └── test_graph_paths.py
```

This structure is **recommended**, not an explicit assessment requirement.

---

# 14. Suggested Graph Shape

Not prescribed by DCKAP, but the following topology visibly satisfies the requested behaviors:

```text
START
  |
  v
Persist Open Case
  |
  v
Load Memory / History
  |
  v
Triage / Classification
  |
  v
Retrieve Historical Cases
  |
  v
Investigation + Dynamic Tools/Skills
  |
  +-------------------------------+
  |                               |
  | enough evidence               | insufficient/conflicting
  v                               v
Generate Resolution          Ask Clarification
  |                               |
  |                            INTERRUPT
  |                               |
  |                         USER RESPONSE
  |                               |
  |                            RESUME
  |                               |
  +<-------- Targeted Retrieval <-+
  |
  v
Evidence / Quality Evaluation
  |
  +-------------------------------+
  | supported                     | unsupported
  v                               |
Resolve / Escalate                 |
  |                               |
  |                      Investigate / Retrieve Again
  |                               |
  +-------------------------------+
  |
  v
Persist Final Case
  |
  v
Index Trusted Resolved Case
  |
  v
END
```

This graph is a recommendation. The assessment explicitly says no specific architecture is prescribed.

---

# 15. Things We Should NOT Waste Time On

The assessment explicitly says this is **not expected to be production-ready**.

Prioritize a coherent architecture and important end-to-end paths.

Do **not** spend significant time on:

- elaborate frontend/UI;
- deployment infrastructure;
- production-scale infrastructure;
- unrelated DevOps;
- visual polish that does not demonstrate agent behavior.

A CLI or API is sufficient.

---

# 16. Hard Compliance Checks Before Submission

Before calling the project complete, verify every item below.

## Architecture

- [ ] Python is used.
- [ ] LangGraph is the orchestrator.
- [ ] Graph has multiple nodes.
- [ ] Graph has conditional routing.
- [ ] Graph has a meaningful retry/loop.
- [ ] Interrupt/resume works.

## RAG

- [ ] Historical dataset is indexed in a vector DB.
- [ ] Retrieval returns multiple cases.
- [ ] Metadata is used where appropriate.
- [ ] Agent compares evidence rather than copying nearest answer.
- [ ] Targeted/follow-up retrieval is possible.

## Memory

- [ ] Short-term thread state works.
- [ ] Long-term customer memory works across conversations.
- [ ] Only useful long-term information is persisted.

## Tools + skills

- [ ] >= 4 meaningful tools.
- [ ] Model determines when tools are needed.
- [ ] >= 3 reusable skills.
- [ ] Skills are dynamically selected/loaded.

## Evaluation

- [ ] LangSmith evaluation exists.
- [ ] Response quality evaluated.
- [ ] Retrieval relevance evaluated.
- [ ] Tool usage evaluated.
- [ ] Overall behavior evaluated.
- [ ] Runtime evidence-check/loop exists.

## Persistent knowledge growth

- [ ] Every incoming query is stored as an open case.
- [ ] Final resolved classification is stored.
- [ ] Final resolution is stored.
- [ ] Newly resolved trusted cases are indexed.
- [ ] Newly indexed case can be retrieved later.

## Final output

- [ ] Classification included.
- [ ] Evidence included.
- [ ] Analysis included.
- [ ] Resolution included.
- [ ] Escalation decision + reason included.
- [ ] Confidence score included.

## Deliverables

- [ ] Runnable source code.
- [ ] Dependency file.
- [ ] Setup instructions.
- [ ] Complete README.
- [ ] Ingestion/indexing script or command.
- [ ] Demo: normal resolution.
- [ ] Demo: clarification + interrupt/resume.
- [ ] Demo: long-term memory reuse.
- [ ] Demo: newly resolved case retrieved later.
- [ ] API keys/secrets not committed.

## Knowledge boundary

- [ ] The provided ticket dataset is the **sole domain knowledge source**.
- [ ] No external product documentation has been used for answers.

---

# 17. Definition of Done

The project is ready to submit only when:

1. it runs locally from documented setup steps;
2. the dataset can be ingested/indexed with the provided command/script;
3. the LangGraph workflow performs real conditional investigation rather than a fixed linear pipeline;
4. the agent can retrieve and compare multiple historical cases;
5. clarification can interrupt and later resume the same workflow;
6. short-term and long-term memory are both demonstrated;
7. at least four meaningful tools and three dynamically selected skills are implemented;
8. LangSmith evaluation coverage exists;
9. every new query is persisted, and finalized cases become future searchable knowledge;
10. the required structured output fields are returned;
11. all four mandatory demo scenarios work end-to-end;
12. the README explains the architecture, graph, state/memory, RAG, tools, skills, limitations, and setup.

---

# 18. One-Line Rule for the Coding Agent

> **Do not mark the project complete until every mandatory checkbox in `REQUIREMENTS.md` is satisfied or explicitly documented as a known limitation. Prioritize the requirements over optional refactoring, UI, infrastructure, or unrelated improvements.**
