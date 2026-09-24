# RAG Design — Ingestion & Retrieval

> **Status:** Draft v1 · the ingest-time decisions (clustering threshold, embed-text
> composition, `answer_class` heuristics) are settled here, grounded in profiling the
> actual dataset, per `docs/project/checkpoints.md` CP1. Retrieval-time parameters (RRF
> constant, MMR λ, over-fetch size) are added at CP2 and are marked **Open — CP2** below.
> **Companion docs:** `architecture.md` §2.1/§2.1b/§4.1 (the six ingest stages),
> `decisions.md` D3–D6, `output-schema.md` §4.3–4.4 (τ_rel and `SIM_CEILING` consume the
> similarity numbers profiled here).

---

## 1. Dataset profile

Profiled directly against `Tobi-Bueck/customer-support-tickets` (not assumed) at CP0/CP1
prep, using `datasets` to pull the dataset and `sentence-transformers` with the locked
embedding model (`BAAI/bge-small-en-v1.5`) to probe similarity. The English-filtered
snapshot is cached at `data/raw/dataset_tickets_en.parquet` (git-ignored, regenerable by
`ingest/load.py`).

### 1.1 Size and the English filter (resolves D6)

| | rows |
|---|---|
| Total (`train` split) | 61,765 |
| `language == 'de'` | 33,504 |
| `language == 'en'` | **28,261** |

**D6 resolved:** filtering on `language == 'en'` alone, with no filter on `version`,
yields 28,261 rows — matching the brief's "~28K English rows" almost exactly. `version`
is *not* a filter criterion.

`version` distinguishes dataset generations that got merged (`None`/unset: 11,923 English
rows; `400`: 10,441; `52`: 5,346; `51`: 551). All four are kept; `version` is carried
through as provenance metadata only, not used to include or exclude rows.

### 1.2 Categorical fields (English subset)

| Field | Values (count) |
|---|---|
| `queue` (10 values) | Technical Support 8149 · Product Support 5305 · Customer Service 4269 · IT Support 3333 · Billing and Payments 2897 · Returns and Exchanges 1402 · Service Outages and Maintenance 1106 · Sales and Pre-Sales 843 · Human Resources 553 · General Inquiry 404 |
| `type` (4 values, ITIL-style) | Incident 11213 · Request 8163 · Problem 5895 · Change 2990 |
| `priority` (3 values) | medium 11570 · high 10917 · low 5774 |
| `tag_1`..`tag_8` fill rate | tag_1 100% · tag_2 99.9% · tag_3 99.6% · tag_4 90.9% · tag_5 57.7% · tag_6 28.0% · tag_7 13.1% · tag_8 5.7% |

**Note for `output-schema.md`/`state-schema.md`:** the historical corpus's `priority`
never takes the value `critical` — only `low`/`medium`/`high`. `Classification.priority`
keeps its fourth value because agent-created tickets and the `escalation_rule_hit` check
(`graph-design.md` §4.2) can still assign it; it just never appears in retrieved evidence.

### 1.3 Text length (subject / body / answer)

| Field | min | median | mean | max |
|---|---|---|---|---|
| `subject` | 0 | 39 | 38 | 258 |
| `body` | 0 | 365 | 371 | 1793 |
| `answer` | 0 | 372 | 373 | 1315 |

**`body` (and occasionally `subject`) can be empty or near-empty.** A meaningful slice of
rows carry generic placeholder bodies as short as two words (`"seeking assistance"`,
`"assistance required"` — see §2.2). `ingest/load.py` must not assume every row has
substantive body text.

---

## 2. Canonicalisation (D3): clustering method and threshold

### 2.1 Method: embedding similarity, never exact-string matching

D3 already commits to embedding-threshold clustering over MinHash/LSH, "because it reuses
machinery already present" (the embedder built for retrieval). Profiling confirms this is
not just convenient but *necessary*: an exact-normalized-body-string check on this corpus
does not reliably identify genuine duplicates — see §2.2.

**Clustering step (`ingest/cluster.py`) computes embeddings for every English row**, not
just eventual canonicals. Only the chosen canonical of each cluster is later upserted into
the persistent Chroma index (architecture.md §4.1.6) — the rest are used transiently for
clustering and then discarded, keeping the retrieval index at canonical-only size while
still paying the embedding cost once per row either way.

### 2.2 Why exact-string matching is unsafe here

A normalized-body exact-match pass (lowercase, whitespace-collapsed) on the English subset
found 9,201 rows (32.6%) sharing a normalized body with at least one other row, across
4,595 clusters. Inspecting the three largest such clusters found in a 1,500-row sample:

| Normalized body | Cluster size | Min pairwise embedding similarity | Verdict |
|---|---|---|---|
| `"seeking assistance"` | 4 | **0.697** | Not a duplicate — subjects range from "Hospital System Security" to "Data Breach in Medical Records" to "Modify Billing Arrangement" |
| `"assistance required"` | 4 | **0.564** | Not a duplicate — same pattern |
| `"request for api details"` | 4 | **0.997** | Genuine near-duplicate — all four subjects are "Management of API Integration Project(s)" |

Two of three "exact duplicates" are a **data-quality artifact**: the dataset's synthetic
generation occasionally produced a generic, near-empty body (`"seeking assistance"`) that
carries no real content, attached to otherwise unrelated tickets. Treating normalized-body
string equality as a duplicate signal would wrongly canonicalize unrelated tickets down to
one, silently discarding real evidence. Only the third case — where the short-but-shared
string is a real template match, corroborated by near-identical subjects and 0.997
similarity — is a genuine duplicate.

**Consequence for `ingest/cluster.py`:** cluster strictly on embedding similarity computed
from `subject + body` (§3), never as a shortcut on raw text equality. A minimum-content
guard (row excluded from the *canonical pool* if `len(subject) + len(body) < 20` chars, a
generous margin below the shortest genuine ticket bodies observed) prevents a cluster of
degenerate placeholder bodies from being embedded as if it were meaningful content; such
rows still get an `HF-<row>` ID and a SQLite row (so `get_ticket_by_id` still resolves
them), they just aren't eligible to become — or be merged into — a canonical.

### 2.3 Threshold: cosine ≥ 0.92

Random-pair and genuine-near-duplicate similarity were measured directly (1,500-row
sample, 1,124,250 pairs; embed text `subject + " " + body`, `BAAI/bge-small-en-v1.5`,
normalized embeddings, cosine via dot product):

| Percentile of **random-pair** similarity | Value |
|---|---|
| p50 | 0.585 |
| p75 | 0.646 |
| p90 | 0.717 |
| p95 | 0.764 |
| p99 | 0.838 |
| p99.5 | 0.860 |
| **p99.9** | **0.900** |
| mean | 0.596 |

Restricting to pairs that already share `queue`+`type`+`priority` barely moves these
numbers (p50 0.622, p90 0.765, p99 0.880) — same-category tickets are not meaningfully
closer in embedding space than random ones. This is itself a strong, measured data point
for **D4** (hybrid retrieval): dense similarity alone does not separate topics well on
this corpus; metadata filters and the lexical (BM25) arm are doing real work, not
redundant with dense.

A qualitative scan of different-body pairs in the 0.90–0.97 band found genuine
paraphrased duplicates — e.g. two tickets both about "data analytics tools... hindering
optimization of investments" phrased differently (0.9547), two about "unapproved
access... medical data/records" (0.9475).

**Threshold: cosine similarity ≥ 0.92.** This sits comfortably above the measured random-
pair ceiling (p99.9 = 0.900) while still catching the qualitatively-confirmed
near-duplicate band (0.90–0.97). At full-corpus scale (28,261 rows ≈ 399M pairs),
even a ~0.1%-level false-positive rate at a lower threshold would produce hundreds of
thousands of spurious pairs — the margin above p99.9 matters, not just the qualitative
examples.

**Clustering algorithm:** connected components over the similarity graph (edge between
two rows iff cosine ≥ 0.92), computed blockwise (not one full 28,261² dense matrix at
once) to bound peak memory. **Watch for chaining:** single-linkage connected components
can merge A–B–C into one cluster when A–C themselves fall below 0.92 but both exceed it
via B as a bridge. `ingest/cluster.py` must check this doesn't produce oversized or
semantically loose clusters (e.g. flag any cluster whose *minimum* internal pairwise
similarity falls below 0.85 for a manual look) rather than trusting transitivity blindly.

**Canonical selection:** within a cluster, pick the row with the longest `body` (a proxy
for the most complete/informative version of the template) as canonical; all cluster
members keep `canonical_of` pointing to it; `cluster_size` is the cluster's row count.

**This is a starting threshold, not a final one.** Re-validate against the full-corpus
run: report the resulting cluster-size distribution in `ingest`'s CP1 done-criteria output
(checkpoints.md), and revisit 0.92 if the distribution looks wrong (e.g. a long tail of
absurdly large clusters, or almost no clusters at all).

---

## 3. Embed-text composition

**Dense embed text (what gets vectorized and stored in Chroma): `subject + "\n\n" + body`
only.** This clarifies architecture.md §4.1.6's "subject + body + selected metadata" —
having profiled the corpus, concatenating tags/queue/type into the embedded text is
deliberately **not** done:

- Tags are short, discriminating keywords (`network`, `billing`, `security`) — exactly
  what the **lexical** (BM25) arm already captures precisely (D4). Folding them into the
  dense text is redundant with a channel that already handles them well, and dilutes the
  narrative semantic signal that dense embedding is actually good at.
- `queue`/`type`/`priority`/`answer_class`/`cluster_size`/tags all live as **Chroma
  metadata only**, used for `where` filtering (`refine_retrieval`'s queue-filtered variant,
  graph-design.md §8) and reranking signals — never concatenated into the embedded text.

**Clustering embed text (§2) uses the same `subject + "\n\n" + body` composition** — not
`answer`, deliberately. Clustering identifies duplicate *problems*, independent of how
each was resolved; including `answer` text would bias clusters toward shared resolution
phrasing rather than shared underlying issue, and would conflate two different root
causes that happened to get the same boilerplate answer opener (see §4 — answers are
themselves templated).

**Lexical (FTS5) index text:** `subject`, `body`, `answer`, and the joined tag string —
covering the answer text is fine (even necessary) for the lexical arm, since BM25 is a
precise keyword matcher, not a semantic one; the "don't bias clustering toward answer
phrasing" concern doesn't apply here (architecture.md §2.1b already specifies this).

---

## 4. `answer_class` heuristics (D5)

### 4.1 Rough signal, to validate the premise before writing real heuristics

A keyword scan over the English subset's `answer` field (not the final classifier —
just checking the premise is real before investing in heuristics):

| Signal | Match rate |
|---|---|
| Clarification-like (`"could you please provide"`, `"could you specify"`, `"please provide more"`, `"could you clarify"`, `"can you provide"`, `"please send us"`, `"please share"`, `"could you send"`) | 8,348 rows — **29.5%** |
| Escalation-like (`"escalate"`, `"forwarded your ticket"`, `"higher support tier"`, `"specialized team"`, `"escalated to"`) | 228 rows — **0.8%** |

This confirms D5's premise: a large share (≈30%) of historical answers are requests for
more information, not resolutions. Escalation as a historical `answer_class` is rare
(<1%) — the `escalation`-labelled bucket in `EvidenceEntry.answer_class` (output-schema.md
§3.1) will be sparse for dataset cases, which is expected and fine; it's mostly populated
by agent-resolved cases going forward.

### 4.2 Starting heuristic rules for `ingest/classify.py`

Regex/keyword first pass (case-insensitive, matched against `answer`), applied in this
order — first match wins:

1. **`clarification_request`** if the answer contains a question mark **and** any of:
   `could you (please )?(provide|specify|clarify|send|share|confirm)`,
   `can you (please )?(provide|send|share|confirm)`,
   `please (provide|send us|share|specify|confirm)`,
   `(what|which|when) (is|was|version|model)`.
2. **`escalation`** if the answer contains any of:
   `escalat(e|ed|ing)`, `forwarded (your|the) (ticket|case|request)`,
   `(specialist|specialized) team`, `higher (support )?tier`, `senior (support|agent)`.
3. **`resolution`** otherwise — the default, since most answers (≈70% per §4.1) are
   substantive replies rather than requests or handoffs.

**Validation (per D5's stated method, unchanged):** run this heuristic over a 200-row
random sample, hand-label the sample independently, measure precision against the hand
labels, and report the measured figure in the README as methodology. Send only the
residue the heuristic is unsure about (i.e. rows matching none of the three rules, if any
— the "otherwise" default currently means there is no true residue category; if hand
labelling reveals the default is imprecise, add a genuine "ambiguous" bucket and route
*that* to the fast-tier LLM, keeping the LLM call count near-zero as D5 requires) to the
fast-tier LLM.

---

## 5. Open — CP2 (retrieval-time parameters)

Not yet decided; to be filled in when `rag/` is built:

- Dense over-fetch `k` per query.
- Reciprocal Rank Fusion constant.
- MMR λ (diversity vs. relevance trade-off).
- `SIM_CEILING` calibration for `output-schema.md` §4.4's confidence formula — the random-
  pair ceiling measured here (p99.9 = 0.900) is a useful anchor: `SIM_CEILING = 0.85`
  (output-schema.md's current default) sits *below* that ceiling, meaning even some
  random/unrelated pairs could score a nonzero `relevance` component today. Revisit
  `SIM_CEILING` (and possibly `τ_rel = 0.55`, which is well below the random-pair p50 of
  0.585) alongside the CP2 retrieval calibration — both may need to move up once the
  fused (RRF) ranking, not raw dense cosine, is what's actually threshold-tested.
- FTS5 query construction (tokenization, phrase vs. OR matching for entity terms).
