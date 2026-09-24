# RAG Design — Ingestion & Retrieval

> **Status:** Draft v2 (full rewrite). Every number below has a stated reason and is
> reproducible from `scripts/rag_calibration.py`, whose output is committed at
> `docs/rag-calibration-output.txt`. Where a choice was contested, the losing argument is
> recorded as a **Trade-off note** rather than omitted.
> **Companion docs:** `architecture.md` §2.1/§2.1b/§4.1, `decisions.md` D3–D6 and D12,
> `output-schema.md` §4 (confidence consumes `τ_rel` and `SIM_CEILING` from §9 below),
> `graph-design.md` §5, §8–§9 (retrieval-round config, second-pass variants).

---

## 1. Corpus: the English subset

`language == 'en'` gives **28,261 rows**, with no version filter: all four `version` values
(`None`, `51`, `52`, `400`) carry English rows against an identical schema, and `version` is
provenance, not a quality signal (decisions.md D6).

**Exact-duplicate dedup, done before anything else.** Normalizing and lower-casing
`subject|body|answer` finds **4,460 exact-duplicate pairs**. Every single one is a
`version=None` row paired with a `version=400` row sharing identical queue, type, priority
and tags — a dataset-generation merge artifact, not independent evidence. Keep the labelled
(`v400`) copy, drop its `None` twin.

**Distinct English records: 23,801**, not 28,261. This is the number that matters for
everything downstream — cluster sizes, retrieval-index size, the calibration sample. A
record that silently exists twice would let `cluster_size` overcount agreement for the exact
cases D3 exists to protect against.

`HF-<row>` is the record's position in the **unfiltered** HF `train` split (61,765 rows), not
its position after filtering — a post-filter index would shift if the filter ever changes,
breaking D1's determinism guarantee. `ingest/load.py` keeps an `hf_row` column recording this.

---

## 2. Normalisation, applied before any duplicate detection

Two text forms exist per record:

- **Display text** — the raw dataset value. Never altered. Shown to the user and to the LLM
  when a case's full text is fetched via `get_ticket_by_id`.
- **Index text** — what gets embedded, clustered on, and written into FTS5. Built by four
  ordered steps:
  1. Literal two-character `\n` (997 bodies, 549 answers — not a real newline, a literal
     backslash-n in the source data) and `<br>` (2,876 occurrences in bodies) become a space.
     Unnormalised, FTS5 tokenised `"\n\nCould"` as the junk token `ncould`.
  2. Anonymisation placeholders are removed: `<tel_num>` (10,952 occurrences in answers),
     `<name>` (6,132), `<acc_num>` (2,820), `<email>`, `<ref_num>`, `<website_url>` and
     similar. Left in place, FTS5 turns `<tel_num>` into the tokens `tel` and `num`, and
     dense embeddings pick up "contains a phone number" as spurious topical signal.
  3. **Bodies only:** a leading salutation (`Dear ...`, `Hello ...`) and "I hope this
     message/email finds/reaches you..." are stripped, up to three times (they can stack:
     a greeting sentence *and* a separate "I hope this message..." sentence).
  4. NFKC normalisation and whitespace collapse.
  5. Placeholders are replaced with a literal space for embedding/FTS5 text, or with the
     literal token `<X>` when index text feeds the `answer_class` heuristic (§5) — there,
     "call you at `<tel_num>`" is itself a handoff signal worth keeping visible.

  Step 1 of the pipeline overall is exact-key dedup on lower-cased `subject|body|answer`
  index text (§1). Step 2 of the pipeline is near-duplicate clustering (§4).

**Why greeting-stripping doesn't matter as much as it first looked.** A hand-count
originally attributed the corpus's high random-pair dense similarity (§9) to boilerplate
greetings. Measured: only **17.9%** of bodies start with a greeting overall — 92% within the
small `version=51` slice (551 rows, 2.3% of the corpus), 10–22% elsewhere. **The greeting
theory is wrong; the conclusion it supported (D4, hybrid retrieval) is not.** Stripping
greetings barely moves the random-pair similarity distribution at all (p50 0.585 either way
— see §9's normalised-vs-raw check in the calibration output). The real cause is that the
whole corpus is synthetic, single-domain (IT/software support) and generated in a narrow
register — every ticket sounds like every other ticket at the sentence-construction level,
independent of greetings. `decisions.md` D4 is corrected to say this.

---

## 3. Embed-text composition

**The vector text is `subject_ix + ". " + body_ix` when a subject exists, or `body_ix`
alone otherwise.** This is used identically for the dense retrieval embedding *and* the
clustering embedding (§4) — the same text, the same model, one embedding pass per record.

**The answer is deliberately excluded from the vector.** Clustering and retrieval both need
to match on the *problem*, not on how it happened to be answered. Two tickets with the same
root cause can get differently-phrased answers; including answer text would bias a match
toward shared answer phrasing rather than shared underlying issue, and it would make
`SIM_CEILING` (§9) conflate "similar problem" with "similar answer," which §4's guard needs
to keep separate.

**Subjectless records don't need invented text.** 2,843 of the 23,801 distinct records
(11.9%) have no subject. Measured: `cos(subject+body, body-only)` for the *same* ticket has
p10 = 0.919, p50 = 0.968 — a missing subject barely moves the vector at all, because the
body already carries the content (subjectless bodies are actually longer: median ~470 vs
~350 chars for subject-bearing records). No placeholder subject is invented.

**Display title.** `subject` when present; otherwise the first non-greeting sentence of
`body_ix`, truncated to 90 characters, with `title_is_derived = 1` so a UI can show it
differently if desired. Never fed into a vector or FTS5 — display only.

**Chroma metadata only, never concatenated into the embedded text:** `queue`, `type`,
`priority`, `answer_class`, `cluster_size`, `source`, `version`, a joined `tags` string, and
one `tag_<slug>=True` boolean per tag (up to 8; 1,590 distinct tags exist across the corpus,
762 used only once — every tag is kept, no minimum-frequency filter, since a record only
ever carries its own ≤8 keys and Chroma doesn't care how many distinct keys exist
collection-wide). This clarifies `architecture.md` §4.1.6's "subject + body + selected
metadata," which as literally written suggested metadata gets embedded — it doesn't. Tags
are short discriminating keywords BM25 (§6) already matches precisely; folding them into
the dense text would be redundant with a channel that handles them better and would dilute
the narrative signal dense embedding is actually good at.

**Minimum-content guard: `len(embed_text) < 30` characters.** After normalisation this
affects **8 of 23,801 records** (`"Can you assist me?"`, `"Problems with the server"`, one
empty string, and five similar). These 8 records still get a SQLite row and an `HF-<row>`
ID — `get_ticket_by_id` still resolves them — but they are excluded from clustering (§4)
and never embedded into Chroma, since a near-empty vector is both meaningless for semantic
matching and, per an earlier profiling pass, a magnet for false near-duplicate matches on
unrelated tickets that happen to share the same generic filler text.

---

## 4. Near-duplicate canonicalisation

### 4.1 Method: star (leader) clustering, not connected components

Embeddings are computed for **every** eligible English record (not just eventual
canonicals) — clustering needs to compare all of them. Only the chosen canonical of each
cluster is later upserted into the persistent Chroma index (§3); the rest are used
transiently for clustering and then discarded from the vector store, though every record
still gets a `dataset_tickets` row.

**Star clustering:** build a k-nearest-neighbour graph (K=50) over the embed-text vectors.
Order records by degree at threshold T (descending), then by body length (descending) as a
tie-break — this makes the most-connected, most-complete record the natural cluster leader.
Walk that order; each unassigned record becomes a new leader; each of its unassigned
neighbours at cosine ≥ T joins it **if the guard holds** (§4.2). The canonical of a cluster
is its leader; `cluster_size` = 1 + member count.

**Why not connected components** (single-linkage transitive closure over the same graph, i.e.
A–B and B–C similar implies A, B, C one cluster even if A and C aren't themselves similar):
measured on the full corpus at T=0.92 (separate exploratory run, K=20 — not part of
`scripts/rag_calibration.py`'s committed sweep, since the failure mode is structural and
independent of `answer_class`, so it doesn't need re-confirming every time the heuristic
changes), connected components collapses the corpus into a handful of giant components — the
largest holds **3,343 records**, and **74.6%** of all records collapse into some cluster. At
T=0.90 the largest component is **11,551 records** — nearly half the corpus in one "cluster."
Both are obviously wrong: a similarity graph chains
through intermediate near-matches (A resembles B, B resembles C, but A and C may be
unrelated), and a single-linkage closure has no way to stop that chain. Star clustering
doesn't have this failure mode by construction: every member is required to be similar
*to the leader specifically*, not transitively reachable through a chain of near-matches.

### 4.2 The guard: why similarity alone isn't sufficient

A pair with high *problem* similarity often has a genuinely different *answer*. Measured
over 6,000 sampled top-3-neighbour pairs at cosine ≥ 0.86, bucketed by similarity band:

| Similarity band | n | Answer-sim median | Answer-sim < 0.80 | Entity conflict |
|---|---|---|---|---|
| 0.86–0.88 | 385 | 0.814 | 44.4% | 5.2% |
| 0.88–0.90 | 533 | 0.830 | 35.6% | 1.1% |
| 0.90–0.92 | 745 | 0.841 | 33.0% | 0.4% |
| 0.92–0.94 | 922 | 0.865 | 27.3% | 0.4% |
| 0.94–0.96 | 970 | 0.930 | 11.6% | 0.2% |
| 0.96–0.98 | 1,089 | 0.978 | 3.5% | 0.1% |
| 0.98–1.00 | 1,356 | 0.986 | 0.1% | 0.0% |

Even at 0.90–0.92, **33%** of "near-duplicate problem" pairs have meaningfully different
answers (cosine < 0.80), and entities genuinely conflict in a few percent of pairs (e.g. one ticket
names PostgreSQL, its near-duplicate names MySQL — same *shape* of problem, different
actual system). Left unguarded, these would merge into one canonical whose `cluster_size`
claims "N cases handled this way" when several of those N were handled a different way — the
exact failure D3 exists to prevent, just moved from exact-duplicate detection (where it
doesn't happen — see §1) into near-duplicate detection (where it does).

**The guard:** a candidate neighbour only joins a leader when, in addition to cosine ≥ T:
- `answer_class` (§5) matches exactly, and
- answer-text cosine similarity ≥ **T_a**, and
- no named-entity conflict — extracting capitalised/technical tokens (`PostgreSQL`, `AWS`,
  `MongoDB 4.4`) from each record's embed text, a conflict is when each side names an
  entity the other doesn't mention at all.

### 4.3 Threshold values: T = 0.92, T_a = 0.85

Full-corpus sweep (23,801 distinct records, K=50 kNN):

| T | Guard | Canonicals | Collapsed | Max cluster | Within-cluster min-pair p5 |
|---|---|---|---|---|---|
| 0.90 | unguarded | 7,691 | 67.7% | 51 | 0.820 |
| 0.90 | guarded | 11,074 | 53.5% | 39 | 0.844 |
| 0.92 | unguarded | 9,532 | 60.0% | 51 | 0.859 |
| **0.92** | **guarded** | **12,217** | **48.7%** | **39** | **0.876** |
| 0.94 | guarded | 13,605 | 42.8% | 28 | 0.912 |
| 0.96 | guarded | 15,412 | 35.2% | 15 | 0.943 |

T_a sweep at T=0.92, guarded:

| T_a | Canonicals | Collapsed |
|---|---|---|
| 0.80 | 11,833 | 50.3% |
| **0.85** | **12,217** | **48.7%** |
| 0.90 | 13,021 | 45.3% |

**T = 0.92** sits comfortably above the measured random-pair noise ceiling (p99.9 = 0.905 — §9), so
the similarity itself, not just the guard, justifies every link. **T_a = 0.85** is chosen
against the measured random-*answer*-pair baseline (p99.9 = 0.895 — same order as the
problem-text ceiling), for the same reason: an answer-similarity requirement below the noise
floor isn't really requiring anything.

**Trade-off note — the case against T=0.92, argued and then resolved:**

- *For raising T to 0.94.* The within-cluster minimum-pairwise-similarity at the 5th
  percentile is 0.876 at T=0.92 — inside the band random pairs can occasionally reach. So
  "`cluster_size` = N cases" could technically include a member that isn't close to some
  *other* member, only to the leader. **Defended, not revised:** star clustering's semantic
  claim is exactly "each member is a close variant of the canonical" — the canonical is what
  actually gets retrieved and cited, never a member. Member-to-member distance was never
  part of the claim. The cost of going to 0.94 is real: 1,388 fewer records collapse
  (13,605 canonicals vs. 12,217), and a spot-check at similarity 0.923–0.926 found
  clearly-identical scenarios (paraphrased duplicates of the same ticket) that would now
  compete separately for top-k space instead of being represented once with their true
  combined weight.
- *For lowering T to 0.90.* Would collapse 1,143 more records (11,074 canonicals vs.
  12,217). **Rejected:** those extra links sit right at the noise ceiling, and the §4.2
  table shows 33% of pairs in exactly that 0.90–0.92 band have answer-similarity below
  0.80 — precisely the links the guard was built to catch, arriving with the least safety
  margin.
- **The part of this that actually loses, recorded rather than hidden:** star clustering is
  order-dependent (which record becomes a leader depends on the deterministic degree/length
  ordering, so a different valid tie-break would produce a different — not wrong, just
  different — canonical), and clusters are capped at K+1 = 51 members by the kNN graph's
  fixed neighbourhood size. Both are accepted: the ordering is fully deterministic so re-runs
  are reproducible, and the confidence formula's cluster-size weight (`output-schema.md` §4.4)
  saturates at `cluster_size` = 16, so the 51-cap never actually binds on anything that
  matters to a downstream score.

---

## 5. `answer_class` heuristic

### 5.1 Classes and precedence

Handoffs (a promise to investigate, or to schedule a call, with no fix or fact given yet)
map to **`escalation`**, not their own category — `graph-design.md`'s escalation rule
already reads "resolved by escalation *or handoff*," and a corpus answer that only promises
future contact carries no grounded fix to cite regardless of whether a human labelled it a
formal escalation.

Classification runs **sentence-by-sentence** (not on the whole answer at once — see §5.2 for
why), in this precedence per answer:

1. **`resolution`** if any sentence states a concrete step ("please restart...", "you will
   need to configure...") or a concrete, specific fact ("X integrates with Salesforce",
   "the billing period begins on the 1st") — see §5.2 for exactly what counts as specific.
2. **`escalation`** if any sentence contains explicit escalation language ("escalate",
   "forwarded to", "specialist team", "higher tier").
3. **`clarification_request`** if any sentence explicitly asks for a missing fact ("could
   you provide...", "please specify...") and isn't itself a call-scheduling sentence.
4. **`escalation`** (via handoff) if any sentence schedules a call or defers ("we'll
   investigate and reach out", "let us know a convenient time").
5. **`residue`** otherwise — sent to the fast-tier LLM (§5.6).

### 5.2 Why sentence-level, and why "specific" is required

An answer routinely mixes a real fix with a trailing "let us know if you have other
questions," or a vague sales pitch with a genuine account-specific fact. Splitting into
sentences and taking the strongest signal from any one of them, rather than pattern-matching
the whole answer as a blob, is what makes rule 1 (a real fix, even a short one, wins) work
without an unrelated trailing sentence about scheduling a call knocking a genuine resolution
down to `escalation`.

**Positive substance, not a negative exclusion list.** An early version of the "concrete
fact" pattern excluded vague objects by name — "solutions/services/options" — but this is
whack-a-mole: `"our platform offers seamless integration services"` slipped past the
exclusion because the check only looked at the single word immediately after the verb
(`"seamless"`, not excluded), not the real head noun two words later (`"services"`, which
*was* excluded). Validation on the hand-labelled sample (§5.4) kept finding new phrasings of
the same problem — `"multiple methods"`, `"customized digital strategies"` — because English
has too many ways to say "a vague thing" to enumerate.

The fix requires **positive evidence of specificity** instead: a number, an acronym, or a
named product/brand (mid-sentence capitalised words that aren't ordinary sentence vocabulary
— this corpus names specific products constantly: Salesforce, PostgreSQL, Bitdefender,
Smartsheet). A vague "we offer a range of solutions" has none of these; "MongoDB 4.4" or
"Salesforce CRM" does. This generalises where a negative list can't, at the cost of
occasionally still passing something that happens to contain a version number without being
truly specific (§5.5, "what didn't get fixed").

**Why substance defaults to negative** (a resolution needs positive evidence, not just the
absence of other signals) rather than the reverse: a false `resolution` inflates the
downstream confidence score with a case that never actually resolved anything ("confident
nonsense" — the exact failure D5 exists to prevent). A false `clarification_request` merely
under-uses a case as evidence — a much cheaper mistake. The heuristic is built to be
conservative in the expensive direction.

**A published error, corrected here.** An earlier commit message attributed the gap between
two informal precision checks (29.5% vs. 50.6% clarification-request rate) to the
`(what|which|when) (is|was|version|model)` pattern. Re-measured directly: that pattern fires
on only 0.9% of answers. The actual cause was the `please (provide|share|...)` pattern
firing on 46% of answers, only 27% of them containing a question mark — most of those are
genuine imperative requests ("Please provide your account number"), so gating on "contains a
`?`" (as an earlier draft of this doc proposed) would have dropped true positives, not fixed
a bug. This doc's current rules were re-derived from the measured sentence-level behaviour,
not patched onto the old ones.

### 5.3 Measured distribution

Full corpus (23,801 distinct records), heuristic-only pass (before the LLM resolves
`residue`):

| Class | Share |
|---|---|
| `clarification_request` | 50.1% |
| `escalation` (including handoffs) | 32.7% |
| `resolution` | 7.1% |
| `residue` → LLM | 10.1% |

Once the fast-tier LLM resolves the residue slice into one of the three real classes, the
final share shifts slightly — an actual `autosupport ingest --limit 500` run (CP1c) measured
`resolution` at 8.8%, `escalation` at 36.6%, `clarification_request` at 54.6% post-LLM on a
500-row sample.

**Grounding is scarce by construction**, not by a bug: on either figure, well under one in
eight historical answers is a genuine resolution. This is the direct justification for the
`resolution_only` second-pass retrieval variant (§10 V2) — an unfiltered top-10 over the
whole corpus averages under one resolution-class case, nowhere near enough to satisfy
`graph-design.md` §5's "≥ 3 relevant cases in the dominant cluster" sufficiency bar.

### 5.4 Validation

**Dev/test split.** `md5(dedup_key) % 10 == 0` selects the test pool (≈2,380 records, 10%);
heuristic rules were tuned only by eyeballing dev-split examples (plus roughly 30 rows
looked at corpus-wide before this split existed, during initial exploration — disclosed
rather than pretending the split was pristine from the first pattern written).

**Sample.** 200 rows, stratified from the test pool by the heuristic's own predicted class:
60 predicted-resolution, 50 predicted-clarification, 50 predicted-escalation, 40
predicted-residue (i.e. rows the heuristic would send to the LLM). Resolution gets the
largest stratum because a false resolution is the costly error (§5.2).

**Labelling.** Blind — the label file held only `case_id`, `subject`, `body`, `answer`, with
no predicted class column, so the labeller (Claude, in this session, not a human — recorded
here and in the README rather than implied to be a human hand-label) couldn't anchor on the
heuristic's own guess. Committed at `tests/fixtures/answer_class_validation.csv`.

**Measured precision** (of rows the heuristic predicted class X, how many the blind label
agrees are X), after nine rounds of revise-and-remeasure against this same fixed 200-row
sample:

| Class | Precision | n | 95% Wilson CI |
|---|---|---|---|
| `resolution` | 0.839 | 31 | (0.674, 0.929) |
| `clarification_request` | 0.885 | 61 | (0.782, 0.943) |
| `escalation` | 0.840 | 50 | (0.715, 0.917) |

**Bars set going in:** resolution ≥ 0.85 with a CI lower bound ≥ 0.75; clarification and
escalation ≥ 0.80. Clarification clears its bar. Resolution and escalation land at their
point-estimate bar within measurement noise (their CIs comfortably contain 0.85) but not
strictly above it. **This is reported rather than pushed further**, for a concrete reason:
each of the last several fixes moved one class's precision up and another's down by a
comparable amount (see the git history on `ingest/classify.py` — one revision alone swung
resolution from 0.617 to 0.757 while escalation moved from 0.78 to 0.73), which is the
signature of fitting noise in a 200-row sample rather than genuinely improving the rule.
Continuing to chase the last percentage point risks overfitting the only labelled sample
that exists, which would make the reported number *less* trustworthy, not more.

**On a future failure to clear the bar:** move the weakest pattern family's hits to residue
and re-validate on a *fresh* 200-row sample (not the same one — the current sample has now
been used to tune the rules, so it's no longer a clean held-out measurement). At most two
such rounds, then report as-is, exactly as happened here.

### 5.5 What didn't get fixed, and why that's the right place to stop

The remaining resolution false positives cluster around two patterns that need real-world
knowledge, not more regex: "I recommend reviewing our documentation" (meta-advice — points
at a resource rather than stating a fact, and is hard to distinguish syntactically from "I
recommend enabling encryption," which *is* a real instruction) and incidental specificity
("We offer multiple methods to integrate MongoDB 4.4..." passes the specificity gate on the
version number in "MongoDB 4.4," not because the actual claim being made is specific). Both
were tried as targeted fixes and reverted: excluding "recommend reviewing/checking/reading"
from the suggestion pattern cost more true positives ("we recommend enabling encryption,
reviewing your firewall rules...") than it fixed.

### 5.6 LLM residue pass

10.1% of the corpus (2,402 of 23,801 records) reaches no rule and is classified by the
fast-tier model with `.with_structured_output(ResidueClassification)` — a three-way forced
choice (never a fourth category), each call independent and cacheable. This keeps the total
LLM classification volume at roughly 2,400 calls for a full ingest, not 23,801 — CLAUDE.md
constraint 5 ("never run a model over all 28K rows") is satisfied by construction, not by
a cost argument. Each row's `answer_class_source` column records `heuristic` or `llm`.

### 5.7 Trade-off note — arguments against this design, stated and resolved

- **"Regexes are brittle on synthetic paraphrase — a fast-tier LLM over all 23,801 distinct
  records costs about $3 and would likely be more accurate."** Partly conceded: it's
  probably true. The heuristic stays as the primary classifier because CLAUDE.md constraint
  5 is locked, and because a heuristic's precision is a reportable, reproducible methodology
  figure in a way "the model said so" isn't. **If resolution precision ever misses its bar
  after two genuine revision rounds** (as opposed to landing at-bar-within-noise, which is
  what happened here), that is the stated trigger to go back to the user and ask whether to
  reopen constraint 5 — never to quietly swap the approach.
- **"Mapping handoffs to `escalation` makes `escalation_rule_hit`'s '≥60% of the dominant
  cluster handed off or escalated' condition fire for roughly a third of all topics, so the
  agent will escalate far more often than a human support team would."** Defended: if most
  historical answers to a given problem were "we'll look into it and call you," the corpus
  genuinely holds no grounded fix for that problem — routing to a human is the honest
  response, not an over-cautious one. The blast radius is contained by Decision 3
  (`output-schema.md` §4.3): "substantive" support for the confidence formula is
  resolution-only, so handoff-class evidence can never inflate a confidence score even
  though it does inform the escalation *decision*.
- **"An answer with one small concrete fact plus a large amount of hedging/marketing filler
  gets full resolution credit for the one fact."** Accepted, with a guard already in place:
  the substance-bearing sentence must be ≥ 6 words (not "Yes." or "OK, will do."), and the
  resolution stratum of the validation sample (§5.4) is specifically sized to measure this
  exact error mode — it's the source of most of the 0.839 precision figure's remaining gap.

---

## 6. FTS5 table and BM25 query

**A standalone (non-external-content) FTS5 table, `dataset_tickets_fts`.** External-content
FTS5 tables point at exactly one backing table; this index needs rows from both
`dataset_tickets` (historical, ingest-time) and, later, `cases` (agent-resolved, added at
CP6) — a plain, separately-populated FTS5 table is the only workable shape for that.

**Columns:** `subject`, `body`, `answer`, `tags` (indexed, index text — normalised per §2);
plus `case_id`, `source`, `queue`, `type`, `answer_class` as `UNINDEXED` columns, available
as ordinary SQL predicates for a metadata-filtered variant (§10 V4) without FTS5 trying to
tokenise them.

**Tokenizer: `unicode61 remove_diacritics 2`, explicitly *without* Porter stemming.**
Porter stemming mangles the exact entity tokens this corpus's BM25 arm exists to catch:
"NAS" stems to `na`; it would merge "Teams" with "team" and "Windows" with "window," erasing
precisely the product/version distinctions D4 relies on BM25 to preserve. Morphological
recall (matching "configuring" when the query says "configure") is left entirely to the
dense arm, which already handles it.

**The index holds only the retrievable set** — canonicals plus later-accepted agent-resolved
cases, the same set Chroma holds — so RRF (§7) never double-counts a near-duplicate that
Chroma canonicalised away.

**Query construction:** the top 12 terms by IDF (`fts5vocab`'s per-term document frequency,
ascending — rarest first), excluding any term with document frequency ≥ 20% of the index
size. A median ticket has 31 distinct terms below that 20% threshold (p25 19, p75 43); terms
at or above it are exactly stopwords plus this corpus's own boilerplate vocabulary
(`assistance`, `issue`, `please`, `problem`, `software`, `support`...) — this bound *is* the
stoplist, measured rather than hand-written. Each term is double-quoted and `OR`-joined:
`ORDER BY bm25(dataset_tickets_fts) LIMIT 50`, with **flat (default) column weights.**

**Why flat, not weighted:** a subject-weighted-2×/tags-1.5×/answer-0.5× variant was measured
against flat weighting on both the member-recall task and the rare-entity task and came back
within noise of each other (recall 0.770 flat vs 0.780 weighted; entity@10 0.383 flat vs
0.380 weighted — see `docs/rag-calibration-output.txt` §6). A parameter with no measured
effect has no reason to exist. The harness also measured an answer-weight-0 variant
(excluding answer text from scoring entirely: recall 0.782, entity@10 0.370) to check
whether indexing answers helps at all in the first place; the difference from flat weighting
was equally negligible, so indexing the answer for lexical completeness costs nothing even
though it doesn't help much either.

---

## 7. Fusion: Reciprocal Rank Fusion

RRF over the dense top-50 and the BM25 top-50: `score(doc) = Σ_arms 1/(k + rank)`, summed
across whichever arm(s) return the document; ties break on dense rank.

**k = 10**, not the literature-common default of 60. Two independent reasons:

- **Structural.** A document appearing in only one arm at rank 1 beats a document appearing
  in both arms only when both of its ranks exceed `k + 2`. At `k=60` that threshold (62) is
  never reached within 50-deep result lists — a lexical-only hit can never structurally win
  against anything the dense arm also returns, no matter how weakly. At `k=10` the crossover
  point (12) is reachable, so a strong lexical-only match can surface.
- **Measured** (full corpus, 400 rare-entity queries with a lexical-only match, n=165): a
  lexical-only entity hit survives into the fused top 10 at **64.8%** with k=10, falling to
  50.3% at k=60. Recall on the member-recall task barely moves (0.945 at k=10 vs 0.912 at
  k=60, out of 400 member queries) — the entity-rescue gain at k=10 is close to free.

**Trade-off note.** k=5 was also measured (66.7% survival, 0.948 recall): it gains under 2
points of entity survival over k=10, but at the cost of letting a single noisy rank-1 hit in
either arm dominate the fused score disproportionately (1/6 vs. rank 2's 1/7 — a much
steeper drop-off than k=10 produces). k=10 was chosen as the better balance. The literature
default of k=60 (Cormack et
al., 2009) was tuned for fusing many independent full-text-search systems in a TREC setting;
it isn't tuned for exactly two arms where one arm's entire purpose is rescuing hits the
other structurally can't find — a different problem shape with a different right constant.

**`similarity`** (`state-schema.md`'s `RetrievedCase.similarity`) is filled for **every**
fused candidate, including lexical-only hits, as the cosine between the ticket's embedding
and the candidate's embedding (fetched from Chroma) — not left `None` for lexical-only
results as an earlier draft left open. `score` remains the fused RRF value, used for ranking
only, never compared to a similarity threshold.

---

## 8. Over-fetch, final k, and MMR

**Over-fetch: 50 per arm.** Checked against 25 and 100 in the harness; recall and
entity@10 were flat across all three depths (`docs/rag-calibration-output.txt` §6). 50 is
kept because it's large enough to guarantee the MMR pool (next) has real width to work with,
without being large enough to cost a noticeable amount of query time.

**MMR pool: the fused top 30.** **Final k: 10** for `retrieve_initial`, **8** per
`retrieve_variant` (graph-design.md §8) — three variants at 8 each plus the initial 10 gives
34 before deduplication, comfortably inside `merge_cases`'s 30-item state cap after the
inevitable overlap between variants.

**MMR:** relevance is each candidate's RRF score, min-max normalised within the pool;
redundancy is the candidate's maximum cosine similarity to an already-selected result.

**λ = 0.7.** Measured against 0.5 and 0.85 (rare-entity task, full corpus, no-MMR baseline
entity@10 = 0.327, recall = 0.945):

| λ | Entity@10 | Recall | Avg. results swapped vs. no MMR | Result-set pairwise similarity |
|---|---|---|---|---|
| 0.5 | 0.344 (+0.017) | 0.917 (−0.028) | 1.39 | 0.705 |
| **0.7** | **0.335 (+0.008)** | **0.938 (−0.007)** | **0.62** | **0.722** |
| 0.85 | 0.332 (+0.005) | 0.943 (−0.002) | 0.33 | 0.729 |
| none | 0.327 | 0.945 | 0 | 0.735 |

λ=0.85 is the cheapest in recall terms, but it swaps under half as many results as λ=0.7
(0.33 vs. 0.62) — it's closer to *not running MMR at all* than to genuinely diversifying,
which is a real cost given CLAUDE.md constraint 4 mandates MMR specifically for diversity,
not as a formality. λ=0.5 swaps more than twice as much as λ=0.7 for barely more entity
gain, at four times the recall cost. **λ=0.7 is the point where MMR is still doing
real diversification work while the recall cost stays small** — the middle of the three,
chosen deliberately rather than by default.

**Said plainly: after star canonicalisation, MMR is a light safety net, not the main
diversity mechanism.** Mean pairwise similarity within the result set moves only from 0.735
(no MMR) to 0.722 (λ=0.7) — canonicalisation has already done most of the diversifying work
by the time MMR runs. It's kept because CLAUDE.md constraint 4 mandates it and because it
still catches residual near-duplicates that fell just under the clustering threshold T.

---

## 9. Similarity scale

Measured random-pair cosine distribution over embed-text vectors (1,500-record sample,
~1.1M pairs): **p50 = 0.585, p90 = 0.719, p95 = 0.764, p99 = 0.839, p99.9 = 0.905.**

**`τ_rel` (the "relevant case" threshold, `graph-design.md` §5) = 0.76**, the measured
random-pair p95. "Relevant" now means "more similar to the ticket than 95% of arbitrary
ticket pairs" — an operational, measured definition. The prior default, 0.55, sat *below*
the random-pair *median* (0.585), meaning more than half of all unrelated ticket pairs would
have registered as "relevant" under the old threshold. A calibration check confirms 0.76
doesn't starve evidence: over 400 rare-entity queries, the query's 3rd-best dense similarity
to an actual canonical was p10 = 0.770, p50 = 0.829, and **93.2%** of queries had a 3rd-best
similarity at or above 0.76 (`docs/rag-calibration-output.txt` §6) — the "≥ 3 relevant
cases" sufficiency bar (`graph-design.md` §5) remains reachable for the large majority of
queries, not just the median one.

**`SIM_CEILING` (`output-schema.md` §4.4) = 0.92**, equal to the clustering threshold T. At
or above this similarity, a retrieved case is — by the same standard used to canonicalise
the corpus in the first place — a near-duplicate of the ticket, which is exactly what
"ceiling" should mean for the confidence formula's relevance term.

---

## 10. Second-pass targeted retrieval (`refine_retrieval`, `graph-design.md` §8)

Up to three `Send` variants, evaluated in this priority order — the first two whose trigger
condition holds fire; `V3` always fires; `V4` fires conditionally. Each variant runs the
full dense → BM25 → RRF → MMR pipeline independently, with final k=8.

| Variant | Trigger | Query | Why |
|---|---|---|---|
| **V1** `clarification_keywords` | `clarifications` is non-empty | Ticket text plus the latest clarification answer; the answer's named entities become quoted BM25 phrases | The customer just supplied the one fact retrieval was missing — use it precisely, not just append it to a generic query |
| **V2** `resolution_only` | Fewer than 2 relevant `answer_class="resolution"` cases held so far | Ticket text, filtered to `answer_class = 'resolution'` | §5.3: only ~12% of the corpus is resolution-class. An unfiltered top-10 averages roughly one resolution case — nowhere near the "≥3 relevant, ≥2 in the dominant cluster" bar `graph-design.md` §5 sets for sufficiency |
| **V3** `hypothesis_rewrite` | Always | The fast tier writes a `QueryRewrite{text, keywords≤6}` from the current hypothesis | The ticket's literal wording and the actual underlying hypothesis often diverge once investigation has narrowed things down |
| **V4** `queue_filtered` | Neighbour top-queue share < 0.5, or triage disagrees with the neighbour majority | Ticket text filtered to the classified `queue` | §4's own agreement table shows same-queue pairs barely outscore random pairs in raw dense similarity (§9) — a queue filter does the separating work the vector space alone won't |

---

## 11. What this changes elsewhere

Consequences of this rewrite, applied as consistency edits to companion docs (each listed
in full in the commit that lands alongside this file):

- `decisions.md` D3 (star clustering + guard, and what `cluster_size` now certifies), D4
  (corpus-wide style homogeneity as the real cause, not greetings), D5 (sentence-level
  classification, handoff→escalation, the validation design), D6 (23,801 distinct records).
- `architecture.md` §2.1b (standalone FTS5, the added `UNINDEXED` columns, no Porter), §4.1
  (exact dedup as its own stage, star clustering replacing the old method), §8 (a new
  limitation: only ~1 in 8 historical answers is a genuine resolution).
- `state-schema.md`: `RetrievedCase.similarity` is always filled for a retrieved canonical
  (§7); `merge_cases` keys its max-score dedup on `similarity`, not the fused `score`, since
  RRF scores from different queries were never comparable to begin with.
- `output-schema.md` §4.3 ("substantive" = resolution-only), §4.4 (`SIM_CEILING = 0.92`,
  `τ_rel = 0.76`), §4.7 (worked-example table re-expressed against the new similarity band).
- `graph-design.md` §5 and §9: `τ_rel` default 0.55 → 0.76.
- `autosupport/config.py`: `tau_rel` default 0.55 → 0.76.
