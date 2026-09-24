"""RAG calibration harness. Every number in docs/design/rag-design.md comes from this run.

Reads the English snapshot written by scripts/profile_dataset.py and applies the
normalisation, exact dedup, answer_class heuristic and star clustering exactly as
rag-design.md specifies them. It then measures the retrieval parameters in memory: a numpy
matrix stands in for Chroma, and FTS5 is a real in-memory SQLite table. Nothing is persisted
except the report. The full corpus takes roughly 16 minutes on CPU, mostly embedding.

This is calibration tooling, not the ingest pipeline — ingest/ reuses the rules, not this file.

Usage: python scripts/rag_calibration.py [--limit N]
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sqlite3
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

SNAPSHOT_PATH = Path("data/tickets_en.parquet")
REPORT_PATH = Path("docs/rag-calibration-output.txt")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
TAG_COLUMNS = [f"tag_{i}" for i in range(1, 9)]

MIN_CONTENT_CHARS = 30
KNN_K = 50
CLUSTER_T = 0.92
ANSWER_T = 0.85
RRF_K = 10
OVERFETCH = 50
MMR_POOL = 30
MMR_LAMBDA = 0.7
FINAL_K = 10
BM25_TERMS = 12
BM25_MAX_DF = 0.20
TAU_REL = 0.76
N_QUERIES = 400

# ---------- normalisation (rag-design.md §2) ----------
PLACEHOLDER = re.compile(r"<[a-zA-Z_]+>|\{[a-zA-Z_]+\}|\[[A-Z][a-zA-Z ]*\]")
SALUTATION = re.compile(
    r"^((dear|hello|hi|greetings|respected)\b[^.!?,]*[,.!]?"
    r"|(customer )?(support|service)( team)?,"
    r"|i hope (this|my) (message|email) (finds|reaches) you[^.!?]*[.!?])\s*",
    re.I,
)


def index_text(s: str, placeholder: str = " ") -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\\n", " ")  # literal backslash-n, not a newline
    s = re.sub(r"<br\s*/?>", " ", s, flags=re.I)
    s = PLACEHOLDER.sub(placeholder, s)
    return re.sub(r"\s+", " ", s).strip()


def index_body(s: str) -> str:
    s = index_text(s)
    for _ in range(3):  # salutation and "I hope this message..." can stack
        s = SALUTATION.sub("", s).strip()
    return s


# ---------- answer_class (rag-design.md §5) ----------
def _rx(p: str) -> re.Pattern:
    return re.compile(p, re.I)


GENERIC_OBJECT = (
    r"(a )?(range|variety|number) of|(customi[sz]ed|tailored|comprehensive|various|several|"
    r"different|our) (solutions|services|options|strategies)|solutions|services|options|"
    r"assistance|support|help|guidance"
)
SUBSTANCE_STEP = _rx(
    r"\b(we|i) (would )?(recommend|suggest|advise)\b|\bit is (recommended|advisable|best)\b"
    r"|\byou (will )?need to (create|obtain|install|update|enable|disable|configure|reset|download|use|restart|clear|verify)\b"
    r"|(^|\bplease |\byou (can|could|may|should) |\bto (resolve|fix|address) (this|the issue),? )"
    r"(try|restart|reboot|reinstall|update|upgrade|clear|reset|disable|enable|configure|navigate|go to|click|select|open"
    r"|log (in|out)|sign (in|out)|install|uninstall|download|access|use|check (your|the)|verify (your|the)|ensure|visit"
    r"|adjust|change|switch|run)\b"
)
SUBSTANCE_FACT = _rx(
    r"\b(we|our (company|team|service|platform|products?|software|system)) (offer|offers|provide|provides|support"
    r"|supports|accept|accepts|include|includes) (?!" + GENERIC_OBJECT + r")\w+"
    r"|\b(our|the|this|these) (products?|platform|software|service|system|plan|subscription|devices?|tool|feature"
    r"|policy|billing (cycle|period)|period|warranty|integration|api|update) (supports?|includes?|offers?|begins?"
    r"|requires?|allows?|covers?|is (available|compatible|supported|included|designed)|are (available|compatible"
    r"|supported|included))\b"
    r"|\b(is|are) (compatible with|available (in|on|for|via)|supported (on|by|for))\b"
    r"|\bhas been (resolved|fixed|restored|updated|processed|refunded|corrected|issued|credited)\b"
)
ESCALATION = _rx(
    r"\bescalat(e|ed|ing)\b|\bforwarded (your|the|this)\b|\btransferr?(ed|ing) (your|the|this)\b"
    r"|\b(specialist|specialized|dedicated|senior|expert|second[- ]level|tier[- ]?2) (team|agent|engineers?|support|department)\b"
    r"|\bhigher (support )?tier\b"
)
REQUEST = _rx(
    r"\b(could|can|would) you (kindly |please )?(provide|share|send|specify|confirm|clarify|tell us|let (us|me) know|describe|list)\b"
    r"|\b(please|kindly) (kindly )?(provide|share|send( us| me)?|specify|confirm|clarify|describe|list"
    r"|let (us|me) know (the|which|what|your|if|whether|more))\b"
    r"|\b(we|i) (need|require|would need|will need) (more|additional|further|some|the|your)\b"
)
CALL_TIME = _rx(
    r"(suitable|convenient|preferred) (time|date)|schedule (a|the) (call|meeting)|\bavailability\b"
    r"|at your convenience|\breach you\b|\bcontact you\b|\bcall you\b|<X>"
)
DEFERRAL = _rx(
    r"\b(will|shall|'ll) (contact|reach out|call|get back|follow up|be in touch|look into|investigate|review|examine"
    r"|analy[sz]e|update you|keep you|proceed|revise|launch|implement|arrange|schedule|prepare|work on"
    r"|provide (guidance|assistance|support|details|information|an update))\b"
    r"|\b(is|are|am) (currently )?(investigating|looking into|reviewing|working on|examining|analy[sz]ing|prioriti[sz]ing)\b"
    r"|^investigating\b|\bwould like to (investigate|discuss|schedule|review|look)\b|\blet'?s (arrange|schedule|set up)\b"
    r"|(happy|glad) to discuss|available (for a call|to discuss)|allow us to contact|sent via (a )?separate|\bhave been sent\b"
)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def classify_answer(answer: str) -> str:
    """Returns resolution | escalation | clarification_request | residue (-> fast-tier LLM)."""
    # Placeholders become <X> here only: "call you at <tel_num>" is itself a handoff signal.
    text = index_text(answer, placeholder="<X>")
    substance = escalation = request = handoff = 0
    for s in (x for x in SENTENCE_SPLIT.split(text) if x):
        if ESCALATION.search(s):
            escalation += 1
        if (SUBSTANCE_STEP.search(s) or SUBSTANCE_FACT.search(s)) and len(s.split()) >= 6:
            substance += 1
        if REQUEST.search(s) and not CALL_TIME.search(s):
            request += 1
        elif CALL_TIME.search(s) or DEFERRAL.search(s):
            handoff += 1
    if substance:
        return "resolution"
    if escalation:
        return "escalation"
    if request:
        return "clarification_request"
    if handoff:
        return "escalation"
    return "residue"


ENTITY = re.compile(r"\b(?:[A-Z][a-z]*[A-Z0-9][A-Za-z0-9.]*|[A-Z]{2,}[0-9A-Za-z]*|[A-Za-z]+[0-9][A-Za-z0-9.]*)\b")


# ---------- report ----------
class Report:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.t0 = time.time()

    def __call__(self, *parts: object) -> None:
        line = " ".join(str(p) for p in parts)
        self.lines.append(line)
        print(line, flush=True)

    def progress(self, msg: str) -> None:
        print(f"  [{time.time() - self.t0:5.0f}s] {msg}", flush=True)


def pct(values, ps=(50, 90, 95, 99, 99.9)) -> str:
    return " ".join(f"p{p}={np.percentile(values, p):.3f}" for p in ps)


# ---------- corpus ----------
def load_distinct(limit: int, out: Report) -> pd.DataFrame:
    raw = pd.read_parquet(SNAPSHOT_PATH)
    for c in ("subject", "body", "answer"):
        raw[c] = raw[c].fillna("").astype(str)
    raw["version_key"] = raw["version"].fillna(-1).astype(int)
    raw["subject_ix"] = raw["subject"].map(index_text)
    raw["body_ix"] = raw["body"].map(index_body)
    raw["answer_ix"] = raw["answer"].map(index_text)
    raw["dedup_key"] = (raw["subject_ix"] + "|" + raw["body_ix"] + "|" + raw["answer_ix"]).str.lower()

    groups = raw.groupby("dedup_key")["version_key"].agg(lambda v: tuple(sorted(v)))
    dup_groups = groups[groups.map(len) > 1]
    out("## 1. Corpus")
    out(f"English rows: {len(raw)}")
    out(f"Exact-duplicate groups on index-text (subject|body|answer): {len(dup_groups)}; "
        f"group sizes: {dup_groups.map(len).value_counts().to_dict()}")
    out(f"Version combos (-1 = None): {dup_groups.value_counts().head(5).to_dict()}")

    # Keep the copy with a labelled version (v400) over its unlabelled (None) twin.
    d = (raw.sort_values("version_key", ascending=False, kind="stable")
         .drop_duplicates("dedup_key").sort_index().reset_index(drop=True))
    out(f"Distinct records: {len(d)}  (dropped {len(raw) - len(d)})")
    out(f"Distinct records without a subject: {int((d['subject_ix'] == '').sum())}")
    if limit:
        d = d.sample(limit, random_state=0).reset_index(drop=True)
        out(f"LIMITED RUN: {limit} records sampled — numbers below are NOT the full-corpus figures")
    d["embed_text"] = np.where(d["subject_ix"] != "", d["subject_ix"] + ". " + d["body_ix"], d["body_ix"])
    d["too_short"] = d["embed_text"].str.len() < MIN_CONTENT_CHARS
    out(f"Below the {MIN_CONTENT_CHARS}-char minimum-content guard: {int(d['too_short'].sum())}")
    return d


def report_answer_classes(d: pd.DataFrame, out: Report) -> None:
    d["answer_class"] = d["answer"].map(classify_answer)
    d["test_pool"] = d["dedup_key"].map(lambda k: int(hashlib.md5(k.encode()).hexdigest(), 16) % 10 == 0)
    out("\n## 2. answer_class heuristic")
    out("All distinct:", d["answer_class"].value_counts(normalize=True).round(3).to_dict())
    out("Counts:", d["answer_class"].value_counts().to_dict())
    out(f"Test pool (md5 % 10 == 0): {int(d['test_pool'].sum())} records; dev: {int((~d['test_pool']).sum())}")
    out("Test-pool class counts:", d.loc[d["test_pool"], "answer_class"].value_counts().to_dict())


# ---------- embeddings and kNN ----------
def embed(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)
    return model.encode(texts, normalize_embeddings=True, batch_size=128,
                        show_progress_bar=False).astype(np.float32)


def knn(E: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(E)
    nbr = np.zeros((n, k), np.int32)
    sim = np.zeros((n, k), np.float32)
    for start in range(0, n, 1024):
        S = E[start:start + 1024] @ E.T
        S[np.arange(len(S)), np.arange(start, start + len(S))] = -1  # never your own neighbour
        part = np.argpartition(-S, k, axis=1)[:, :k]
        ps = np.take_along_axis(S, part, 1)
        order = np.argsort(-ps, 1)
        nbr[start:start + 1024] = np.take_along_axis(part, order, 1)
        sim[start:start + 1024] = np.take_along_axis(ps, order, 1)
    return nbr, sim


def random_pair_sims(E: np.ndarray, n: int = 1500) -> np.ndarray:
    idx = np.random.default_rng(0).choice(len(E), min(n, len(E)), replace=False)
    S = E[idx] @ E[idx].T
    return S[np.triu_indices(len(idx), 1)]


# ---------- star clustering (rag-design.md §4) ----------
class Clusterer:
    def __init__(self, d: pd.DataFrame, E: np.ndarray, EA: np.ndarray, nbr: np.ndarray, sim: np.ndarray):
        self.E, self.EA, self.nbr, self.sim = E, EA, nbr, sim
        self.cls = d["answer_class"].values
        self.ents = [set(ENTITY.findall(t)) for t in d["embed_text"]]
        self.body_len = d["body_ix"].str.len().values
        self.eligible = ~d["too_short"].values

    def compatible(self, i: int, j: int, answer_t: float) -> bool:
        if self.cls[i] != self.cls[j]:
            return False
        if float(self.EA[i] @ self.EA[j]) < answer_t:
            return False
        a, b = self.ents[i], self.ents[j]
        return not (a - b and b - a)  # conflict = each names an entity the other lacks

    def star(self, t: float, guarded: bool = True, answer_t: float = ANSWER_T) -> np.ndarray:
        n = len(self.E)
        degree = (self.sim >= t).sum(1)
        order = np.lexsort((-self.body_len, -degree))  # most central first, then most complete
        leader = np.full(n, -1)
        for i in order:
            if leader[i] >= 0:
                continue
            leader[i] = i
            if not self.eligible[i]:
                continue
            for k in range(self.nbr.shape[1]):
                if self.sim[i, k] < t:
                    break
                j = self.nbr[i, k]
                if leader[j] >= 0 or not self.eligible[j]:
                    continue
                if guarded and not self.compatible(i, j, answer_t):
                    continue
                leader[j] = i
        return leader

    def describe(self, leader: np.ndarray) -> str:
        sizes = pd.Series(leader).value_counts()
        buckets = pd.cut(sizes, [0, 1, 3, 7, 15, 1e9], labels=["1", "2-3", "4-7", "8-15", "16+"])
        min_pairs = []
        for lead, size in sizes[sizes >= 3].items():
            ids = np.where(leader == lead)[0]
            S = self.E[ids] @ self.E[ids].T
            min_pairs.append(S[np.triu_indices(len(ids), 1)].min())
        n = len(leader)
        return (f"canonicals={len(sizes)} collapsed={100 * (n - len(sizes)) / n:.1f}% max={sizes.max()} "
                f"sizes={buckets.value_counts().sort_index().to_dict()} "
                f"minpair_p5={np.percentile(min_pairs, 5) if min_pairs else float('nan'):.3f}")


def report_similarity_and_clustering(d, E, EA, nbr, sim, out: Report) -> Clusterer:
    out("\n## 3. Similarity baselines")
    out("Random-pair cosine, embed text:", pct(random_pair_sims(E)))
    out("Random-pair cosine, answers:   ", pct(random_pair_sims(EA)))
    has_subject = np.where(d["subject_ix"].values != "")[0]
    sample = np.random.default_rng(1).choice(has_subject, min(2000, len(has_subject)), replace=False)
    EB = embed(d["body_ix"].iloc[sample].tolist())
    out("cos(subject+body, body-only), same ticket:", pct((EB * E[sample]).sum(1), (10, 50, 90)))

    out("\n## 4. Problem-similarity band vs answer / entity agreement (top-3 neighbours, sim >= 0.86)")
    ents = [set(ENTITY.findall(t)) for t in d["embed_text"]]
    rows = [(i, int(nbr[i, k]), float(sim[i, k])) for i in range(len(E)) for k in range(3)
            if sim[i, k] >= 0.86 and i < nbr[i, k]]
    pairs = pd.DataFrame(rows, columns=["i", "j", "sim"])
    pairs = pairs.sample(min(6000, len(pairs)), random_state=0)
    pairs["answer_sim"] = (EA[pairs.i] * EA[pairs.j]).sum(1)
    pairs["entity_conflict"] = [bool(ents[a] - ents[b] and ents[b] - ents[a]) for a, b in zip(pairs.i, pairs.j)]
    pairs["band"] = pd.cut(pairs.sim, [0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.001])
    table = pairs.groupby("band", observed=True).agg(
        n=("sim", "size"), answer_sim_median=("answer_sim", "median"),
        answer_sim_below_080=("answer_sim", lambda x: (x < 0.80).mean()),
        answer_sim_below_085=("answer_sim", lambda x: (x < 0.85).mean()),
        entity_conflict=("entity_conflict", "mean")).round(3)
    out(table.to_string())

    out("\n## 5. Star clustering sweep")
    c = Clusterer(d, E, EA, nbr, sim)
    for t in (0.90, 0.92, 0.94, 0.96):
        out(f"T={t:.2f} unguarded: {c.describe(c.star(t, guarded=False))}")
        out(f"T={t:.2f} guarded:   {c.describe(c.star(t))}")
    for at in (0.80, 0.90):
        out(f"T=0.92 guarded T_a={at:.2f}: {c.describe(c.star(0.92, answer_t=at))}")
    return c


# ---------- retrieval (rag-design.md §6-§9) ----------
class Retriever:
    def __init__(self, d: pd.DataFrame, E: np.ndarray, canon: np.ndarray):
        self.d, self.E, self.canon = d, E, canon
        self.Ec = E[canon]
        tags = d[TAG_COLUMNS].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
        self.db = sqlite3.connect(":memory:")
        self.db.execute(
            "CREATE VIRTUAL TABLE dataset_tickets_fts USING fts5("
            "subject, body, answer, tags, case_id UNINDEXED, source UNINDEXED, queue UNINDEXED, "
            "type UNINDEXED, answer_class UNINDEXED, tokenize = 'unicode61 remove_diacritics 2')")
        self.db.executemany(
            "INSERT INTO dataset_tickets_fts VALUES (?,?,?,?,?,?,?,?,?)",
            [(d.subject_ix[c], d.body_ix[c], d.answer_ix[c], tags[c], int(c), "dataset",
              d.queue[c], d.type[c], d.answer_class[c]) for c in canon])
        self.db.execute("CREATE VIRTUAL TABLE fts_vocab USING fts5vocab(dataset_tickets_fts, 'row')")
        self.doc_freq = dict(self.db.execute("SELECT term, doc FROM fts_vocab"))
        self.n_docs = len(canon)

    def query_terms(self, text: str) -> list[str]:
        cap = BM25_MAX_DF * self.n_docs
        cands = {t for t in re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", text).lower())
                 if len(t) > 1 and self.doc_freq.get(t, cap) < cap}
        return sorted(cands, key=lambda t: (self.doc_freq[t], t))[:BM25_TERMS]  # rarest = highest IDF

    def bm25(self, text: str, exclude=None, n: int = OVERFETCH, weights=None) -> list[int]:
        terms = self.query_terms(text)
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in terms)
        rank = "bm25(dataset_tickets_fts)" if weights is None else \
            "bm25(dataset_tickets_fts, %s, %s, %s, %s)" % weights
        rows = self.db.execute(
            f"SELECT case_id FROM dataset_tickets_fts WHERE dataset_tickets_fts MATCH ? ORDER BY {rank} LIMIT ?",
            (match, n + 1))
        return [r for (r,) in rows if r != exclude][:n]

    def dense(self, qv: np.ndarray, exclude=None, n: int = OVERFETCH) -> list[int]:
        s = self.Ec @ qv
        top = np.argsort(-s)[:n + 1]
        return [int(self.canon[t]) for t in top if self.canon[t] != exclude][:n]

    @staticmethod
    def rrf(lists: list[list[int]], k: int = RRF_K) -> tuple[list[int], dict[int, float]]:
        score: dict[int, float] = {}
        for ranked in lists:
            for rank, doc in enumerate(ranked, 1):
                score[doc] = score.get(doc, 0.0) + 1.0 / (k + rank)
        # ties break on the first list's (dense) order via stable sort
        order = {doc: i for i, doc in enumerate(lists[0])}
        return sorted(score, key=lambda x: (-score[x], order.get(x, 10**9))), score

    def mmr(self, fused: list[int], score: dict[int, float], lam: float = MMR_LAMBDA) -> list[int]:
        pool = fused[:MMR_POOL]
        raw = np.array([score[x] for x in pool])
        rel = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        chosen: list[int] = []
        rest = list(range(len(pool)))
        while rest and len(chosen) < FINAL_K:
            def value(x):
                redundancy = max((float(self.E[pool[x]] @ self.E[pool[s]]) for s in chosen), default=0.0)
                return lam * rel[x] - (1 - lam) * redundancy
            best = max(rest, key=value)
            chosen.append(best)
            rest.remove(best)
        return [pool[x] for x in chosen]


def pair_sim(E, docs):
    return np.mean([float(E[a] @ E[b]) for i, a in enumerate(docs) for b in docs[i + 1:]])


def report_retrieval(d, E, leader, out: Report) -> None:
    canon = np.where((leader == np.arange(len(d))) & ~d["too_short"].values)[0]
    members = np.where(leader != np.arange(len(d)))[0]
    r = Retriever(d, E, canon)
    rng = np.random.default_rng(1)
    out(f"\n## 6. Retrieval (index = {len(canon)} canonicals at T={CLUSTER_T}, T_a={ANSWER_T}, guarded)")
    out("Canonical answer classes:", d.loc[canon, "answer_class"].value_counts(normalize=True).round(3).to_dict())

    variants = {"weighted_2_1_.5_1.5": (2, 1, 0.5, 1.5), "answer_weight_0": (1, 1, 0, 1)}
    res: dict[str, list] = {}

    def add(key, val):
        res.setdefault(key, []).append(val)

    # Q1: a cluster member (never indexed) should retrieve its own canonical.
    for q in rng.choice(members, min(N_QUERIES, len(members)), replace=False):
        target, text = leader[q], d.embed_text[q]
        D, B = r.dense(E[q]), r.bm25(text)
        add("dense", target in D[:FINAL_K])
        add("bm25_flat", target in B[:FINAL_K])
        for name, w in variants.items():
            add("bm25_" + name, target in r.bm25(text, weights=w)[:FINAL_K])
        for k in (5, 10, 20, 60):
            fused, score = r.rrf([D, B], k)
            add(f"rrf_k{k}", target in fused[:FINAL_K])
            if k == RRF_K:
                for lam in (0.5, 0.7, 0.85):
                    add(f"mmr_lambda{lam}", target in r.mmr(fused, score, lam))
        for depth in (25, 100):
            add(f"rrf_k10_depth{depth}", target in r.rrf([r.dense(E[q], n=depth), r.bm25(text, n=depth)])[0][:FINAL_K])
    out("Q1 member->canonical recall@10:", {k: round(float(np.mean(v)), 3) for k, v in res.items()})

    # Q2: a canonical naming a rare entity; how many of the top 10 name the same entity?
    ents = [set(ENTITY.findall(t)) for t in d.embed_text]
    ent_df = pd.Series([e for c in canon for e in {x.lower() for x in ents[c]}]).value_counts()
    rare = set(ent_df[(ent_df >= 2) & (ent_df <= 30)].index)
    cands = [c for c in canon if any(e.lower() in rare for e in ents[c])]

    def names(doc, ent):
        return re.search(r"\b" + re.escape(ent) + r"\b", d.embed_text[doc], re.I) is not None

    res, overlap, survive, third_best = {}, [], {}, []
    for q in rng.choice(cands, min(N_QUERIES, len(cands)), replace=False):
        ent = min((x for x in ents[q] if x.lower() in rare), key=lambda x: ent_df[x.lower()])
        text = d.embed_text[q]
        D, B = r.dense(E[q], exclude=q), r.bm25(text, exclude=q)
        third_best.append(float(E[D[2]] @ E[q]))
        overlap.append(len(set(D) & set(B)))
        frac = lambda docs: np.mean([names(x, ent) for x in docs[:FINAL_K]]) if docs else 0.0
        add("dense", frac(D))
        add("bm25_flat", frac(B))
        for name, w in variants.items():
            add("bm25_" + name, frac(r.bm25(text, exclude=q, weights=w)))
        lexical_only = next((x for x in B if names(x, ent)), None)
        for k in (5, 10, 20, 60):
            fused, score = r.rrf([D, B], k)
            add(f"rrf_k{k}", frac(fused))
            if lexical_only is not None and lexical_only not in D:
                survive.setdefault(k, []).append(lexical_only in fused[:FINAL_K])
            if k == RRF_K:
                add("pairsim_no_mmr", pair_sim(E, fused[:FINAL_K]))
                for lam in (0.5, 0.7, 0.85):
                    chosen = r.mmr(fused, score, lam)
                    add(f"mmr_lambda{lam}", frac(chosen))
                    add(f"pairsim_mmr{lam}", pair_sim(E, chosen))
                    add(f"swapped_mmr{lam}", len(set(chosen) - set(fused[:FINAL_K])))
        for depth in (25, 100):
            add(f"rrf_k10_depth{depth}",
                frac(r.rrf([r.dense(E[q], exclude=q, n=depth), r.bm25(text, exclude=q, n=depth)])[0]))
    out("Q2 entity@10 (and MMR diagnostics):", {k: round(float(np.mean(v)), 3) for k, v in res.items()})
    out(f"Q2 arm overlap |dense50 & bm25_50|: p25={np.percentile(overlap, 25):.0f} "
        f"p50={np.percentile(overlap, 50):.0f} p75={np.percentile(overlap, 75):.0f}")
    out("Q2 lexical-only entity hit survives fused top-10 (rate, n):",
        {k: (round(float(np.mean(v)), 3), len(v)) for k, v in survive.items()})
    out(f"3rd-best dense similarity per query: {pct(third_best, (10, 25, 50))}; "
        f"share >= tau_rel {TAU_REL}: {np.mean(np.array(third_best) >= TAU_REL):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    out = Report()
    out("# RAG calibration output (scripts/rag_calibration.py)")
    d = load_distinct(args.limit, out)
    report_answer_classes(d, out)
    out.progress("embedding embed_text")
    E = embed(d["embed_text"].tolist())
    out.progress("embedding answers")
    EA = embed(d["answer_ix"].tolist())
    out.progress("kNN")
    nbr, sim = knn(E, KNN_K)
    clusterer = report_similarity_and_clustering(d, E, EA, nbr, sim, out)
    report_retrieval(d, E, clusterer.star(CLUSTER_T), out)
    out.progress("done")

    REPORT_PATH.write_text("\n".join(out.lines) + "\n", encoding="utf-8")
    print(f"\nReport written: {REPORT_PATH}")


if __name__ == "__main__":
    main()
