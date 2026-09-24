"""CP1 profiling — run once against the real dataset, ahead of writing ingest code.
Not part of the ingest pipeline (see ingest/load.py, built later): this is throwaway-
reusable tooling, kept because re-profiling after a dataset update is cheap this way.

Snapshots the English subset to data/tickets_en.parquet (git-ignored, offline reuse —
`datasets` also caches the HF download itself, so re-running this costs no network call)
and writes a capped-length report to docs/profile-output.txt and stdout.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import load_dataset

DATASET_NAME = "Tobi-Bueck/customer-support-tickets"
SNAPSHOT_PATH = Path("data/tickets_en.parquet")
REPORT_PATH = Path("docs/profile-output.txt")
MAX_REPORT_LINES = 250

TAG_COLUMNS = [f"tag_{i}" for i in range(1, 9)]
TEXT_COLUMNS = ["subject", "body", "answer"]

# Kept identical to rag-design.md §4.2's starting answer_class heuristic — if one
# changes, change both, so the profiling report stays a true check on the real rules.
CLARIFICATION_PATTERNS = [
    r"could you (please )?(provide|specify|clarify|send|share|confirm)",
    r"can you (please )?(provide|send|share|confirm)",
    r"please (provide|send us|share|specify|confirm)",
    r"(what|which|when) (is|was|version|model)",
]
ESCALATION_PATTERNS = [
    r"escalat(e|ed|ing)",
    r"forwarded (your|the) (ticket|case|request)",
    r"(specialist|specialized) team",
    r"higher (support )?tier",
    r"senior (support|agent)",
]


def _norm(s: object) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def _is_filled(v: object) -> bool:
    return not (v is None or v == "" or (isinstance(v, float) and pd.isna(v)))


def _load_raw() -> pd.DataFrame:
    ds = load_dataset(DATASET_NAME, split="train")
    return ds.to_pandas()


def _dup_counts(df: pd.DataFrame, col: str) -> tuple[int, int]:
    """(rows sharing a normalized value with >=1 other row, distinct such values).
    Empty-after-normalization values are excluded — those are counted separately under
    NULL/EMPTY COUNTS, and lumping them in here would just measure "how many blank
    rows exist" rather than real duplication."""
    normed = df[col].map(_norm)
    counts = normed[normed != ""].value_counts()
    dup_values = counts[counts > 1]
    return int(dup_values.sum()), int(len(dup_values))


def _pattern_hit_rate(series: pd.Series, patterns: list[str]) -> tuple[int, float]:
    combined = re.compile("|".join(patterns), re.IGNORECASE)
    hits = series.fillna("").astype(str).map(lambda s: bool(combined.search(s)))
    return int(hits.sum()), 100 * hits.mean()


def _pick_sample_rows(en: pd.DataFrame) -> pd.DataFrame:
    """5 rows, distinct (queue, type) combos, at least one clarification-style answer."""
    combined = re.compile("|".join(CLARIFICATION_PATTERNS), re.IGNORECASE)
    is_clar = en["answer"].fillna("").astype(str).map(lambda s: bool(combined.search(s)))

    picked: list[int] = []
    seen_combos: set[tuple[str, str]] = set()

    clar_rows = en[is_clar]
    first_idx = clar_rows.index[0]
    picked.append(first_idx)
    seen_combos.add((en.loc[first_idx, "queue"], en.loc[first_idx, "type"]))

    for idx, row in en.iterrows():
        if len(picked) >= 5:
            break
        combo = (row["queue"], row["type"])
        if idx in picked or combo in seen_combos:
            continue
        picked.append(idx)
        seen_combos.add(combo)

    return en.loc[picked]


def build_report(raw: pd.DataFrame, en: pd.DataFrame) -> list[str]:
    lines: list[str] = []

    def emit(*parts: object) -> None:
        lines.append(" ".join(str(p) for p in parts))

    emit("=" * 70)
    emit(f"Dataset profile -- {DATASET_NAME}")
    emit("All sections below are over the English subset unless noted.")
    emit("=" * 70)
    emit()

    emit("SHAPE & COLUMNS")
    emit(f"  raw shape: {raw.shape}")
    emit(f"  english shape: {en.shape}")
    emit(f"  columns: {list(raw.columns)}")
    emit()

    emit("LANGUAGE x VERSION (raw, before the English filter)")
    version_key = raw["version"].fillna(-1).astype(int)
    cross = raw.groupby([raw["language"], version_key]).size().sort_values(ascending=False)
    for (lang, ver), c in cross.items():
        emit(f"  lang={lang!r} version={'None' if ver == -1 else ver}: {c}")
    emit()

    emit("ENGLISH ROWS PER VERSION")
    en_ver = en["version"].fillna(-1).astype(int).value_counts()
    for ver, c in en_ver.items():
        emit(f"  version={'None' if ver == -1 else ver}: {c}")
    emit()

    for col in ["queue", "type", "priority"]:
        emit(f"{col.upper()} VALUE COUNTS")
        for val, c in en[col].fillna("<null>").value_counts().items():
            emit(f"  {val}: {c}")
        emit()

    for col in ["body", "answer"]:
        emit(f"{col.upper()} LENGTH -- describe()")
        desc = en[col].fillna("").map(len).describe()
        for k, v in desc.items():
            emit(f"  {k}: {v:.1f}")
        emit()

    emit("NULL COUNTS (pandas isna(), all columns)")
    for col in raw.columns:
        emit(f"  {col}: {int(en[col].isna().sum())}")
    emit()
    emit("EMPTY-STRING COUNTS (text fields; distinct from null above)")
    for col in TEXT_COLUMNS:
        emit(f"  {col}: {int((en[col].fillna('') == '').sum())}")
    emit()

    emit("EXACT DUPLICATE COUNTS (normalized: lowercase, whitespace-collapsed)")
    for col in TEXT_COLUMNS:
        dup_rows, dup_clusters = _dup_counts(en, col)
        emit(f"  {col}: {dup_rows} rows across {dup_clusters} distinct duplicate values")
    emit()

    emit("TAGS PER ROW -- distribution (count of non-null tag_1..tag_8 per row)")
    tag_count = en[TAG_COLUMNS].apply(lambda r: sum(1 for v in r if _is_filled(v)), axis=1)
    for n, c in tag_count.value_counts().sort_index().items():
        emit(f"  {n} tags: {c} rows")
    emit()

    emit("TOP 50 FLATTENED TAGS (tag_1..tag_8 combined)")
    all_tags: list[str] = []
    for col in TAG_COLUMNS:
        all_tags.extend(v for v in en[col] if _is_filled(v))
    for tag, c in Counter(all_tags).most_common(50):
        emit(f"  {tag}: {c}")
    emit()

    emit("ANSWER PATTERN HIT RATES (same patterns as rag-design.md §4.2)")
    clar_n, clar_pct = _pattern_hit_rate(en["answer"], CLARIFICATION_PATTERNS)
    esc_n, esc_pct = _pattern_hit_rate(en["answer"], ESCALATION_PATTERNS)
    emit(f"  clarification-like: {clar_n} ({clar_pct:.1f}%)")
    emit(f"  escalation-like: {esc_n} ({esc_pct:.1f}%)")
    emit()

    emit("5 SAMPLE ROWS (untruncated, distinct queue/type, >=1 clarification-style)")
    for idx, row in _pick_sample_rows(en).iterrows():
        emit("-" * 70)
        emit(f"row index={idx}  queue={row['queue']!r}  type={row['type']!r}  priority={row['priority']!r}")
        emit(f"language={row['language']!r}  version={row['version']!r}")
        tags = [row[c] for c in TAG_COLUMNS if _is_filled(row[c])]
        emit(f"tags={tags}")
        emit(f"SUBJECT: {row['subject']}")
        emit(f"BODY: {row['body']}")
        emit(f"ANSWER: {row['answer']}")
    emit("-" * 70)

    return lines


def main() -> None:
    print(f"Loading {DATASET_NAME} via datasets...")
    raw = _load_raw()
    en = raw[raw["language"] == "en"].reset_index(drop=True)

    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    en.to_parquet(SNAPSHOT_PATH)
    print(f"Snapshot written: {SNAPSHOT_PATH} ({len(en)} rows)")

    lines = build_report(raw, en)
    if len(lines) > MAX_REPORT_LINES:
        lines = lines[: MAX_REPORT_LINES - 1] + [f"... truncated at {MAX_REPORT_LINES} lines ..."]

    report = "\n".join(lines)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nReport written: {REPORT_PATH} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
