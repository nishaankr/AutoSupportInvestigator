"""Regression guard for the answer_class heuristic against the 200-row hand-labelled
validation fixture. Precision floors are set a little below
the last measured values so real regressions are caught without the test being brittle to
noise from an unchanged classifier. If a change to classify.py intentionally moves these
numbers, re-measure and update the floors here."""

import csv
from pathlib import Path

from autosupport.ingest.classify import classify_answer

FIXTURE = Path(__file__).parent / "fixtures" / "answer_class_validation.csv"

# Last measured: resolution 0.839 (n=31), escalation
# 0.840 (n=50), clarification_request 0.885 (n=61).
PRECISION_FLOOR = {
    "resolution": 0.75,
    "clarification_request": 0.80,
    "escalation": 0.75,
}


def _load_fixture() -> list[dict]:
    with FIXTURE.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_fixture_has_two_hundred_rows():
    assert len(_load_fixture()) == 200


def test_heuristic_precision_per_class():
    rows = _load_fixture()
    predictions = [(classify_answer(row["answer"]), row["hand_label"]) for row in rows]

    for cls, floor in PRECISION_FLOOR.items():
        predicted_as_cls = [hand_label for pred, hand_label in predictions if pred == cls]
        assert predicted_as_cls, f"heuristic never predicted {cls} on the validation sample"
        precision = sum(1 for h in predicted_as_cls if h == cls) / len(predicted_as_cls)
        assert precision >= floor, (
            f"{cls} precision {precision:.3f} (n={len(predicted_as_cls)}) fell below the "
            f"floor {floor} — re-run the validation and check for a real regression"
        )


def test_classify_residue_never_sends_an_empty_message_to_the_llm():
    """Regression: a full ingest crashed with a 400 ('user messages must have non-empty
    content') on a row whose answer was empty after normalisation."""
    from autosupport.ingest.classify import classify_residue

    class _Boom:
        def invoke(self, _messages):
            raise AssertionError("LLM called for an empty answer")

    for empty in ("", "   ", "<name> <tel_num>"):
        assert classify_residue(empty, _Boom()).answer_class == "escalation"
