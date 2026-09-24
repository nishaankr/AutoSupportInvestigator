import sqlite3

import pytest

from autosupport.rag.lexical import query_terms, search


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute(
        "CREATE VIRTUAL TABLE dataset_tickets_fts USING fts5("
        "subject, body, answer, tags, case_id UNINDEXED, source UNINDEXED, "
        "queue UNINDEXED, type UNINDEXED, answer_class UNINDEXED, "
        "tokenize = 'unicode61 remove_diacritics 2')"
    )
    c.execute("CREATE VIRTUAL TABLE dataset_tickets_fts_vocab USING fts5vocab(dataset_tickets_fts, 'row')")
    # The 20%-of-corpus document-frequency cap only means something once there are enough
    # documents for "20%" to exceed the minimum possible frequency of 1 — ten rows here
    # (vs. a handful) keeps the fixture from being a degenerate case of its own threshold.
    rows = [
        ("HF-1", "QNAP NAS shares unreachable", "shares not visible after firmware update", "network storage"),
        ("HF-2", "Billing question", "please provide your account number for the billing issue", "billing"),
        ("HF-3", "Password reset request", "please provide your account details to reset the password", "account"),
        ("HF-4", "General issue with software", "we recommend checking your settings for the issue", "general"),
        ("HF-5", "Billing dispute", "please provide your account number so we can review the charge", "billing"),
        ("HF-6", "Access issue", "please provide your account details for verification", "account"),
        ("HF-7", "Software crash report", "please provide details about the issue you are facing", "general"),
        ("HF-8", "Refund request", "please provide your account number for the refund", "billing"),
        ("HF-9", "Login issue", "please provide additional details about the issue", "account"),
        ("HF-10", "Feature request", "we recommend reviewing the documentation for this issue", "general"),
    ]
    for case_id, subject, body, tags in rows:
        c.execute(
            "INSERT INTO dataset_tickets_fts (subject, body, answer, tags, case_id, source, queue, type, answer_class) "
            "VALUES (?, ?, '', ?, ?, 'dataset', 'Technical Support', 'Incident', 'resolution')",
            (subject, body, tags, case_id),
        )
    yield c
    c.close()


def test_query_terms_excludes_high_frequency_words(conn):
    # "please", "provide", "account", "issue" all appear in >=20% of the 10-row corpus;
    # "qnap" and "nas" appear in exactly one row and should survive.
    terms = query_terms(conn, "QNAP NAS shares unreachable, please provide account details", n_docs=10)
    assert "qnap" in terms
    assert "nas" in terms
    assert "please" not in terms
    assert "provide" not in terms
    assert "account" not in terms


def test_search_finds_entity_match(conn):
    hits = search(conn, "QNAP NAS problem", n=10)
    assert hits
    assert hits[0].case_id == "HF-1"


def test_search_empty_query_returns_nothing(conn):
    # every token here (the, and, a) is common enough to be excluded as boilerplate,
    # or too short (<2 chars) to be a candidate at all.
    assert search(conn, "a", n=10) == []


def test_search_where_filters_on_unindexed_metadata_and_rejects_unknown_columns(conn):
    assert search(conn, "QNAP NAS problem", n=10, where={"queue": "Technical Support"})
    assert search(conn, "QNAP NAS problem", n=10, where={"queue": "Billing and Payments"}) == []
    with pytest.raises(ValueError):
        search(conn, "QNAP NAS problem", where={"body; DROP TABLE x": "1"})


def test_search_forces_quoted_phrases_even_when_the_text_has_no_rare_terms(conn):
    hits = search(conn, "a", n=10, phrases=["QNAP NAS"])
    assert hits and hits[0].case_id == "HF-1"


def test_ask_user_payload_is_a_pure_function_of_the_checkpoint():
    """Interrupt nodes re-run from the top on resume (graph-design.md §7.3): the pre-interrupt
    payload must rebuild identically, so the customer sees the question they answered."""
    from autosupport.graph.nodes.ask_user import build_payload

    state = {"ticket_id": "T-1", "pending_question": "Which version?"}
    assert build_payload(state) == build_payload(state)
