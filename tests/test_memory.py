"""memory-design.md §4 write policy, enforced by `graph/memory.apply_update` (W1-W6)."""

from __future__ import annotations

from autosupport.graph.memory import (
    MAX_TRIED_FIXES, MemoryUpdate, RememberedFact, RememberedFix, RememberedPreference, apply_update,
)
from autosupport.graph.state import CustomerMemory

SOURCE = (
    "Backups to S3 fail. We run Synology DS920+ on DSM 7.2 and I already restarted Hyper Backup. "
    "Please keep it simple, I'm not comfortable with the command line. My email is ann@corp.com"
)


def _apply(update: MemoryUpdate, profile=None, escalations=0, ticket="T-20260924-aaaaaa") -> CustomerMemory:
    return apply_update(profile, "C-1", update, SOURCE, ticket, recent_escalations=escalations)


def test_keeps_customer_stated_items_with_provenance():
    memory = _apply(MemoryUpdate(
        facts=[RememberedFact(key="product", value="Synology DS920+", quote="we run synology DS920+")],
        preferences=[RememberedPreference(key="technical_level", value="no command line",
                                          quote="not comfortable with the command line")],
        tried_fixes=[RememberedFix(fix="restarted Hyper Backup", quote="I already restarted Hyper Backup.")],
        reasoning="r"))
    assert memory.facts == {"product": "Synology DS920+"}
    assert memory.preferences == {"technical_level": "no command line"}
    assert memory.tried_fixes == ["restarted Hyper Backup"]
    assert memory.provenance == {"facts.product": "T-20260924-aaaaaa", "preferences.technical_level": "T-20260924-aaaaaa"}


def test_w1_drops_items_whose_quote_the_customer_never_wrote():
    memory = _apply(MemoryUpdate(
        facts=[RememberedFact(key="plan", value="enterprise", quote="enterprise customer")],
        tried_fixes=[RememberedFix(fix="rebooted NAS", quote="")], reasoning="r"))
    assert memory.facts == {} and memory.tried_fixes == []


def test_w3_drops_contact_pii_and_secrets_even_when_quoted():
    memory = _apply(MemoryUpdate(
        preferences=[RememberedPreference(key="contact_channel", value="ann@corp.com", quote="My email is ann@corp.com")],
        reasoning="r"))
    assert memory.preferences == {}


def test_w5_facts_overwrite_and_tried_fixes_dedupe_and_cap():
    old = CustomerMemory(customer_id="C-1", facts={"os": "DSM 6"}, provenance={"facts.os": "T-old"},
                         tried_fixes=[f"fix {i}" for i in range(MAX_TRIED_FIXES)] + ["Restarted hyper backup"])
    memory = _apply(MemoryUpdate(
        facts=[RememberedFact(key="os", value="DSM 7.2", quote="on DSM 7.2")],
        tried_fixes=[RememberedFix(fix="restarted Hyper Backup", quote="restarted Hyper Backup")], reasoning="r"),
        profile=old)
    assert memory.facts["os"] == "DSM 7.2" and memory.provenance["facts.os"] == "T-20260924-aaaaaa"
    assert memory.tried_fixes[-1] == "restarted Hyper Backup"
    assert len(memory.tried_fixes) == MAX_TRIED_FIXES
    assert sum(f.lower() == "restarted hyper backup" for f in memory.tried_fixes) == 1


def test_w6_repeat_unresolved_is_computed_and_clears():
    flagged = _apply(MemoryUpdate(reasoning="r"), escalations=2)
    assert flagged.flags == ["repeat_unresolved"]
    cleared = _apply(MemoryUpdate(reasoning="r"), profile=flagged, escalations=1)
    assert cleared.flags == []


def test_gate_only_calls_the_model_when_the_customer_text_has_a_candidate():
    from autosupport.graph.memory import has_memory_candidates

    assert has_memory_candidates("We run DSM 7.2 on our NAS")          # version
    assert has_memory_candidates("I already restarted the router")     # tried fix
    assert has_memory_candidates("Please reply by email")              # preference
    assert not has_memory_candidates("How do I export a report to PDF?")
