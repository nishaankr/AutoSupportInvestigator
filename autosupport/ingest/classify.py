"""answer_class heuristic + LLM residue pass.

Heuristic first, sentence-level, in this precedence: a substantive step or fact ->
`resolution`; an explicit escalation phrase -> `escalation`; a request for missing facts ->
`clarification_request`; a call-scheduling or "we'll look into it" deferral -> `escalation`
(historical handoffs carry no grounded fix, same as an explicit escalation); otherwise
`residue`, sent to the fast-tier LLM. A model never runs over the whole corpus: the residue
is a small minority by construction.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel

from autosupport.ingest.text import index_text

AnswerClass = Literal["resolution", "clarification_request", "escalation"]

_GENERIC_OBJECT = (
    r"(a )?(range|variety|number) of|(customi[sz]ed|tailored|comprehensive|various|several|"
    r"different|our) (solutions|services|options|strategies)|solutions|services|options|"
    r"assistance|support|help|guidance"
)

_ACTION_VERBS = (
    r"try|restart|reboot|reinstall|update|upgrade|clear|reset|disable|enable|configure|navigate|go to|click|select"
    r"|open|log (in|out)|sign (in|out)|install|uninstall|download|access|use|check (your|the)|verify (your|the)"
    r"|ensure|visit|adjust|change|switch|run"
)
# Strict imperative commands ("please restart", "you will need to configure") are concrete
# regardless of topic — no specificity gate needed.
SUBSTANCE_STEP = re.compile(
    r"\byou (will )?need to (create|obtain|install|update|enable|disable|configure|reset|download|use|restart|clear|verify)\b"
    r"|(^|\bplease |\bto (resolve|fix|address) (this|the issue),? )(" + _ACTION_VERBS + r")\b",
    re.I,
)
# Softer suggestion forms ("we recommend X", "you could try X"). Not gated by SPECIFICITY —
# validation found that broke far more genuine cases ("we recommend enabling encryption",
# "we suggest using access controls") than it fixed, since concrete security/config advice
# rarely names a number or acronym. The one failure mode this shares with SUBSTANCE_FACT —
# "we recommend escalating" — is handled by the is_escalation exclusivity check below instead.
SUBSTANCE_SUGGESTION = re.compile(
    r"\b(we|i) (would )?(recommend|suggest|advise)\b|\bit is (recommended|advisable|best)\b"
    r"|\byou (can|could|may|should) (" + _ACTION_VERBS + r")\b",
    re.I,
)
SUBSTANCE_FACT = re.compile(
    r"\b(we|our (company|team|service|platform|products?|software|system)) (offer|offers|provide|provides|support"
    r"|supports|accept|accepts|include|includes) (?!" + _GENERIC_OBJECT + r")\w+"
    r"|\b(our|the|this|these) (products?|platform|software|service|system|plan|subscription|devices?|tool|feature"
    r"|policy|billing (cycle|period)|period|warranty|integration|api|update) (supports?|includes?|offers?|begins?"
    r"|requires?|allows?|covers?|is (available|compatible|supported|included|designed)|are (available|compatible"
    r"|supported|included))\b"
    r"|\b(is|are) (compatible with|available (in|on|for|via)|supported (on|by|for))\b"
    r"|\bhas been (resolved|fixed|restored|updated|processed|refunded|corrected|issued|credited)\b"
    r"|\b\w+ integrates? with\b"  # "your product integrates with Salesforce and Smartsheet"
    r"|\b(solutions?|options?|steps?|measures?) include\b"  # "solutions include implementing X"
    r"|\b(we|our team) (understand|believe) that\b.{0,60}\b(is|was) (likely )?(caused by|due to)\b",  # diagnosis
    re.I,
)
# A SUBSTANCE_FACT match alone is not enough: "our platform offers seamless integration
# services" matches it too, because the exclusion list below only tests the single word
# right after the verb ("seamless"), not the real head noun two words later ("services").
# Enumerating every vague-marketing adjective/noun pair is whack-a-mole (validation found
# "seamless integration services", "multiple methods", "customized digital strategies" all
# slipping past a same-shaped exclusion). Requiring positive evidence of a specific fact
# generalises where a negative list can't: a number, an acronym, or a named product/brand
# (this corpus names them constantly — Salesforce, PostgreSQL, Bitdefender — and a mid-
# sentence capitalised word that isn't ordinary sentence vocabulary is one on this corpus).
_SPECIFICITY_DIGIT_OR_ACRONYM = re.compile(r"\d|\b[A-Z]{2,}\b")
_MID_SENTENCE_CAPITALISED_WORD = re.compile(r"(?<=.)\b[A-Z][a-zA-Z]{2,}\b")
_COMMON_CAPITALISED_WORDS = frozenset({
    "we", "our", "us", "you", "your", "yours", "the", "this", "these", "those", "that",
    "to", "if", "for", "and", "but", "so", "it", "its", "they", "he", "she", "an",
    "is", "are", "was", "were", "will", "would", "could", "can", "may", "might", "should",
    "must", "additionally", "however", "also", "regarding", "kindly", "furthermore",
    "thank", "thanks", "dear", "hello", "hi", "sincerely", "best", "please", "since",
    "once", "when", "while", "after", "before", "certainly", "alternatively",
    "unfortunately", "currently", "review", "here", "there", "esteemed", "customer",
})


def _has_specificity(sentence: str) -> bool:
    if _SPECIFICITY_DIGIT_OR_ACRONYM.search(sentence):
        return True
    return any(
        w.lower() not in _COMMON_CAPITALISED_WORDS for w in _MID_SENTENCE_CAPITALISED_WORD.findall(sentence)
    )

ESCALATION = re.compile(
    r"\bescalat(e|ed|ing)\b|\bforwarded (your|the|this)\b|\btransferr?(ed|ing) (your|the|this)\b"
    r"|\b(specialist|specialized|dedicated|senior|expert|second[- ]level|tier[- ]?2) (team|agent|engineers?|support|department)\b"
    r"|\bhigher (support )?tier\b",
    re.I,
)
REQUEST = re.compile(
    r"\b(could|can|would) you (kindly |please )?(provide|share|send|specify|confirm|clarify|tell us|let (us|me) know|describe|list)\b"
    r"|\b(please|kindly) (kindly )?(provide|share|send( us| me)?|specify|confirm|clarify|describe|list"
    r"|let (us|me) know (the|which|what|your|if|whether|more))\b"
    r"|\b(we|i) (need|require|would need|will need) (more|additional|further|some|the|your)\b",
    re.I,
)
CALL_TIME = re.compile(
    r"(suitable|convenient|preferred) (time|date)|schedule (a|the) (call|meeting)|\bavailability\b"
    r"|at your convenience|\breach you\b|\bcontact you\b|\bcall you\b|<X>",
    re.I,
)
DEFERRAL = re.compile(
    r"\b(will|shall|'ll) (contact|reach out|call|get back|follow up|be in touch|look into|investigate|review|examine"
    r"|analy[sz]e|update you|keep you|proceed|revise|launch|implement|arrange|schedule|prepare|work on"
    r"|provide (guidance|assistance|support|details|information|an update))\b"
    r"|\b(is|are|am) (currently )?(investigating|looking into|reviewing|working on|examining|analy[sz]ing|prioriti[sz]ing)\b"
    r"|^investigating\b|\bwould like to (investigate|discuss|schedule|review|look)\b|\blet'?s (arrange|schedule|set up)\b"
    r"|(happy|glad) to discuss|available (for a call|to discuss)|allow us to contact|sent via (a )?separate|\bhave been sent\b",
    re.I,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def classify_answer(answer: str) -> AnswerClass | Literal["residue"]:
    """Heuristic pass. Returns `residue` when no rule fires — send those to the LLM."""
    text = index_text(answer, placeholder="<X>")  # "call you at <tel_num>" is itself a handoff signal
    # Substance splits strong (an imperative command — concrete regardless of topic) from
    # soft (a suggestion or a specific-but-not-instructive fact). Validation found soft
    # substance losing to an explicit REQUEST sentence in the same message far more often
    # than it should win: "we recommend vigilant monitoring... could you provide details on
    # the vulnerability?" is a request the answer barely gestures at satisfying, not a fix.
    strong_substance = soft_substance = weak_substance = escalation = request = handoff = 0
    for sentence in (s for s in _SENTENCE_SPLIT.split(text) if s):
        # Checked first and made exclusive with substance below: "we recommend escalating
        # this to the technical team" matches SUBSTANCE_SUGGESTION's "we recommend" too —
        # the thing being recommended is a handoff, not a fix.
        is_escalation = bool(ESCALATION.search(sentence))
        if is_escalation:
            escalation += 1
        long_enough = len(sentence.split()) >= 6
        has_step = SUBSTANCE_STEP.search(sentence)
        has_suggestion = SUBSTANCE_SUGGESTION.search(sentence)
        fact_match = SUBSTANCE_FACT.search(sentence)
        has_fact = fact_match and _has_specificity(sentence)
        if not is_escalation and long_enough:
            if has_step:
                strong_substance += 1
            elif has_suggestion or has_fact:
                soft_substance += 1
        # A FACT match that fails the specificity check ("we offer several pricing tiers")
        # is too weak to call resolution on its own, but it is genuine evidence something
        # was actually said — not nothing. Tracked separately so it can block a confident
        # `escalation` call below rather than being silently discarded (validation found
        # exactly this pattern voting `escalation` on answers that had real, if vague, content).
        if not is_escalation and long_enough and fact_match and not has_fact:
            weak_substance += 1
        if REQUEST.search(sentence) and not CALL_TIME.search(sentence):
            request += 1
        elif CALL_TIME.search(sentence) or DEFERRAL.search(sentence):
            handoff += 1
    if strong_substance:
        return "resolution"
    if escalation:
        return "escalation"
    if request:
        return "clarification_request"
    if soft_substance:
        return "resolution"
    if handoff:
        return "residue" if weak_substance else "escalation"
    return "residue"


class ResidueClassification(BaseModel):
    answer_class: AnswerClass
    rationale: str


_RESIDUE_SYSTEM_PROMPT = """You classify one historical support ticket's answer into exactly
one of three classes, matching the same rule the heuristic pass uses:
- resolution: the answer takes a concrete step or states a concrete fact that could fix or
  explain the problem (a setting to change, a policy stated, a refund confirmed).
- clarification_request: the answer's main content is asking the customer for a missing
  fact needed to proceed (a version number, an error message, account details).
- escalation: the answer hands the case off — an explicit escalation, a promise to
  investigate and call back, or scheduling a call — with no fix or fact given yet.
This ticket reached you because the heuristic found no rule that matched. Pick the closest
class; do not invent a fourth category."""


def classify_residue(answer: str, fast_llm) -> ResidueClassification:
    """LLM pass over rows the heuristic left as `residue`. `fast_llm` is a chat model
    already bound with `.with_structured_output(ResidueClassification)` — injected so this
    module has no import-time dependency on autosupport.llm, and so tests can pass a fake."""
    if not index_text(answer):
        # No content to classify (an empty answer, or one that was only anonymisation
        # placeholders): the API rejects an empty user message, which crashed a full ingest.
        # Nothing was said, so nothing was fixed or asked — the non-resolution default for
        # anything the heuristic can't positively call a resolution.
        return ResidueClassification(answer_class="escalation", rationale="empty answer: no content to classify")
    return fast_llm.invoke([
        {"role": "system", "content": _RESIDUE_SYSTEM_PROMPT},
        {"role": "user", "content": answer},
    ])
