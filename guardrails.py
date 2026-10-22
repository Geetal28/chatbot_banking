"""Rule-based input guardrails.

Rules run before any LLM call, so they are cheap, deterministic and testable.
Order matters: safety escalation first, then prompt injection, then advice refusal.

The canned replies deliberately contain no phone numbers or URLs: any contact detail
shown to a customer must come from the bank, not from this demo.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class GuardResult:
    action: str  # "allow" | "escalate" | "injection" | "refuse_advice"
    reason: str = ""


_ESCALATE = [
    r"\bfraud(ulent)?\b",
    r"\bscam(med)?\b",
    r"\bphishing\b",
    r"\bstolen\b",
    r"\b(lost|missing)\b.*\bcard\b",
    r"\bunauthori[sz]ed\b",
    r"\b(didn'?t|did not) (make|authori[sz]e)\b",
    r"\bdispute\b",
    r"\bchargeback\b",
    r"\bhacked\b",
    r"\bcompromised\b",
    r"\bcomplain(t|ing)?\b",
    r"\bbereave(d|ment)\b|\bpassed away\b|\bdeath of\b",
    r"\b(can'?t|cannot|struggling to) (afford|pay)\b",
    r"\bfinancial difficult(y|ies)\b",
    r"\bin debt\b|\bdebt problems?\b",
]

_INJECTION = [
    r"ignore (all |any |the )?(previous|prior|above) (instructions|rules|prompts?)",
    r"disregard (all |any |the )?(previous|prior|above)",
    r"(reveal|show|print|repeat) (your |the )?(system|hidden|initial) (prompt|instructions)",
    r"\byou are now\b",
    r"\bact as (an? )?(unrestricted|jailbroken|dan)\b",
    r"\bdeveloper mode\b",
]

_ADVICE = [
    r"\bshould i (invest|buy|sell)\b",
    r"\bwhich (stocks?|shares?|funds?|crypto\w*|etfs?) (should|to)\b",
    r"\bbest (stocks?|shares?|funds?|investments?|crypto\w*) to (buy|invest)\b",
    r"\b(invest|put) (my|the) (savings|money|pension)\b",
    r"\bis (it|now) a good (time|idea) to (invest|buy|sell)\b",
    r"\bshould i (take out|get|switch to|remortgage)\b",
    r"\bwhich (mortgage|loan|pension|isa) (is|would be) (best|right) for me\b",
    r"\btax (advice|planning)\b",
]

_ESC_RE = [re.compile(p, re.IGNORECASE) for p in _ESCALATE]
_INJ_RE = [re.compile(p, re.IGNORECASE) for p in _INJECTION]
_ADV_RE = [re.compile(p, re.IGNORECASE) for p in _ADVICE]


def check_input(text: str) -> GuardResult:
    for rx in _ESC_RE:
        if rx.search(text):
            return GuardResult("escalate", rx.pattern)
    for rx in _INJ_RE:
        if rx.search(text):
            return GuardResult("injection", rx.pattern)
    for rx in _ADV_RE:
        if rx.search(text):
            return GuardResult("refuse_advice", rx.pattern)
    return GuardResult("allow")


CANNED = {
    "escalate": (
        "This sounds like something that needs a person, not a chatbot. "
        "Please contact your bank directly through its official channels (for a lost or "
        "stolen card or suspected fraud, use the number on the back of your card or in "
        "the bank's official app, and do it now). I can't investigate or resolve this "
        "here, and I will never ask you for your PIN, full card number or passwords."
    ),
    "injection": (
        "I can't follow instructions that try to change how I work. "
        "I can help with questions about the bank's fees, products and policies, "
        "or about the demo account data."
    ),
    "refuse_advice": (
        "I can't give personal financial, investment or tax advice. "
        "I can explain what the bank's published documents say about a product, such as "
        "its fees or terms, so you can compare options yourself, or point you to "
        "speaking with a qualified adviser."
    ),
    "out_of_scope": (
        "I can only help with questions about the bank's published products, fees and "
        "policies, and about the demo account data. Could you rephrase your question "
        "along those lines?"
    ),
    "not_found": (
        "I couldn't find this in the documents I have, so I won't guess. "
        "Please check the bank's website or contact the bank directly."
    ),
}
