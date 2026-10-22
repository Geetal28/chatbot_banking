"""PII masking. Runs on user text before it reaches the LLM, the logs or the vector store.

Regex-based, so it is a first line of defence and not a guarantee. For production you
would add a trained detector (e.g. Presidio) and test against real-world formats.
"""
from __future__ import annotations

import re

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_IBAN_GB = re.compile(r"\bGB\d{2}\s?[A-Z]{4}(?:\s?\d{4}){3}\s?\d{2}\b", re.IGNORECASE)
_NINO = re.compile(r"\b[A-CEGHJ-PR-TW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b", re.IGNORECASE)
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_SORT_CODE = re.compile(r"(?<!\d)\d{2}-\d{2}-\d{2}(?!\d)")
_PHONE = re.compile(r"(?<!\d)(?:\+44\s?|0)\d{3,4}[\s-]?\d{3}[\s-]?\d{3,4}(?!\d)")
_ACCOUNT_NO = re.compile(r"(?<!\d)\d{8}(?!\d)")


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def mask_pii(text: str) -> tuple[str, list[str]]:
    """Return (masked_text, list of PII types found)."""
    found: list[str] = []

    def sub(pattern: re.Pattern[str], token: str, s: str, check=None) -> str:
        def repl(m: re.Match[str]) -> str:
            raw = m.group(0)
            if check is not None and not check(raw):
                return raw
            found.append(token)
            return f"[{token}]"

        return pattern.sub(repl, s)

    out = text
    out = sub(_EMAIL, "EMAIL", out)
    out = sub(_IBAN_GB, "IBAN", out)
    out = sub(_NINO, "NI_NUMBER", out)
    out = sub(_CARD, "CARD", out, check=lambda r: _luhn_ok(re.sub(r"\D", "", r)))
    out = sub(_SORT_CODE, "SORT_CODE", out)
    out = sub(_PHONE, "PHONE", out)
    out = sub(_ACCOUNT_NO, "ACCOUNT_NO", out)
    return out, found
