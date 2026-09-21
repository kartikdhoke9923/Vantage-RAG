"""Output-layer PII / secret masking.

Applied to the *final answer* only, and only when MASK_PII_IN_OUTPUT=true
(off by default — the résumé contact details are intentionally part of the
corpus). Patterns target obvious personal contact data and secret-shaped
tokens; they are deliberately conservative so legitimate doc content
(hashes, dates, addresses) is not corrupted.
"""
import re

# Type → regex. Deterministic masking order: secret → email → phone, so a
# longer secret token is never re-masked as a phone-like run.
PATTERNS: dict[str, re.Pattern] = {
    "secret": re.compile(
        r"\b(?:sk-[A-Za-z0-9_-]{8,}|gsk_[A-Za-z0-9_-]{8,}|jina_[A-Za-z0-9_-]{8,}"
        r"|ck-[A-Za-z0-9_-]{8,}|npg_[A-Za-z0-9_]{8,}|pylf_[A-Za-z0-9_]{8,}"
        r"|pc-[A-Za-z0-9_-]{8,}|Bearer\s+[A-Za-z0-9._-]{12,})\b"
    ),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    # Phone-like runs of digits (incl. +country code, spaces/dots/dashes).
    "phone": re.compile(r"\+?\d[\d\s().-]{7,}\d"),
}

_ORDER = ("secret", "email", "phone")

PLACEHOLDERS = {
    "email": "[email masked]",
    "phone": "[phone masked]",
    "secret": "[secret masked]",
}


def _looks_like_date(candidate: str) -> bool:
    """Reject YYYY-MM-DD / DD-MM-YYYY / YYYY/MM/DD style strings as phones."""
    digits = re.sub(r"[^0-9]", "", candidate)
    if len(digits) != 8:
        return False
    for sep in ("-", "/"):
        if sep in candidate:
            parts = candidate.split(sep)
            if len(parts) == 3 and all(p.isdigit() for p in parts):
                year, month, day = (parts if len(parts[0]) == 4 else parts[::-1])
                return 1900 <= int(year) <= 2100 and 1 <= int(month) <= 12 and 1 <= int(day) <= 31
    return False


def mask_pii(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Replace PII/secret-shaped substrings in `text`.

    Returns (masked_text, findings) where findings is a list of
    (pattern_name, placeholder) for logging.
    """
    masked = text
    findings: list[tuple[str, str]] = []

    for kind in _ORDER:
        pattern = PATTERNS[kind]
        matches = list(pattern.finditer(masked))
        if not matches:
            continue

        def _repl(m: re.Match, kind=kind) -> str:
            if kind == "phone" and _looks_like_date(m.group(0)):
                return m.group(0)
            findings.append((kind, PLACEHOLDERS[kind]))
            return PLACEHOLDERS[kind]

        masked = pattern.sub(_repl, masked)

    return masked, findings