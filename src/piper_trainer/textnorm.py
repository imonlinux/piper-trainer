"""Spoken-form text normalization for transcripts (§1: the transcript must
read the way the clip actually sounds).

Whisper writes how text is usually WRITTEN: digits ("9000") and standard
abbreviations ("Mr."). A TTS transcript should read how the clip SOUNDS
("nine thousand", "Mister") — otherwise every clip needs a hand correction
in the audit page before it teaches the model anything consistent.

Two passes, both deliberately conservative:
- abbreviations: a small fixed map of unambiguous titles only. "St." stays
  put (Saint vs Street is not decidable from text), everything here is.
- numbers: integers and decimals in word-boundary-ish isolation. Digits
  glued to letters/symbols — times ("12:30"), versions ("v2.1"), hardware
  names ("HAL-9000"), ordinals ("1st"), money/percent ("$50", "50%") — are
  left alone: espeak-ng expands those forms acceptably at training time,
  and a wrong guess is worse than no guess.

inflect is an optional dependency (runtime extra). Without it the numbers
pass is skipped; the abbreviation pass is pure stdlib and always runs.
"""
from __future__ import annotations

import re

# Unambiguous, spoken-form titles. Matched case-insensitively and only WITH
# the trailing period — bare "ms" or "gen" can be ordinary words, and a
# false expansion poisons a clip. The replacement follows the match's case.
_ABBREVIATIONS = {
    "mr": "mister",
    "mrs": "misses",
    "ms": "miss",
    "dr": "doctor",
    "prof": "professor",
    "capt": "captain",
    "col": "colonel",
    "gen": "general",
    "lt": "lieutenant",
    "sgt": "sergeant",
    "rev": "reverend",
    "hon": "honorable",
    "jr": "junior",
    "sr": "senior",
    "etc": "et cetera",
    "vs": "versus",
}
_ABBR_RE = re.compile(
    r"\b(" + "|".join(sorted(_ABBREVIATIONS, key=len, reverse=True))
    + r")\.(?=\s|$|[,.;:!?])",
    re.IGNORECASE)
# Written-out forms with internal dots need their own literals.
_LATIN_RE = re.compile(r"\be\.\s?g\.?(?=\s)", re.IGNORECASE)
_ID_EST_RE = re.compile(r"\bi\.\s?e\.?(?=\s)", re.IGNORECASE)

# Integers (with optional thousands commas) and decimals, standing clear of
# letters, digits, and the punctuation that marks non-prose numbers.
_NUMBER_RE = re.compile(
    r"(?<![\w.:\-$%])"
    r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?"
    r"(?![\w.:\-$%])")
_DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven",
           "eight", "nine"]


def _inflect_engine():
    import inflect
    return inflect.engine()


def engine():
    """Shared inflect engine for batch use; None when inflect is absent."""
    try:
        return _inflect_engine()
    except ImportError:
        return None


def expand_numbers(text: str, engine=None) -> tuple[str, int]:
    """Digits to spoken words. Returns (text, count of tokens expanded)."""
    if not _NUMBER_RE.search(text):
        return text, 0
    if engine is None:
        engine = _inflect_engine()  # ImportError propagates: caller opted in

    def sub(m: re.Match) -> str:
        whole, frac = m.group(1), m.group(2)
        try:
            words = engine.number_to_words(int(whole.replace(",", "")),
                                           andword="")
        except Exception:
            return m.group(0)
        if frac:
            words += " point " + " ".join(_DIGITS[int(d)] for d in frac)
        # a digit that opens a sentence should not start it lowercase
        prefix = text[:m.start()]
        if not prefix or prefix.endswith((". ", "! ", "? ")):
            words = words.capitalize()
        return words

    out, n = _NUMBER_RE.subn(sub, text)
    return out, n


def expand_abbreviations(text: str) -> tuple[str, int]:
    """Abbreviations to the words actually spoken. Returns (text, count)."""
    def sub(m: re.Match) -> str:
        word = _ABBREVIATIONS[m.group(1).lower()]
        return word.capitalize() if m.group(1)[:1].isupper() else word

    out, n = _LATIN_RE.subn("for example", text)
    out, k = _ID_EST_RE.subn("that is", out)
    n += k
    out, n_title = _ABBR_RE.subn(sub, out)
    return out, n + n_title


def normalize(text: str, engine=None) -> tuple[str, dict]:
    """Full spoken-form pass: abbreviations, then numbers. Never raises —
    if inflect is absent (engine=None and the import fails) the numbers
    pass is skipped and digits stay as written. Returns
    (normalized_text, {"abbreviations": n, "numbers": m})."""
    text, n_abbr = expand_abbreviations(text)
    try:
        text, n_num = expand_numbers(text, engine=engine)
    except ImportError:
        n_num = 0
    return text, {"abbreviations": n_abbr, "numbers": n_num}
