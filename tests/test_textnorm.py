"""Spoken-form normalization: digits and titles out, spoken words in.

The boundary rules matter as much as the expansions — times, versions,
hardware names, ordinals and money/percent must survive untouched, because
espeak-ng expands those acceptably and a wrong guess poisons a clip.
"""
from __future__ import annotations

import pytest

from piper_trainer import textnorm


@pytest.fixture(scope="module")
def engine():
    return textnorm.engine()


# ------------------------------------------------------------------ numbers

def test_integers_expand_to_spoken_words(engine):
    out, n = textnorm.expand_numbers("we have 9000 credits and 40 men",
                                     engine=engine)
    assert out == "we have nine thousand credits and forty men"
    assert n == 2


def test_thousands_commas_and_decimals(engine):
    out, n = textnorm.expand_numbers("1,000,000 reasons, about 3.14 of them",
                                     engine=engine)
    # the number opens the sentence, so it is capitalized
    assert "One million reasons" in out
    assert "three point one four of them" in out
    assert n == 2


def test_sentence_start_capitalized(engine):
    out, _ = textnorm.expand_numbers("9000 was the figure.", engine=engine)
    assert out == "Nine thousand was the figure."


def test_non_prose_numbers_untouched(engine):
    for text in ("meeting at 12:30 sharp",
                 "running v2.1 today",
                 "the HAL-9000 pod bay doors",
                 "her 1st attempt",
                 "that costs $50 now",
                 "about 50% done"):
        out, n = textnorm.expand_numbers(text, engine=engine)
        assert out == text, text
        assert n == 0


# ------------------------------------------------------------ abbreviations

def test_titles_expand_with_case_preserved():
    out, n = textnorm.expand_abbreviations("Mr. Smith thanked dr. Jones")
    assert out == "Mister Smith thanked doctor Jones"
    assert n == 2


def test_abbreviation_without_period_untouched():
    # bare "prof"/"col" can be ordinary words; the period is required
    out, n = textnorm.expand_abbreviations("ask Prof when Col arrives")
    assert out == "ask Prof when Col arrives"
    assert n == 0


def test_written_out_latin_forms():
    out, _ = textnorm.expand_abbreviations("e.g. this vs. that")
    assert out == "for example this versus that"


def test_ambiguous_or_plain_words_untouched():
    for text in ("St. Patrick's cathedral",  # Saint vs Street: undecidable
                 "the museum has van goghs",
                 "doctor who is a title too"):  # no period, not capitalized
        out, _ = textnorm.expand_abbreviations(text)
        assert out == text, text


# ----------------------------------------------------------------- combined

def test_normalize_applies_both_passes(engine):
    text = "Mr. Scott logged 1,800 crew and 9000 pods, i.e. all of them."
    out, counts = textnorm.normalize(text, engine=engine)
    assert "Mister Scott" in out
    # inflect's spoken form for 1,800 (the comma is a spoken pause, fine for TTS)
    assert "one thousand, eight hundred crew" in out
    assert "nine thousand pods" in out
    assert "that is all of them" in out
    assert counts["abbreviations"] == 2  # Mr. + i.e.
    assert counts["numbers"] == 2


def test_normalize_without_inflect_still_expands_titles(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_inflect(name, *a, **k):
        if name == "inflect":
            raise ImportError("blocked for test")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_inflect)
    assert textnorm.engine() is None
    out, counts = textnorm.normalize("Mr. Smith had 40 days")
    assert out == "Mister Smith had 40 days"  # numbers need inflect
    assert counts == {"abbreviations": 1, "numbers": 0}
