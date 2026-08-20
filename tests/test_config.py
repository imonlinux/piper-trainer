"""Tests for config.voice_stem — the one shared naming convention."""
from piper_trainer.config import voice_stem


def test_voice_stem_convention():
    assert voice_stem("marvin", "medium", "en-gb") == "marvin-medium"
    assert voice_stem("marvin", "low") == "marvin-low"


def test_voice_stem_ignores_language():
    """No language prefix: the language lives in the config's language block,
    and the stem matches the deployed .onnx filename."""
    assert voice_stem("m", "high", "en-us") == voice_stem("m", "high", "de")
