"""Tests for conservative normalization."""

from nexodocs_ai.processing.normalizer import normalize_text


def test_normalization_preserves_accents_and_meaningful_structure() -> None:
    """Only conservative whitespace changes are made."""
    assert normalize_text(" Título  á  \r\n\r\n\r\n- item  ") == "Título á\n\n- item"
