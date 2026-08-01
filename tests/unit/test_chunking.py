"""Tests for deterministic chunk construction."""

from nexodocs_ai.processing.chunker import split_text


def test_long_text_is_split_without_exceeding_maximum() -> None:
    """Word boundaries are retained for oversized content."""
    chunks = split_text(" ".join(["palavra"] * 400))
    assert len(chunks) > 1
    assert all(len(chunk) <= 1200 for chunk in chunks)
    assert all(chunk.split() for chunk in chunks)
