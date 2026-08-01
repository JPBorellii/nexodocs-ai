"""Tests for generation data helpers."""

from __future__ import annotations

from nexodocs_ai.knowledge_base.generator import load_canonical, repository_root


def test_loads_the_five_canonical_documents() -> None:
    """Canonical source has the mandated five documents."""
    assert len(load_canonical(repository_root())) == 5
