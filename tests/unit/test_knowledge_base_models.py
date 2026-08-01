"""Tests for canonical knowledge-base parsing helpers."""

from __future__ import annotations

import pytest

from nexodocs_ai.knowledge_base.models import KnowledgeBaseError, required_string


def test_required_string_rejects_missing_value() -> None:
    """Required metadata cannot be blank."""
    with pytest.raises(KnowledgeBaseError):
        required_string({}, "document_id")
