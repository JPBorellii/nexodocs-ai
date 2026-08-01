"""Tests for explicit knowledge-base safety rules."""

from __future__ import annotations

import pytest

from nexodocs_ai.knowledge_base.models import KnowledgeBaseError
from nexodocs_ai.knowledge_base.validator import assert_safe_text


def test_prohibited_data_pattern_is_rejected() -> None:
    """A CPF-shaped value is not accepted in documentary content."""
    with pytest.raises(KnowledgeBaseError):
        assert_safe_text("CPF 123.456.789-00")
