"""Tests for retrieval configuration and filter contracts."""

from __future__ import annotations

import pytest

from nexodocs_ai.retrieval.config import load_config
from nexodocs_ai.retrieval.models import ConfigurationError, RetrievalError, RetrievalFilters


def _valid_test_values() -> dict[str, str]:
    return {
        "APP_ENV": "test",
        "EMBEDDING_PROVIDER": "fake",
        "OPENAI_API_KEY": "openai-secret-value",
        "OPENAI_EMBEDDING_DIMENSIONS": "64",
        "QDRANT_MODE": "memory",
        "QDRANT_API_KEY": "qdrant-secret-value",
        "QDRANT_TIMEOUT_SECONDS": "10",
        "RETRIEVAL_TOP_K": "5",
        "RETRIEVAL_MAX_TOP_K": "20",
        "RETRIEVAL_MAX_PER_DOCUMENT": "2",
        "RETRIEVAL_SCORE_THRESHOLD": "",
    }


def test_load_config_parses_safe_test_configuration() -> None:
    """Typed values are parsed without exposing credentials."""
    config = load_config(_valid_test_values())

    assert config.embedding_dimensions == 64
    assert config.qdrant_timeout_seconds == 10
    assert isinstance(config.qdrant_timeout_seconds, int)
    assert config.score_threshold is None
    assert config.safe_summary() == {
        "app_env": "test",
        "embedding_provider": "fake",
        "embedding_model": "text-embedding-3-small",
        "embedding_dimensions": 64,
        "qdrant_mode": "memory",
        "collection_name": "nexodocs_chunks_v1",
    }
    representation = repr(config)
    assert "openai-secret-value" not in representation
    assert "qdrant-secret-value" not in representation


def test_load_config_parses_optional_threshold() -> None:
    values = _valid_test_values()
    values["RETRIEVAL_SCORE_THRESHOLD"] = "0.42"

    assert load_config(values).score_threshold == pytest.approx(0.42)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"APP_ENV": "development"}, "fake"),
        (
            {
                "APP_ENV": "development",
                "EMBEDDING_PROVIDER": "openai",
                "QDRANT_MODE": "memory",
            },
            "memory",
        ),
        ({"QDRANT_TIMEOUT_SECONDS": "1.5"}, "inteiro"),
        ({"QDRANT_TIMEOUT_SECONDS": "0"}, "positivo"),
        ({"QDRANT_MODE": "remote", "QDRANT_URL": ""}, "QDRANT_URL"),
        ({"RETRIEVAL_TOP_K": "21"}, "RETRIEVAL_TOP_K"),
        ({"EMBEDDING_PROVIDER": "unknown"}, "inv\u00e1lido"),
    ],
)
def test_load_config_rejects_unsafe_or_invalid_values(
    updates: dict[str, str], message: str
) -> None:
    values = _valid_test_values()
    values.update(updates)

    with pytest.raises(ConfigurationError, match=message):
        load_config(values)


def test_retrieval_filters_emit_only_present_allowed_fields() -> None:
    filters = RetrievalFilters(
        document_id="NSI-POL-OPS-001",
        source_format="pdf",
        classification="internal",
    )

    assert filters.as_dict() == {
        "document_id": "NSI-POL-OPS-001",
        "source_format": "pdf",
        "classification": "internal",
    }


@pytest.mark.parametrize("value", ["", " ", "\t"])
def test_retrieval_filters_reject_blank_values(value: str) -> None:
    with pytest.raises(RetrievalError, match="blank"):
        RetrievalFilters(category=value).as_dict()
