"""Tests for the validated offline evaluation boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nexodocs_ai.retrieval.config import load_config
from nexodocs_ai.retrieval.evaluation import load_evaluation_cases
from nexodocs_ai.retrieval.models import ConfigurationError, RetrievalError


def _valid_case(case_id: str = "valid-case") -> dict[str, object]:
    return {
        "case_id": case_id,
        "query": "cancelamento fict\u00edcio",
        "expected_status": "found",
        "expected_document_ids": ["NSI-POL-OPS-001"],
        "optional_filters": {"document_id": "NSI-POL-OPS-001"},
        "top_k": 5,
        "fake_score_threshold": 0.25,
    }


def _write_cases(tmp_path: Path, value: object) -> Path:
    path = tmp_path / "retrieval_cases.json"
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


INVALID_CASES: list[tuple[object, str]] = [
    ([], "non-empty array"),
    ({}, "non-empty array"),
    ([None], "object"),
    ([{**_valid_case(), "expected_document_ids": "DOC-001"}], "contain strings"),
    ([{**_valid_case(), "expected_document_ids": [1]}], "contain strings"),
    ([{**_valid_case(), "optional_filters": []}], "map strings"),
    ([{**_valid_case(), "optional_filters": {"category": 1}}], "map strings"),
    ([{**_valid_case(), "top_k": True}], "integer"),
    ([{**_valid_case(), "fake_score_threshold": "high"}], "numeric"),
    ([{**_valid_case(), "expected_status": "maybe"}], "expected_status"),
    ([{**_valid_case(), "query": " "}], "non-empty string"),
]


def test_load_evaluation_cases_parses_typed_filters_and_threshold(tmp_path: Path) -> None:
    path = _write_cases(tmp_path, [_valid_case()])

    cases = load_evaluation_cases(path)

    assert len(cases) == 1
    assert cases[0].case_id == "valid-case"
    assert cases[0].expected_document_ids == ("NSI-POL-OPS-001",)
    assert cases[0].optional_filters.as_dict() == {"document_id": "NSI-POL-OPS-001"}
    assert cases[0].fake_score_threshold == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("raw", "message"),
    INVALID_CASES,
)
def test_load_evaluation_cases_rejects_invalid_shapes(
    tmp_path: Path, raw: object, message: str
) -> None:
    with pytest.raises(RetrievalError, match=message):
        load_evaluation_cases(_write_cases(tmp_path, raw))


def test_load_evaluation_cases_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    with pytest.raises(RetrievalError, match="unique"):
        load_evaluation_cases(
            _write_cases(tmp_path, [_valid_case("duplicate"), _valid_case("duplicate")])
        )


def test_config_rejects_non_numeric_score_threshold() -> None:
    values = {
        "APP_ENV": "test",
        "EMBEDDING_PROVIDER": "fake",
        "OPENAI_EMBEDDING_DIMENSIONS": "192",
        "QDRANT_MODE": "memory",
        "RETRIEVAL_SCORE_THRESHOLD": "not-a-score",
    }

    with pytest.raises(ConfigurationError, match="num"):
        load_config(values)
