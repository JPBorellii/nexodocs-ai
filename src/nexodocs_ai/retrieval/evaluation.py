"""Deterministic offline retrieval evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from .models import EvaluationCase, EvaluationMetrics, RetrievalError, RetrievalFilters
from .retriever import Retriever


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RetrievalError(f"Evaluation field must be a non-empty string: {field}")
    return value


def load_evaluation_cases(path: Path) -> tuple[EvaluationCase, ...]:
    """Load and validate the closed offline evaluation JSON boundary."""
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise RetrievalError("Evaluation cases must be a non-empty array")
    cases: list[EvaluationCase] = []
    for raw_case in cast(list[object], raw):
        if not isinstance(raw_case, dict):
            raise RetrievalError("Each evaluation case must be an object with string keys")
        raw_mapping = cast(dict[object, object], raw_case)
        if not all(isinstance(key, str) for key in raw_mapping):
            raise RetrievalError("Each evaluation case must be an object with string keys")
        item = cast(dict[str, object], raw_mapping)
        raw_expected = item.get("expected_document_ids")
        if not isinstance(raw_expected, list):
            raise RetrievalError("expected_document_ids must contain strings")
        expected_values = cast(list[object], raw_expected)
        if not all(isinstance(value, str) and value for value in expected_values):
            raise RetrievalError("expected_document_ids must contain strings")
        raw_filters = item.get("optional_filters", {})
        if not isinstance(raw_filters, dict):
            raise RetrievalError("optional_filters must map strings to strings")
        filter_values = cast(dict[object, object], raw_filters)
        if not all(
            isinstance(key, str) and isinstance(value, str) for key, value in filter_values.items()
        ):
            raise RetrievalError("optional_filters must map strings to strings")
        filters = cast(dict[str, str], filter_values)
        top_k = item.get("top_k")
        threshold = item.get("fake_score_threshold")
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise RetrievalError("top_k must be an integer")
        if threshold is not None and not isinstance(threshold, int | float):
            raise RetrievalError("fake_score_threshold must be numeric")
        expected_status = _string(item.get("expected_status"), "expected_status")
        if expected_status not in {"found", "no_evidence"}:
            raise RetrievalError("Invalid expected_status")
        cases.append(
            EvaluationCase(
                _string(item.get("case_id"), "case_id"),
                _string(item.get("query"), "query"),
                expected_status,
                tuple(cast(list[str], expected_values)),
                RetrievalFilters(
                    document_id=filters.get("document_id"),
                    category=filters.get("category"),
                    source_format=filters.get("source_format"),
                    owner_area=filters.get("owner_area"),
                    version=filters.get("version"),
                    classification=filters.get("classification"),
                ),
                top_k,
                None if threshold is None else float(threshold),
            )
        )
    if len({case.case_id for case in cases}) != len(cases):
        raise RetrievalError("Evaluation case IDs must be unique")
    return tuple(cases)


def evaluate(retriever: Retriever, cases_path: Path) -> EvaluationMetrics:
    """Measure lexical-fake retrieval; this is not real semantic evaluation."""
    cases = load_evaluation_cases(cases_path)
    hits = recalls = reciprocal = fallback_correct = fallback_total = passed = 0
    answerable = 0
    for case in cases:
        response = retriever.retrieve(
            case.query,
            case.top_k,
            case.optional_filters,
            case.fake_score_threshold,
        )
        expected = set(case.expected_document_ids)
        actual = {str(result.metadata["document_id"]) for result in response.results}
        if case.expected_status == "no_evidence":
            fallback_total += 1
            correct = response.status == "no_evidence"
            fallback_correct += correct
            passed += correct
            continue
        answerable += 1
        matches = [
            index
            for index, result in enumerate(response.results, 1)
            if result.metadata["document_id"] in expected
        ]
        hits += bool(matches)
        recalls += len(actual & expected) / len(expected)
        reciprocal += 1 / matches[0] if matches else 0
        passed += response.status == "found" and bool(matches)
    return EvaluationMetrics(
        len(cases),
        passed,
        hits / max(answerable, 1),
        recalls / max(answerable, 1),
        reciprocal / max(answerable, 1),
        fallback_correct / max(fallback_total, 1),
    )
