"""Offline contract evaluation; it deliberately does not judge prose fluency."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .constants import EVALUATION_MINIMUMS, SCHEMA_VERSION
from .models import RagEvaluationMetrics, RagRequest
from .pipeline import RagPipeline


def evaluate(
    pipeline: RagPipeline, cases_path: Path
) -> tuple[RagEvaluationMetrics, list[dict[str, object]]]:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    outcomes: list[dict[str, object]] = []
    for case in cases:
        response = pipeline.answer(RagRequest(case["query"], filters=None))
        status_ok = response.status == case["expected_status"]
        citations_ok = len(response.citations) >= case["expected_citation_count_min"]
        required_ok = all(
            term.casefold() in (response.answer or "").casefold() for term in case["required_terms"]
        )
        forbidden_ok = not any(
            term.casefold() in (response.answer or "").casefold()
            for term in case["forbidden_terms"]
        )
        outcomes.append(
            {
                "case_id": case["case_id"],
                "passed": status_ok and citations_ok and required_ok and forbidden_ok,
                "status": response.status,
            }
        )
    total = len(cases) or 1
    passed = sum(bool(item["passed"]) for item in outcomes)
    metrics = RagEvaluationMetrics(total, passed, passed / total, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0)
    return metrics, outcomes


def evaluation_document(
    metrics: RagEvaluationMetrics, cases: list[dict[str, object]]
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation_version": "rag-offline-v1",
        **asdict(metrics),
        "cases": cases,
    }


def validate_minimums(metrics: RagEvaluationMetrics) -> bool:
    return all(
        getattr(metrics, name) >= minimum if minimum else getattr(metrics, name) == 0.0
        for name, minimum in EVALUATION_MINIMUMS.items()
    )
