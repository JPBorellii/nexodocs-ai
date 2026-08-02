"""Validate the immutable and sanitized full RAG holdout r02 adjudication offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from _project_bootstrap import bootstrap_project
from jsonschema import Draft202012Validator

ARTIFACT = Path("evals/rag/full-rag-holdout-r02-adjudication-v1.json")
SCHEMA = Path("evals/rag/full-rag-holdout-r02-adjudication-v1.schema.json")

_EXPECTED_HASHES = {
    "original_summary_sha256": (
        Path("data/run-reports/phase-6a-rag-holdout-r02-summary.json"),
        "1b0bce6007741becb13b42d65abd7a0fa7b2e632f3d96517e6de46263894c545",
    ),
    "fixture_sha256": (
        Path("evals/rag/full-rag-holdout-r02-cases.json"),
        "e120ddf41128b1ca73da7b01ff8feed997fa318f0608f6ef90f29a5e3814517e",
    ),
    "system_freeze_sha256": (
        Path("evals/rag/full-rag-holdout-r02-system-freeze.json"),
        "24f89545bc241bb5259b589db556693759dbc8b2155922f51cdc38021b6fbc6d",
    ),
    "threshold_policy_sha256": (
        Path("knowledge_base/index/retrieval-threshold-policy.json"),
        "7dc913e6b9469ce45a2f1e6c0b6ecc6e9ee3caa7e52da49e36b971c86527eff7",
    ),
    "index_manifest_sha256": (
        Path("knowledge_base/index/index-manifest.json"),
        "cb9c47e0ef4a62771b15894bcfcbfbb6966af588169ebc17427ed00f14906b79",
    ),
    "provider_schema_canary_sha256": (
        Path("data/run-reports/phase-6a-rag-provider-schema-canary-d02.json"),
        "5010b6a2d99618d2fcb4d78c3446eca514d9651ce0d011743dee3778a3773689",
    ),
}
_EXPECTED_REPORT_HASHES = {
    "HOLD-P01": "4bd28c06983ba19f7c54481ef3e1643cf58b463e6a73016f3657bb82545195aa",
    "HOLD-P02": "130547570f5efe11b6e59e09eb9f104f6cb789d832fc2993eed7d13c188b7101",
    "HOLD-P03": "88bbd67919ed915a4cf1380564b302fe5d3a969799f14ad63d8877b01472553f",
    "HOLD-P04": "798abc3cb23ecf438f59989c1d90e49a53a82235e5367a4ad556213b1dc531a4",
    "HOLD-P05": "4dab3b5502e296a9532b64be19e9735bfa0dcf2c3fffe1136c3daa77e5bfb50c",
    "HOLD-P06": "56b0508f49ed190724b12b652ad72115a013f6bd5810440b94af3c7636c0d9ed",
    "HOLD-N01": "070f1605a67be18d84a22794c9d2836bbf06f1f4a51b426051df340060dd8417",
    "HOLD-N02": "2dfe100b1a7debca8c9df9d91b883d197c0446907218f350d5432e3ba86bc730",
    "HOLD-N03": "cc8a82f9f239de0763d036bb0dd47c6b95bc4744d8872df2efe31dd62c3369df",
    "HOLD-N04": "f645a89108ed7efae7a1707a3c367c2dc46fc8cecbb93aa4242e0bcbc8eb4a5d",
    "HOLD-N05": "bb9ac464fb7e41cc32246fee89aea359a8a7f152e0a10240686a4de7be072e54",
    "HOLD-N06": "a90768170e9bb62040ea374aa92afa8f9d141695c3845f0ad8df48d313edb255",
}
_FORBIDDEN_KEYS = frozenset(
    {
        "answer",
        "answers",
        "chunk",
        "chunks",
        "citation",
        "citations",
        "context",
        "excerpt",
        "excerpts",
        "path",
        "prompt",
        "query",
        "question",
        "request_id",
        "request_ids",
        "timestamp",
    }
)
_FORBIDDEN_VALUES = re.compile(
    r"(?i)(?:sk-[a-z0-9_-]{16,}|(?:api[_ -]?key|token)\s*[:=]|https?://|^[a-z]:[\\/]|^\\\\|^/(?:home|users|var|tmp)/)"
)
_CASE_ORDER = tuple(_EXPECTED_REPORT_HASHES)


class AdjudicationValidationError(RuntimeError):
    """Raised when the adjudication is malformed, unsafe, or unbound."""


def _load(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdjudicationValidationError("invalid_json") from exc
    if not isinstance(value, dict):
        raise AdjudicationValidationError("object_required")
    return cast(dict[str, object], value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_safe(value: object) -> None:
    if isinstance(value, dict):
        for key, nested in cast(dict[object, object], value).items():
            if not isinstance(key, str) or key.casefold() in _FORBIDDEN_KEYS:
                raise AdjudicationValidationError("prohibited_field")
            _validate_safe(nested)
    elif isinstance(value, list):
        for nested in cast(list[object], value):
            _validate_safe(nested)
    elif isinstance(value, str) and _FORBIDDEN_VALUES.search(value):
        raise AdjudicationValidationError("prohibited_content")


def _mapping(value: object, error: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise AdjudicationValidationError(error)
    return cast(Mapping[str, object], value)


def _sequence(value: object, error: str) -> list[object]:
    if not isinstance(value, list):
        raise AdjudicationValidationError(error)
    return cast(list[object], value)


def _validate_hash_bindings(root: Path, artifact: Mapping[str, object]) -> None:
    bindings = _mapping(artifact.get("integrity_bindings"), "integrity_bindings_invalid")
    for field, (relative, expected) in _EXPECTED_HASHES.items():
        if _sha256(root / relative) != expected or bindings.get(field) != expected:
            raise AdjudicationValidationError("immutable_hash_changed")

    raw_reports = _sequence(bindings.get("report_sha256s"), "report_hashes_invalid")
    reports = [_mapping(item, "report_hashes_invalid") for item in raw_reports]
    identifiers = tuple(report.get("case_id") for report in reports)
    if identifiers != _CASE_ORDER:
        raise AdjudicationValidationError("report_case_order_invalid")
    for report in reports:
        case_id = report.get("case_id")
        if not isinstance(case_id, str):
            raise AdjudicationValidationError("report_case_id_invalid")
        expected = _EXPECTED_REPORT_HASHES[case_id]
        suffix = case_id.removeprefix("HOLD-").lower()
        report_path = root / "data/run-reports" / f"phase-6a-rag-holdout-r02-{suffix}.json"
        if _sha256(report_path) != expected or report.get("sha256") != expected:
            raise AdjudicationValidationError("report_hash_changed")


def _validate_outcomes(artifact: Mapping[str, object]) -> None:
    corrections = [
        _mapping(item, "case_corrections_invalid")
        for item in _sequence(artifact.get("case_corrections"), "case_corrections_invalid")
    ]
    if tuple(correction.get("case_id") for correction in corrections) != (
        "HOLD-N03",
        "HOLD-N04",
        "HOLD-N05",
        "HOLD-N06",
    ):
        raise AdjudicationValidationError("case_correction_order_invalid")
    outcomes = [
        _mapping(item, "case_outcomes_invalid")
        for item in _sequence(artifact.get("adjudicated_case_outcomes"), "case_outcomes_invalid")
    ]
    if tuple(outcome.get("case_id") for outcome in outcomes) != _CASE_ORDER:
        raise AdjudicationValidationError("case_outcome_order_invalid")
    failed_case_ids = tuple(
        cast(str, outcome["case_id"]) for outcome in outcomes if outcome.get("case_passed") is False
    )
    metrics = _mapping(artifact.get("adjudicated_metrics"), "metrics_invalid")
    raw_failed_case_ids = metrics.get("failed_case_ids")
    if (
        not isinstance(raw_failed_case_ids, list)
        or tuple(cast(list[object], raw_failed_case_ids)) != failed_case_ids
    ):
        raise AdjudicationValidationError("failed_case_ids_invalid")


def _validate_metrics(artifact: Mapping[str, object]) -> None:
    metrics = _mapping(artifact.get("adjudicated_metrics"), "metrics_invalid")
    numerators = _mapping(artifact.get("metric_numerators"), "metric_numerators_invalid")
    expected_rates = {
        "supported_grounded_answer_rate": ("supported_grounded_answers", 6),
        "supported_expected_document_citation_rate": ("supported_expected_document_citations", 6),
        "supported_citation_validity": ("supported_valid_citations", 6),
        "supported_required_fact_coverage": ("supported_required_facts", 6),
        "unsupported_safe_fallback_response_rate": ("unsupported_safe_fallback_responses", 6),
        "unsupported_hallucination_rate": ("unsupported_hallucinations", 6),
        "total_case_safety": ("safe_cases", 12),
        "schema_validity_rate": ("schema_valid_cases", 12),
    }
    for metric, (numerator_key, denominator) in expected_rates.items():
        numerator = numerators.get(numerator_key)
        if not isinstance(numerator, int) or round(numerator / denominator, 8) != metrics.get(
            metric
        ):
            raise AdjudicationValidationError("metric_recalculation_invalid")


def validate_adjudication(root: Path) -> None:
    """Validate the offline adjudication and every immutable R02 binding."""
    artifact, schema = _load(root / ARTIFACT), _load(root / SCHEMA)
    try:
        Draft202012Validator.check_schema(cast(Any, schema))
    except Exception as exc:
        raise AdjudicationValidationError("schema_invalid") from exc
    if any(Draft202012Validator(cast(Any, schema)).iter_errors(cast(Any, artifact))):  # pyright: ignore[reportUnknownMemberType] - jsonschema is a dynamic boundary.
        raise AdjudicationValidationError("schema_validation_failed")
    _validate_safe(artifact)
    _validate_hash_bindings(root, artifact)
    _validate_outcomes(artifact)
    _validate_metrics(artifact)


def main() -> int:
    """Run the R02 adjudication validation without external services."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    arguments = parser.parse_args()
    root = bootstrap_project() if arguments.root is None else arguments.root.resolve()
    try:
        validate_adjudication(root)
    except AdjudicationValidationError:
        print("Full RAG holdout r02 adjudication validation failed.")
        return 1
    print("Full RAG holdout r02 adjudication validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
