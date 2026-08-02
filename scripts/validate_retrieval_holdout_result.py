"""Validate the sanitized retrieval holdout result without runtime services."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, cast

from jsonschema import Draft202012Validator


class HoldoutValidationError(RuntimeError):
    """Raised when the versioned holdout evidence is inconsistent or unsafe."""


ARTIFACT = Path("evals/retrieval/holdout-r01-result.json")
SCHEMA = Path("evals/retrieval/holdout-result.schema.json")
FORBIDDEN_FIELD_PARTS = frozenset(
    {
        "answer",
        "chunk",
        "context",
        "credential",
        "header",
        "hostname",
        "key",
        "password",
        "payload",
        "path",
        "prompt",
        "query",
        "question",
        "request_id",
        "secret",
        "text",
        "timestamp",
        "url",
        "username",
        "vector",
    }
)
SAFE_FIELD_NAMES = frozenset(
    {
        "threshold_policy_sha256",
        "source_summary_sha256",
        "index_manifest_sha256",
        "chunks_sha256",
        "processed_manifest_sha256",
        "index_plan_sha256",
        "prompt_tokens",
    }
)
ABSOLUTE_PATH = re.compile(r"(?:^[A-Za-z]:[\\/]|^/|^\\\\)")
RATE_PRECISION = Decimal("0.00000001")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HoldoutValidationError("invalid_json") from exc
    if not isinstance(value, dict):
        raise HoldoutValidationError("object_required")
    return cast(dict[str, object], value)


def _validate_schema(schema: dict[str, object], artifact: dict[str, object]) -> None:
    Draft202012Validator.check_schema(cast(Any, schema))
    errors = sorted(
        Draft202012Validator(cast(Any, schema)).iter_errors(cast(Any, artifact)),  # pyright: ignore[reportUnknownMemberType] - jsonschema is a dynamic boundary.
        key=lambda item: list(item.path),
    )
    if errors:
        raise HoldoutValidationError("schema_invalid")


def _validate_safe_content(value: object) -> None:
    if isinstance(value, dict):
        for key, child in cast(dict[str, object], value).items():
            lowered = key.lower()
            if key not in SAFE_FIELD_NAMES and (
                any(part in lowered for part in FORBIDDEN_FIELD_PARTS) or lowered.endswith("_at")
            ):
                raise HoldoutValidationError("prohibited_field")
            _validate_safe_content(child)
    elif isinstance(value, list):
        for child in cast(list[object], value):
            _validate_safe_content(child)
    elif isinstance(value, str) and ("://" in value or ABSOLUTE_PATH.search(value)):
        raise HoldoutValidationError("prohibited_location")


def _decimal(value: object) -> Decimal:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise HoldoutValidationError("number_required")
    return Decimal(str(value))


def _outcomes(artifact: dict[str, object]) -> list[dict[str, object]]:
    value = artifact.get("case_outcomes")
    if not isinstance(value, list):
        raise HoldoutValidationError("outcomes_required")
    outcomes: list[dict[str, object]] = []
    for item in cast(list[object], value):
        if not isinstance(item, dict):
            raise HoldoutValidationError("outcome_object_required")
        outcomes.append(cast(dict[str, object], item))
    return outcomes


def _rounded(value: Decimal) -> Decimal:
    return value.quantize(RATE_PRECISION, rounding=ROUND_HALF_UP)


def _expect(artifact: dict[str, object], key: str, expected: Decimal) -> None:
    if _decimal(artifact.get(key)) != expected:
        raise HoldoutValidationError("metric_mismatch")


def _validate_case_math(artifact: dict[str, object]) -> None:
    outcomes = _outcomes(artifact)
    identifiers = [item.get("case_id") for item in outcomes]
    expected_identifiers = [
        "HOLD-P01",
        "HOLD-P02",
        "HOLD-P03",
        "HOLD-P04",
        "HOLD-P05",
        "HOLD-P06",
        "HOLD-N01",
        "HOLD-N02",
        "HOLD-N03",
        "HOLD-N04",
        "HOLD-N05",
        "HOLD-N06",
    ]
    if not all(isinstance(identifier, str) for identifier in identifiers):
        raise HoldoutValidationError("case_identifiers_invalid")
    if set(cast(list[str], identifiers)) != set(expected_identifiers):
        raise HoldoutValidationError("case_identifiers_invalid")

    supported = [item for item in outcomes if item.get("kind") == "supported"]
    unsupported = [item for item in outcomes if item.get("kind") != "supported"]
    if len(supported) != 6 or len(unsupported) != 6:
        raise HoldoutValidationError("case_counts_invalid")
    if any(item.get("expected_status") != "found" for item in supported):
        raise HoldoutValidationError("supported_status_invalid")
    if any(item.get("expected_status") != "no_evidence" for item in unsupported):
        raise HoldoutValidationError("unsupported_status_invalid")

    ranks: list[int] = []
    expected_scores: list[Decimal] = []
    for item in supported:
        rank = item.get("expected_rank")
        score = item.get("expected_document_score")
        if (
            item.get("actual_status") != "found"
            or item.get("passed") is not True
            or not isinstance(rank, int)
            or isinstance(rank, bool)
            or not isinstance(item.get("expected_document_id"), str)
        ):
            raise HoldoutValidationError("supported_outcome_invalid")
        ranks.append(rank)
        expected_scores.append(_decimal(score))

    fallback_successes = sum(item.get("actual_status") == "no_evidence" for item in unsupported)
    status_successes = sum(
        item.get("actual_status") == item.get("expected_status") for item in outcomes
    )
    failed = [item for item in outcomes if item.get("passed") is False]
    failed_ids = [item.get("case_id") for item in failed]
    if failed_ids != ["HOLD-N01", "HOLD-N02", "HOLD-N04"]:
        raise HoldoutValidationError("failed_cases_invalid")
    if artifact.get("failed_case_ids") != failed_ids:
        raise HoldoutValidationError("failed_case_ids_invalid")

    recall = Decimal(sum(rank <= 5 for rank in ranks)) / Decimal(len(supported))
    hit_rate = Decimal(sum(item.get("actual_status") == "found" for item in supported)) / Decimal(
        len(supported)
    )
    top1 = Decimal(sum(rank == 1 for rank in ranks)) / Decimal(len(supported))
    mrr = sum((Decimal(1) / Decimal(rank) for rank in ranks), Decimal(0)) / Decimal(len(supported))
    fallback = Decimal(fallback_successes) / Decimal(len(unsupported))
    status_accuracy = Decimal(status_successes) / Decimal(len(outcomes))
    metrics = {
        "supported_recall_at_5": _rounded(recall),
        "supported_hit_rate_at_5": _rounded(hit_rate),
        "supported_top1_accuracy": _rounded(top1),
        "supported_mrr": _rounded(mrr),
        "unsupported_fallback_accuracy": _rounded(fallback),
        "status_accuracy": _rounded(status_accuracy),
    }
    for key, value in metrics.items():
        _expect(artifact, key, value)

    unsupported_scores = [
        _decimal(item.get("top_score"))
        for item in unsupported
        if item.get("actual_status") == "found"
    ]
    supported_floor = min(expected_scores)
    unsupported_ceiling = max(unsupported_scores)
    separation_gap = supported_floor - unsupported_ceiling
    _expect(artifact, "holdout_supported_floor", supported_floor)
    _expect(artifact, "holdout_unsupported_ceiling", unsupported_ceiling)
    _expect(artifact, "holdout_scalar_separation_gap", separation_gap)
    if separation_gap != Decimal("-0.00206978"):
        raise HoldoutValidationError("separation_gap_invalid")
    if artifact.get("threshold_only_separation_possible") is not (
        supported_floor > unsupported_ceiling
    ):
        raise HoldoutValidationError("separation_decision_invalid")

    failed_cases = artifact.get("failed_cases")
    if not isinstance(failed_cases, list):
        raise HoldoutValidationError("failed_case_details_invalid")
    failed_case_entries = cast(list[object], failed_cases)
    if len(failed_case_entries) != len(failed):
        raise HoldoutValidationError("failed_case_details_invalid")
    for item, failed_case in zip(failed, failed_case_entries, strict=True):
        if not isinstance(failed_case, dict):
            raise HoldoutValidationError("failed_case_details_invalid")
        detail = cast(dict[str, object], failed_case)
        for key in (
            "case_id",
            "kind",
            "actual_status",
            "result_count",
            "top_document_id",
            "top_score",
        ):
            if detail.get(key) != item.get(key):
                raise HoldoutValidationError("failed_case_details_invalid")


def _validate_artifact_constants(artifact: dict[str, object]) -> None:
    expected = {
        "logical_api_calls": 12,
        "physical_attempts": 12,
        "generation_api_calls": 0,
        "prompt_tokens": 230,
        "total_tokens": 230,
        "threshold_frozen": True,
        "threshold_changed": False,
        "supported_quality_passed": True,
        "unsupported_fallback_passed": False,
        "retrieval_decision": "HOLDOUT_FAILED",
        "score_threshold": 0.46,
    }
    if any(artifact.get(key) != value for key, value in expected.items()):
        raise HoldoutValidationError("artifact_constants_invalid")


def _validate_bound_artifacts(root: Path, artifact: dict[str, object]) -> None:
    paths = {
        "threshold_policy_sha256": root / "knowledge_base/index/retrieval-threshold-policy.json",
        "index_manifest_sha256": root / "knowledge_base/index/index-manifest.json",
        "chunks_sha256": root / "knowledge_base/processed/chunks.jsonl",
        "processed_manifest_sha256": root / "knowledge_base/processed/manifest.json",
        "index_plan_sha256": root / "knowledge_base/index/index-plan.json",
    }
    for key, path in paths.items():
        if artifact.get(key) != _sha256(path):
            raise HoldoutValidationError("artifact_hash_invalid")

    policy = _load_object(root / "knowledge_base/index/retrieval-threshold-policy.json")
    manifest = _load_object(root / "knowledge_base/index/index-manifest.json")
    if (
        policy.get("policy_version") != artifact.get("threshold_policy_version")
        or policy.get("threshold_frozen") is not True
        or policy.get("score_threshold") != artifact.get("score_threshold")
    ):
        raise HoldoutValidationError("policy_binding_invalid")
    for key in (
        "collection_name",
        "distance",
        "embedding_provider",
        "embedding_model",
        "embedding_dimensions",
        "total_points",
    ):
        if policy.get(key) != manifest.get(key):
            raise HoldoutValidationError("manifest_binding_invalid")


def validate_holdout_result(root: Path) -> None:
    """Validate the versioned holdout artifact, schema, math, and local bindings."""
    artifact = _load_object(root / ARTIFACT)
    schema = _load_object(root / SCHEMA)
    _validate_safe_content(artifact)
    _validate_schema(schema, artifact)
    _validate_artifact_constants(artifact)
    _validate_case_math(artifact)
    _validate_bound_artifacts(root, artifact)


def main() -> int:
    """Run the offline retrieval holdout result validation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    arguments = parser.parse_args()
    try:
        validate_holdout_result(arguments.root.resolve())
    except HoldoutValidationError:
        print("Retrieval holdout result validation failed.")
        return 1
    print("Retrieval holdout result validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
