"""Validate the versioned D04 result and optional ignored local sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, cast

from _project_bootstrap import bootstrap_project
from jsonschema import Draft202012Validator

SUMMARY_SCHEMA_PATH = Path("evals/rag/grounding-diagnostic-d04-result.schema.json")
VERSIONED_SCHEMA_PATH = Path("evals/rag/grounding-diagnostic-d04-result-v1.schema.json")
RESULT_PATH = Path("evals/rag/grounding-diagnostic-d04-result-v1.json")

_CASE_IDS = ("HOLD-P02", "HOLD-P04")
_ARTIFACT_FILENAMES = {
    "HOLD-P02": "phase-6a-grounding-diagnostic-d04-p02.json",
    "HOLD-P04": "phase-6a-grounding-diagnostic-d04-p04.json",
}
_USAGE_FILENAMES = {
    "HOLD-P02": "phase-6a-grounding-diagnostic-d04-p02-usage.json",
    "HOLD-P04": "phase-6a-grounding-diagnostic-d04-p04-usage.json",
}
_SUMMARY_FILENAME = "phase-6a-grounding-diagnostic-d04-summary.json"
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_COMMIT = re.compile(r"^[a-f0-9]{40}$")
_PROHIBITED_KEYS = frozenset(
    {
        "answer",
        "chunk",
        "citation",
        "content",
        "document",
        "evidence",
        "normalized_content",
        "path",
        "prompt",
        "query",
        "question",
        "quote",
        "request_id",
        "timestamp",
    }
)
_PROHIBITED_VALUE = re.compile(
    r"(?i)(?:sk-[A-Za-z0-9_-]{8,}|authorization\s*:\s*bearer|https?://|"
    r"^[A-Za-z]:[\\/]|^\\\\|^/(?:home|users|var|tmp)/|\breq[_-][A-Za-z0-9]|"
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
)
_ALLOWED_STRINGS = frozenset(
    {
        "grounding-diagnostic-d04-result",
        "1.0.0",
        "HOLD-P02",
        "HOLD-P04",
        "WHITESPACE_ONLY_DIFFERENCE",
        "NOT_APPLICABLE",
        "ROOT_CAUSE_CONFIRMED_WHITESPACE_CANONICALIZATION_GAP",
    }
)


class D04ResultValidationError(RuntimeError):
    """Closed consolidated validation failure without local artifact content."""


def _load(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        raise D04ResultValidationError("invalid_json") from None
    if not isinstance(value, dict):
        raise D04ResultValidationError("object_required")
    return cast(dict[str, object], value)


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        raise D04ResultValidationError("source_unavailable") from None


def _check_schema(schema: dict[str, object]) -> None:
    try:
        Draft202012Validator.check_schema(cast(Any, schema))
    except Exception:
        raise D04ResultValidationError("schema_invalid") from None


def _validate_schema(schema: dict[str, object], data: dict[str, object]) -> None:
    _check_schema(schema)
    errors = list(Draft202012Validator(cast(Any, schema)).iter_errors(cast(Any, data)))  # pyright: ignore[reportUnknownMemberType]
    if errors:
        raise D04ResultValidationError("schema_validation_failed")


def _privacy_scan(value: object) -> None:
    if isinstance(value, dict):
        for raw_key, nested in cast(dict[object, object], value).items():
            if not isinstance(raw_key, str) or raw_key.casefold() in _PROHIBITED_KEYS:
                raise D04ResultValidationError("prohibited_field")
            _privacy_scan(nested)
        return
    if isinstance(value, list):
        for nested in cast(list[object], value):
            _privacy_scan(nested)
        return
    if isinstance(value, str):
        if _PROHIBITED_VALUE.search(value):
            raise D04ResultValidationError("prohibited_content")
        if (
            value not in _ALLOWED_STRINGS
            and not _SHA256.fullmatch(value)
            and not _COMMIT.fullmatch(value)
        ):
            raise D04ResultValidationError("free_text_not_allowed")


def _validate_summary_semantics(data: dict[str, object]) -> None:
    classifications = cast(dict[str, str], data["case_classifications"])
    counts = Counter(classifications.values())
    if data["classification_counts"] != dict(sorted(counts.items())):
        raise D04ResultValidationError("classification_counts_invalid")
    inconclusive = sorted(
        case for case, classification in classifications.items() if classification == "INCONCLUSIVE"
    )
    classified = sorted(case for case in classifications if case not in inconclusive)
    if sorted(cast(list[str], data["inconclusive_cases"])) != inconclusive:
        raise D04ResultValidationError("inconclusive_cases_invalid")
    if sorted(cast(list[str], data["classified_cases"])) != classified:
        raise D04ResultValidationError("classified_cases_invalid")


def _validate_versioned_result(root: Path, result: Path | None) -> dict[str, object]:
    schema = _load(root / VERSIONED_SCHEMA_PATH)
    candidate = root / RESULT_PATH if result is None else result
    if not candidate.is_absolute():
        candidate = root / candidate
    data = _load(candidate)
    _privacy_scan(data)
    _validate_schema(schema, data)
    classifications = cast(dict[str, str], data["case_classifications"])
    if data["classification_counts"] != dict(sorted(Counter(classifications.values()).items())):
        raise D04ResultValidationError("classification_counts_invalid")
    if data["inconclusive_count"] != len(cast(list[str], data["inconclusive_cases"])):
        raise D04ResultValidationError("inconclusive_count_invalid")
    return data


def _validate_operational_summary(
    root: Path,
    result: Path,
    artifact_p02: Path,
    artifact_p04: Path,
    *,
    expected_commit: str | None = None,
) -> dict[str, object]:
    schema = _load(root / SUMMARY_SCHEMA_PATH)
    candidate = result if result.is_absolute() else root / result
    data = _load(candidate)
    _validate_schema(schema, data)
    _validate_summary_semantics(data)
    try:
        from scripts.validate_grounding_diagnostic_d04 import validate_d04_artifact_file
    except ModuleNotFoundError:
        from validate_grounding_diagnostic_d04 import validate_d04_artifact_file
    artifacts = {"HOLD-P02": artifact_p02, "HOLD-P04": artifact_p04}
    hashes = cast(dict[str, str], data["artifact_sha256_by_case"])
    classifications = cast(dict[str, str], data["case_classifications"])
    for case_id, path in artifacts.items():
        resolved = path if path.is_absolute() else root / path
        if hashes[case_id] != _sha256(resolved):
            raise D04ResultValidationError("artifact_hash_mismatch")
        try:
            artifact = validate_d04_artifact_file(
                root,
                resolved,
                expected_commit=expected_commit,
                verify_sources=True,
            )
        except Exception:
            raise D04ResultValidationError("artifact_invalid") from None
        if artifact.get("case_id") != case_id:
            raise D04ResultValidationError("artifact_case_invalid")
        if artifact.get("sanitized_root_cause_class") != classifications[case_id]:
            raise D04ResultValidationError("artifact_classification_invalid")
    return data


def validate_result(
    root: Path,
    result: Path | None = None,
    *,
    artifact_p02: Path | None = None,
    artifact_p04: Path | None = None,
    schema_only: bool = False,
) -> dict[str, object] | None:
    """Validate the versioned result, source-summary schema, or operational summary."""
    summary_schema = _load(root / SUMMARY_SCHEMA_PATH)
    versioned_schema = _load(root / VERSIONED_SCHEMA_PATH)
    _check_schema(summary_schema)
    _check_schema(versioned_schema)
    if schema_only:
        if artifact_p02 is not None or artifact_p04 is not None:
            raise D04ResultValidationError("schema_only_artifacts_forbidden")
        if result is None:
            return None
        candidate = result if result.is_absolute() else root / result
        data = _load(candidate)
        _validate_schema(summary_schema, data)
        _validate_summary_semantics(data)
        return data
    if artifact_p02 is not None or artifact_p04 is not None:
        if result is None or artifact_p02 is None or artifact_p04 is None:
            raise D04ResultValidationError("operational_sources_incomplete")
        return _validate_operational_summary(root, result, artifact_p02, artifact_p04)
    return _validate_versioned_result(root, result)


def verify_local_sources(root: Path, data: dict[str, object], source_dir: Path) -> None:
    """Verify all five ignored D04 sources without displaying their content."""
    try:
        from scripts.validate_grounding_diagnostic_d04 import validate_d04
    except ModuleNotFoundError:
        from validate_grounding_diagnostic_d04 import validate_d04

    directory = source_dir if source_dir.is_absolute() else root / source_dir
    artifact_hashes = cast(dict[str, str], data["artifact_sha256_by_case"])
    usage_hashes = cast(dict[str, str], data["usage_report_sha256_by_case"])
    classifications = cast(dict[str, str], data["case_classifications"])
    failure_counts = cast(dict[str, int], data["quote_failure_counts_by_case"])
    expected_commit = cast(str, data["system_commit"])
    artifact_paths: dict[str, Path] = {}
    for case_id in _CASE_IDS:
        artifact_path = directory / _ARTIFACT_FILENAMES[case_id]
        usage_path = directory / _USAGE_FILENAMES[case_id]
        artifact_paths[case_id] = artifact_path
        if _sha256(artifact_path) != artifact_hashes[case_id]:
            raise D04ResultValidationError("artifact_hash_mismatch")
        if _sha256(usage_path) != usage_hashes[case_id]:
            raise D04ResultValidationError("usage_hash_mismatch")
        try:
            artifact = validate_d04(
                root,
                artifact_path,
                usage_report=usage_path,
                expected_commit=expected_commit,
            )
        except Exception:
            raise D04ResultValidationError("artifact_or_usage_invalid") from None
        if artifact is None:
            raise D04ResultValidationError("artifact_missing")
        expected = {
            "case_id": case_id,
            "sanitized_root_cause_class": classifications[case_id],
            "quote_failure_count": failure_counts[case_id],
            "usage_report_sha256": usage_hashes[case_id],
        }
        if any(artifact.get(key) != value for key, value in expected.items()):
            raise D04ResultValidationError("artifact_contract_invalid")
    summary_path = directory / _SUMMARY_FILENAME
    if _sha256(summary_path) != data["summary_sha256"]:
        raise D04ResultValidationError("summary_hash_mismatch")
    summary = _validate_operational_summary(
        root,
        summary_path,
        artifact_paths["HOLD-P02"],
        artifact_paths["HOLD-P04"],
        expected_commit=expected_commit,
    )
    expected_summary = {
        "classification_counts": data["classification_counts"],
        "case_classifications": data["case_classifications"],
        "classified_cases": data["classified_cases"],
        "inconclusive_cases": data["inconclusive_cases"],
        "artifact_integrity_passed": data["artifact_integrity_passed"],
        "privacy_contract_passed": data["privacy_contract_passed"],
        "holdout_quality_decision": data["holdout_quality_decision"],
        "r03_created": data["r03_created"],
    }
    if any(summary.get(key) != value for key, value in expected_summary.items()):
        raise D04ResultValidationError("summary_contract_invalid")


def main() -> int:
    """Run CI-safe validation and optional local-source verification."""
    parser = argparse.ArgumentParser(
        description="Validate the sanitized D04 result and optional ignored local reports."
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--p02-artifact", type=Path)
    parser.add_argument("--p04-artifact", type=Path)
    parser.add_argument("--schema-only", action="store_true")
    parser.add_argument("--verify-local-sources", action="store_true")
    parser.add_argument(
        "--local-source-dir",
        type=Path,
        default=Path("data/run-reports"),
    )
    arguments = parser.parse_args()
    root = bootstrap_project() if arguments.root is None else arguments.root.resolve()
    try:
        if arguments.verify_local_sources and (
            arguments.schema_only
            or arguments.p02_artifact is not None
            or arguments.p04_artifact is not None
        ):
            raise D04ResultValidationError("incompatible_modes")
        data = validate_result(
            root,
            arguments.result,
            artifact_p02=arguments.p02_artifact,
            artifact_p04=arguments.p04_artifact,
            schema_only=arguments.schema_only,
        )
        if arguments.verify_local_sources:
            if data is None:
                raise D04ResultValidationError("result_required")
            verify_local_sources(root, data, arguments.local_source_dir)
    except D04ResultValidationError:
        print("Grounding diagnostic D04 result validation failed.")
        return 1
    if arguments.schema_only:
        message = "Grounding diagnostic D04 result schema validation passed."
    elif arguments.p02_artifact is not None:
        message = "Grounding diagnostic D04 result operational validation passed."
    else:
        message = "Grounding diagnostic D04 result validation passed."
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
