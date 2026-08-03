"""Validate the consolidated D03 result and optional ignored local sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, cast

from jsonschema import Draft202012Validator

try:
    from scripts._project_bootstrap import bootstrap_project
except ModuleNotFoundError:  # Direct execution adds scripts/, not the project root, to sys.path.
    from _project_bootstrap import bootstrap_project

SCHEMA_PATH = Path("evals/rag/grounding-diagnostic-d03-result-v1.schema.json")
RESULT_PATH = Path("evals/rag/grounding-diagnostic-d03-result-v1.json")
D04_CONCEPT_SCHEMA_PATH = Path("evals/rag/grounding-diagnostic-d04-signals-concept.schema.json")

_CASE_ORDER = ("HOLD-P02", "HOLD-P03", "HOLD-P04", "HOLD-N04")
_REPRODUCED = ("HOLD-P02", "HOLD-P04")
_NON_REPRODUCED = ("HOLD-P03", "HOLD-N04")
_USAGE_FILENAMES = {
    "HOLD-P02": "phase-6a-grounding-diagnostic-d03-p02-usage.json",
    "HOLD-P03": "phase-6a-grounding-diagnostic-d03-p03-usage.json",
    "HOLD-P04": "phase-6a-grounding-diagnostic-d03-p04-usage.json",
    "HOLD-N04": "phase-6a-grounding-diagnostic-d03-n04-usage.json",
}
_DIAGNOSTIC_FILENAMES = {
    "HOLD-P02": "phase-6a-grounding-diagnostic-d03-p02.json",
    "HOLD-P04": "phase-6a-grounding-diagnostic-d03-p04.json",
}
_EXPECTED_SOURCE_HASHES = {
    "HOLD-P02": (
        "7cc78f487b1d235c178981a58c64a3ba265c9b982902d38e7b94ff1d01f1efdc",
        "9cb7a0b2acacbefc4a35a9f220a137a8ac7f630a63bacd48bee3899d2bfea90f",
    ),
    "HOLD-P03": (
        "bd3f30bcb4001f760f07da378dc8289fbaa7599de3d52fbd82d0b148584974d9",
        None,
    ),
    "HOLD-P04": (
        "8b763ad18ceba298b2bfd5f721f1ca26f4c33f1ccbf48b51967abea5e5757dfb",
        "4fc88fbc90d232e5f3d9277b3380d1b60f076f4cc288f49fc0f63b5ca4698cfc",
    ),
    "HOLD-N04": (
        "64112dcef7b95f4658dd14011fa59a2a6a4ffde9a736c8dec5c59b4beda2812e",
        None,
    ),
}
_FORBIDDEN_KEYS = frozenset(
    {
        "answer",
        "chunk",
        "chunks",
        "citation",
        "citations",
        "context",
        "document",
        "evidence",
        "path",
        "prompt",
        "query",
        "question",
        "quote",
        "request_id",
        "request_ids",
        "timestamp",
        "timestamp_utc",
    }
)
_SENSITIVE_VALUE = re.compile(
    r"(?i)(?:sk-[A-Za-z0-9_-]{8,}|authorization\s*:\s*bearer|"
    r"(?:openai|qdrant)[_-]?(?:api[_-]?)?key\s*[:=]|https?://|"
    r"^[A-Za-z]:[\\/]|^\\\\|^/(?:home|users|var|tmp)/|\breq[_-][A-Za-z0-9])"
)
_TIMESTAMP_VALUE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_COMMIT = re.compile(r"^[a-f0-9]{40}$")
_ALLOWED_STRINGS = frozenset(
    {
        "grounding-diagnostic-d03",
        "1.0.0",
        "COMPLETED",
        "NOT_APPLICABLE",
        "text-embedding-3-small",
        "gpt-5.6-luna",
        "grounding_quote_not_in_evidence",
        "grounding_failed",
        "answered",
        "REPRODUCED_GROUNDING_FAILURE",
        "NON_REPRODUCED_IN_D03",
        "ROOT_CAUSE_INSUFFICIENT_REQUIRE_D04",
        *_CASE_ORDER,
    }
)


class D03ResultValidationError(RuntimeError):
    """Raised with a closed reason that contains no local artifact content."""


def _load_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise D03ResultValidationError("invalid_json") from exc
    if not isinstance(value, dict):
        raise D03ResultValidationError("object_required")
    return cast(dict[str, object], value)


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise D03ResultValidationError("local_source_unavailable") from exc


def _validate_closed_content(value: object) -> None:
    if isinstance(value, dict):
        for raw_key, nested in cast(dict[object, object], value).items():
            if not isinstance(raw_key, str) or raw_key.casefold() in _FORBIDDEN_KEYS:
                raise D03ResultValidationError("prohibited_field")
            _validate_closed_content(nested)
        return
    if isinstance(value, list):
        for nested in cast(list[object], value):
            _validate_closed_content(nested)
        return
    if isinstance(value, str):
        if _SENSITIVE_VALUE.search(value) or _TIMESTAMP_VALUE.search(value):
            raise D03ResultValidationError("prohibited_content")
        if (
            value not in _ALLOWED_STRINGS
            and not _SHA256.fullmatch(value)
            and not _COMMIT.fullmatch(value)
        ):
            raise D03ResultValidationError("free_text_not_allowed")


def _check_schema(schema: dict[str, object]) -> None:
    try:
        Draft202012Validator.check_schema(cast(Any, schema))
    except Exception as exc:
        raise D03ResultValidationError("schema_invalid") from exc


def _validate_schema(schema: dict[str, object], data: dict[str, object]) -> None:
    _check_schema(schema)
    errors = list(Draft202012Validator(cast(Any, schema)).iter_errors(cast(Any, data)))  # pyright: ignore[reportUnknownMemberType] - jsonschema is a dynamic boundary.
    if errors:
        raise D03ResultValidationError("schema_validation_failed")


def _case_map(data: dict[str, object]) -> dict[str, dict[str, object]]:
    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list):
        raise D03ResultValidationError("cases_invalid")
    cases: dict[str, dict[str, object]] = {}
    order: list[str] = []
    for raw_case in cast(list[object], raw_cases):
        if not isinstance(raw_case, dict):
            raise D03ResultValidationError("case_invalid")
        case = cast(dict[str, object], raw_case)
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or case_id in cases:
            raise D03ResultValidationError("case_set_invalid")
        cases[case_id] = case
        order.append(case_id)
    if tuple(order) != _CASE_ORDER:
        raise D03ResultValidationError("case_order_invalid")
    return cases


def _validate_semantics(data: dict[str, object]) -> None:
    cases = _case_map(data)
    if set(cases) != set(_CASE_ORDER):
        raise D03ResultValidationError("case_set_invalid")
    for case_id, case in cases.items():
        reproduced = case_id in _REPRODUCED
        expected_status = "grounding_failed" if reproduced else "answered"
        expected_code = "grounding_quote_not_in_evidence" if reproduced else None
        expected_classification = (
            "REPRODUCED_GROUNDING_FAILURE" if reproduced else "NON_REPRODUCED_IN_D03"
        )
        usage_hash, artifact_hash = _EXPECTED_SOURCE_HASHES[case_id]
        expected = {
            "status": expected_status,
            "safe_error_code": expected_code,
            "classification": expected_classification,
            "grounding_failure_reproduced": reproduced,
            "diagnostic_artifact_created": reproduced,
            "usage_report_sha256": usage_hash,
            "diagnostic_artifact_sha256": artifact_hash,
            "retrieval_logical_api_calls": 1,
            "retrieval_physical_attempts": 1,
            "answer_logical_api_calls": 1,
            "answer_physical_attempts": 1,
        }
        if any(case.get(key) != expected_value for key, expected_value in expected.items()):
            raise D03ResultValidationError("case_contract_invalid")
        if case["retrieval_logical_api_calls"] != case["retrieval_physical_attempts"]:
            raise D03ResultValidationError("retrieval_attempts_invalid")
        if case["answer_logical_api_calls"] != case["answer_physical_attempts"]:
            raise D03ResultValidationError("answer_attempts_invalid")
    consolidated = {
        "cases_executed": len(cases),
        "grounding_failures_reproduced": len(_REPRODUCED),
        "diagnostic_artifacts_created": len(_REPRODUCED),
        "reproduced_case_ids": list(_REPRODUCED),
        "non_reproduced_case_ids": list(_NON_REPRODUCED),
        "safe_error_code_counts": {"grounding_quote_not_in_evidence": len(_REPRODUCED)},
    }
    if any(data.get(key) != expected for key, expected in consolidated.items()):
        raise D03ResultValidationError("consolidated_counts_invalid")
    fixed = {
        "threshold": 0.46,
        "top_k": 5,
        "embedding_model": "text-embedding-3-small",
        "answer_model": "gpt-5.6-luna",
        "holdout_quality_decision": "NOT_APPLICABLE",
        "r02_history_preserved": True,
        "threshold_preserved": True,
        "r03_created": False,
        "privacy_contract_passed": True,
        "root_cause_classification": "ROOT_CAUSE_INSUFFICIENT_REQUIRE_D04",
        "d04_required": True,
    }
    if any(data.get(key) != expected for key, expected in fixed.items()):
        raise D03ResultValidationError("fixed_contract_invalid")


def validate_result(root: Path, result: Path | None = None) -> dict[str, object]:
    """Validate the closed schema, privacy boundary, and D03 semantic invariants."""
    schema = _load_object(root / SCHEMA_PATH)
    _check_schema(_load_object(root / D04_CONCEPT_SCHEMA_PATH))
    candidate = root / RESULT_PATH if result is None else result
    if not candidate.is_absolute():
        candidate = root / candidate
    data = _load_object(candidate)
    _validate_closed_content(data)
    _validate_schema(schema, data)
    _validate_semantics(data)
    return data


def _usage_components(report: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    retrieval = report.get("retrieval_usage")
    answer = report.get("answer_usage")
    if not isinstance(retrieval, dict) or not isinstance(answer, dict):
        raise D03ResultValidationError("usage_components_invalid")
    return cast(dict[str, object], retrieval), cast(dict[str, object], answer)


def _validate_usage_report(
    case_id: str, report: dict[str, object], case: dict[str, object]
) -> None:
    reproduced = case_id in _REPRODUCED
    expected_status = "grounding_failed" if reproduced else "answered"
    expected_code = "grounding_quote_not_in_evidence" if reproduced else None
    if (report.get("status"), report.get("safe_error_code")) != (
        expected_status,
        expected_code,
    ):
        raise D03ResultValidationError("usage_status_invalid")
    retrieval, answer = _usage_components(report)
    if retrieval.get("model") != "text-embedding-3-small":
        raise D03ResultValidationError("retrieval_model_invalid")
    if answer.get("model") != "gpt-5.6-luna":
        raise D03ResultValidationError("answer_model_invalid")
    pairs = (
        (retrieval, "retrieval_logical_api_calls", "retrieval_physical_attempts"),
        (answer, "answer_logical_api_calls", "answer_physical_attempts"),
    )
    for usage, logical_key, physical_key in pairs:
        if usage.get("logical_api_calls") != case.get(logical_key):
            raise D03ResultValidationError("usage_calls_invalid")
        if usage.get("physical_attempts") != case.get(physical_key):
            raise D03ResultValidationError("usage_attempts_invalid")
        if usage.get("logical_api_calls") != usage.get("physical_attempts"):
            raise D03ResultValidationError("usage_attempts_invalid")
        token_parts = (
            usage.get("prompt_tokens"),
            usage.get("output_tokens"),
            usage.get("total_tokens"),
        )
        if all(isinstance(value, int) and not isinstance(value, bool) for value in token_parts):
            prompt_tokens, output_tokens, total_tokens = cast(tuple[int, int, int], token_parts)
            if prompt_tokens + output_tokens != total_tokens:
                raise D03ResultValidationError("usage_tokens_invalid")
    logical_total = cast(int, retrieval["logical_api_calls"]) + cast(
        int, answer["logical_api_calls"]
    )
    physical_total = cast(int, retrieval["physical_attempts"]) + cast(
        int, answer["physical_attempts"]
    )
    if report.get("logical_api_calls") != logical_total:
        raise D03ResultValidationError("aggregate_calls_invalid")
    if report.get("physical_attempts") != physical_total:
        raise D03ResultValidationError("aggregate_attempts_invalid")
    report_token_parts = (
        report.get("prompt_tokens"),
        report.get("output_tokens"),
        report.get("total_tokens"),
    )
    if all(isinstance(value, int) and not isinstance(value, bool) for value in report_token_parts):
        prompt_tokens, output_tokens, total_tokens = cast(tuple[int, int, int], report_token_parts)
        if prompt_tokens + output_tokens != total_tokens:
            raise D03ResultValidationError("aggregate_tokens_invalid")
    component_tokens = (retrieval.get("total_tokens"), answer.get("total_tokens"))
    if all(isinstance(value, int) and not isinstance(value, bool) for value in component_tokens):
        if report.get("total_tokens") != sum(cast(tuple[int, int], component_tokens)):
            raise D03ResultValidationError("token_total_invalid")


def verify_local_sources(root: Path, data: dict[str, object], source_dir: Path) -> None:
    """Verify the ignored D03 sources by hash, closed schemas, privacy, and usage arithmetic."""
    bootstrap_project()
    try:
        from scripts.validate_grounding_diagnostic import (
            GroundingDiagnosticValidationError,
            validate_grounding_diagnostic,
        )
    except ModuleNotFoundError:
        from validate_grounding_diagnostic import (
            GroundingDiagnosticValidationError,
            validate_grounding_diagnostic,
        )

    from nexodocs_ai.observability.reports import (
        ReportError,
        validate_privacy_safe_report,
    )

    directory = source_dir if source_dir.is_absolute() else root / source_dir
    cases = _case_map(data)
    for case_id in _CASE_ORDER:
        case = cases[case_id]
        usage_path = directory / _USAGE_FILENAMES[case_id]
        if _sha256(usage_path) != case.get("usage_report_sha256"):
            raise D03ResultValidationError("usage_hash_mismatch")
        usage = _load_object(usage_path)
        try:
            validate_privacy_safe_report(usage)
        except ReportError as exc:
            raise D03ResultValidationError("usage_privacy_invalid") from exc
        _validate_usage_report(case_id, usage, case)
        artifact_name = _DIAGNOSTIC_FILENAMES.get(case_id)
        if artifact_name is None:
            if case.get("diagnostic_artifact_sha256") is not None:
                raise D03ResultValidationError("unexpected_diagnostic_hash")
            continue
        artifact_path = directory / artifact_name
        if _sha256(artifact_path) != case.get("diagnostic_artifact_sha256"):
            raise D03ResultValidationError("diagnostic_hash_mismatch")
        try:
            validate_grounding_diagnostic(root, artifact_path)
        except GroundingDiagnosticValidationError as exc:
            raise D03ResultValidationError("diagnostic_artifact_invalid") from exc
        artifact = _load_object(artifact_path)
        artifact_expected = {
            "case_id": case_id,
            "status": "grounding_failed",
            "safe_error_code": "grounding_quote_not_in_evidence",
            "usage_report_sha256": case["usage_report_sha256"],
            "retrieval_logical_api_calls": case["retrieval_logical_api_calls"],
            "retrieval_physical_attempts": case["retrieval_physical_attempts"],
            "answer_logical_api_calls": case["answer_logical_api_calls"],
            "answer_physical_attempts": case["answer_physical_attempts"],
            "execution_completed": True,
            "privacy_scan_passed": True,
        }
        if any(artifact.get(key) != value for key, value in artifact_expected.items()):
            raise D03ResultValidationError("diagnostic_contract_invalid")
        tokens = (
            artifact.get("retrieval_total_tokens"),
            artifact.get("answer_total_tokens"),
        )
        if all(isinstance(value, int) and not isinstance(value, bool) for value in tokens):
            if artifact.get("total_tokens") != sum(cast(tuple[int, int], tokens)):
                raise D03ResultValidationError("diagnostic_tokens_invalid")


def main() -> int:
    """Run the CI-safe result validation and optional local-source verification."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate the sanitized D03 result; local ignored reports are optional and offline."
        )
    )
    parser.add_argument("--root", type=Path, help="Project root (defaults to auto-detection).")
    parser.add_argument(
        "--result", type=Path, help="Result JSON (defaults to the versioned v1 file)."
    )
    parser.add_argument(
        "--verify-local-sources",
        action="store_true",
        help="Verify the six ignored D03 files without displaying their content.",
    )
    parser.add_argument(
        "--local-source-dir",
        type=Path,
        default=Path("data/run-reports"),
        help="Directory containing the six local D03 files.",
    )
    arguments = parser.parse_args()
    root = bootstrap_project() if arguments.root is None else arguments.root.resolve()
    try:
        data = validate_result(root, arguments.result)
        if arguments.verify_local_sources:
            verify_local_sources(root, data, arguments.local_source_dir)
    except D03ResultValidationError:
        print("Grounding diagnostic D03 result validation failed.")
        return 1
    print("Grounding diagnostic D03 result validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
