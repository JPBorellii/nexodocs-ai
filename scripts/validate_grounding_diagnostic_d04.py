"""Validate the closed D04 artifact without executing providers or retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from _project_bootstrap import bootstrap_project
from jsonschema import Draft202012Validator

if TYPE_CHECKING:
    from nexodocs_ai.rag.sanitized_grounding import SanitizedGroundingSignalSet

SCHEMA_PATH = Path("evals/rag/grounding-diagnostic-d04.schema.json")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_COMMIT = re.compile(r"^[a-f0-9]{40}$")
_PROHIBITED_VALUE = re.compile(
    r"(?i)(?:sk-[A-Za-z0-9_-]{8,}|authorization\s*:\s*bearer|https?://|"
    r"^[A-Za-z]:[\\/]|^\\\\|^/(?:home|users|var|tmp)/|\breq[_-][A-Za-z0-9]|"
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
)
_PROHIBITED_KEYS = frozenset(
    {
        "question",
        "query",
        "answer",
        "response",
        "quote",
        "evidence",
        "citation",
        "citations",
        "supporting_excerpt",
        "document",
        "document_id",
        "chunk",
        "chunk_id",
        "context",
        "prompt",
        "marker",
        "tokens_text",
        "substring",
        "position",
        "n_gram",
        "embedding",
        "request_id",
        "request_ids",
        "run_id",
        "timestamp",
        "timestamp_utc",
        "path",
        "hostname",
        "username",
        "exception",
        "traceback",
        "key",
        "secret",
    }
)


class D04ValidationError(RuntimeError):
    """Closed validation failure without candidate content."""


def _load(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        raise D04ValidationError("invalid_json") from None
    if not isinstance(value, dict):
        raise D04ValidationError("object_required")
    return cast(dict[str, object], value)


def _hash(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        raise D04ValidationError("source_unavailable") from None


def _head(root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=False, capture_output=True, text=True
        )
    except OSError:
        raise D04ValidationError("commit_unavailable") from None
    value = completed.stdout.strip()
    if completed.returncode != 0 or not _COMMIT.fullmatch(value):
        raise D04ValidationError("commit_unavailable")
    return value


def _privacy_scan(value: object, allowed_strings: frozenset[str]) -> None:
    if isinstance(value, dict):
        for raw_key, nested in cast(dict[object, object], value).items():
            if not isinstance(raw_key, str):
                raise D04ValidationError("non_text_key")
            if raw_key.casefold() in _PROHIBITED_KEYS:
                raise D04ValidationError("prohibited_field")
            _privacy_scan(nested, allowed_strings)
        return
    if isinstance(value, list):
        for nested in cast(list[object], value):
            _privacy_scan(nested, allowed_strings)
        return
    if isinstance(value, str):
        if _PROHIBITED_VALUE.search(value):
            raise D04ValidationError("prohibited_content")
        if (
            value not in allowed_strings
            and not _SHA256.fullmatch(value)
            and not _COMMIT.fullmatch(value)
        ):
            raise D04ValidationError("free_text_not_allowed")


def _signal_from_dict(value: dict[str, object]) -> SanitizedGroundingSignalSet:
    bootstrap_project()
    from nexodocs_ai.rag.sanitized_grounding import (
        LengthBucket,
        OverlapBucket,
        SanitizedGroundingSignalSet,
    )

    converted = dict(value)
    converted["quote_length_bucket"] = LengthBucket(cast(str, value["quote_length_bucket"]))
    converted["evidence_length_bucket"] = LengthBucket(cast(str, value["evidence_length_bucket"]))
    converted["character_overlap_bucket"] = OverlapBucket(
        cast(str, value["character_overlap_bucket"])
    )
    converted["token_overlap_bucket"] = OverlapBucket(cast(str, value["token_overlap_bucket"]))
    return SanitizedGroundingSignalSet(**converted)  # pyright: ignore[reportArgumentType]


def _validate_intrinsic_semantics(data: dict[str, object]) -> None:
    bootstrap_project()
    from nexodocs_ai.rag.sanitized_grounding import aggregate_classification

    raw_signals = data.get("signal_sets")
    if not isinstance(raw_signals, list):
        raise D04ValidationError("signals_invalid")
    raw_signal_items = cast(list[object], raw_signals)
    if not all(isinstance(item, dict) for item in raw_signal_items):
        raise D04ValidationError("signals_invalid")
    signals = tuple(_signal_from_dict(cast(dict[str, object], item)) for item in raw_signal_items)
    if data.get("quote_failure_count") != len(signals):
        raise D04ValidationError("failure_count_invalid")
    if data.get("sanitized_root_cause_class") != aggregate_classification(signals).value:
        raise D04ValidationError("classification_incoherent")
    for logical, physical in (
        ("retrieval_logical_api_calls", "retrieval_physical_attempts"),
        ("answer_logical_api_calls", "answer_physical_attempts"),
    ):
        if data.get(logical) != 1 or data.get(physical) != 1:
            raise D04ValidationError("calls_attempts_invalid")
    left, right, total = (
        data.get("retrieval_total_tokens"),
        data.get("answer_total_tokens"),
        data.get("total_tokens"),
    )
    if left is None or right is None:
        if total is not None:
            raise D04ValidationError("tokens_incoherent")
    elif not isinstance(left, int) or not isinstance(right, int) or total != left + right:
        raise D04ValidationError("tokens_incoherent")


def _validate_source_bindings(root: Path, data: dict[str, object], expected_commit: str) -> None:
    bootstrap_project()
    from nexodocs_ai.observability.grounding_diagnostic_d04 import SOURCE_PATHS

    if data.get("system_commit") != expected_commit:
        raise D04ValidationError("commit_mismatch")
    for field, relative in SOURCE_PATHS.items():
        if data.get(field) != _hash(root / relative):
            raise D04ValidationError("source_hash_mismatch")


def _validate_usage_consistency(artifact: dict[str, object], usage: dict[str, object]) -> None:
    retrieval = usage.get("retrieval_usage")
    answer = usage.get("answer_usage")
    if not isinstance(retrieval, dict) or not isinstance(answer, dict):
        raise D04ValidationError("usage_components_invalid")
    retrieval_data = cast(dict[str, object], retrieval)
    answer_data = cast(dict[str, object], answer)
    comparisons = (
        ("retrieval_logical_api_calls", retrieval_data.get("logical_api_calls")),
        ("retrieval_physical_attempts", retrieval_data.get("physical_attempts")),
        ("retrieval_total_tokens", retrieval_data.get("total_tokens")),
        ("answer_logical_api_calls", answer_data.get("logical_api_calls")),
        ("answer_physical_attempts", answer_data.get("physical_attempts")),
        ("answer_total_tokens", answer_data.get("total_tokens")),
        ("total_tokens", usage.get("total_tokens")),
    )
    if any(artifact.get(field) != value for field, value in comparisons):
        raise D04ValidationError("usage_counters_mismatch")
    logical_total = cast(int, artifact["retrieval_logical_api_calls"]) + cast(
        int, artifact["answer_logical_api_calls"]
    )
    physical_total = cast(int, artifact["retrieval_physical_attempts"]) + cast(
        int, artifact["answer_physical_attempts"]
    )
    if usage.get("logical_api_calls") != logical_total:
        raise D04ValidationError("usage_calls_mismatch")
    if usage.get("physical_attempts") != physical_total:
        raise D04ValidationError("usage_attempts_mismatch")


def _allowed_strings() -> frozenset[str]:
    bootstrap_project()
    from nexodocs_ai.rag.sanitized_grounding import (
        LengthBucket,
        OverlapBucket,
        SanitizedRootCauseClass,
    )

    return frozenset(
        {
            "grounding-diagnostic-d04",
            "1.0.0",
            "HOLD-P02",
            "HOLD-P04",
            "grounding_failed",
            "grounding_quote_not_in_evidence",
            "quote_membership",
            "NOT_APPLICABLE",
            *(item.value for item in LengthBucket),
            *(item.value for item in OverlapBucket),
            *(item.value for item in SanitizedRootCauseClass),
        }
    )


def validate_d04_artifact_file(
    root: Path,
    artifact: Path,
    *,
    expected_commit: str | None = None,
    verify_sources: bool,
) -> dict[str, object]:
    """Validate one artifact, optionally binding it to the current local sources."""
    schema = _load(root / SCHEMA_PATH)
    try:
        Draft202012Validator.check_schema(cast(Any, schema))
    except Exception:
        raise D04ValidationError("schema_invalid") from None
    candidate = artifact if artifact.is_absolute() else root / artifact
    data = _load(candidate)
    errors = list(Draft202012Validator(cast(Any, schema)).iter_errors(cast(Any, data)))  # pyright: ignore[reportUnknownMemberType]
    if errors:
        raise D04ValidationError("schema_validation_failed")
    _privacy_scan(data, _allowed_strings())
    _validate_intrinsic_semantics(data)
    if verify_sources:
        _validate_source_bindings(root, data, expected_commit or _head(root))
    return data


def validate_d04(
    root: Path,
    artifact: Path | None = None,
    *,
    usage_report: Path | None = None,
    expected_commit: str | None = None,
    schema_only: bool = False,
) -> dict[str, object] | None:
    """Validate explicitly in structural CI mode or complete operational mode."""
    schema = _load(root / SCHEMA_PATH)
    try:
        Draft202012Validator.check_schema(cast(Any, schema))
    except Exception:
        raise D04ValidationError("schema_invalid") from None
    if schema_only:
        if usage_report is not None or expected_commit is not None:
            raise D04ValidationError("schema_only_sources_forbidden")
        return (
            None
            if artifact is None
            else validate_d04_artifact_file(root, artifact, verify_sources=False)
        )
    if artifact is None:
        raise D04ValidationError("artifact_required")
    if usage_report is None:
        raise D04ValidationError("usage_report_required")
    data = validate_d04_artifact_file(
        root,
        artifact,
        expected_commit=expected_commit,
        verify_sources=True,
    )
    usage_path = usage_report if usage_report.is_absolute() else root / usage_report
    if data.get("usage_report_sha256") != _hash(usage_path):
        raise D04ValidationError("usage_hash_mismatch")
    from nexodocs_ai.observability.reports import ReportError, validate_privacy_safe_report

    usage = _load(usage_path)
    try:
        validate_privacy_safe_report(usage)
    except ReportError:
        raise D04ValidationError("usage_privacy_invalid") from None
    if (usage.get("status"), usage.get("safe_error_code")) != (
        "grounding_failed",
        "grounding_quote_not_in_evidence",
    ):
        raise D04ValidationError("usage_status_invalid")
    _validate_usage_consistency(data, usage)
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate sanitized grounding diagnostic D04")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--usage-report", type=Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--schema-only", action="store_true")
    arguments = parser.parse_args()
    root = bootstrap_project() if arguments.root is None else arguments.root.resolve()
    try:
        validate_d04(
            root,
            arguments.artifact,
            usage_report=arguments.usage_report,
            expected_commit=arguments.expected_commit,
            schema_only=arguments.schema_only,
        )
    except D04ValidationError:
        print("Grounding diagnostic D04 validation failed.")
        return 1
    print(
        "Grounding diagnostic D04 schema validation passed."
        if arguments.schema_only
        else "Grounding diagnostic D04 operational validation passed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
