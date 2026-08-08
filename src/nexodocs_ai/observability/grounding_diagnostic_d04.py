"""Closed artifact boundary for the opt-in sanitized grounding diagnostic D04."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import tempfile
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from jsonschema import Draft202012Validator

from nexodocs_ai.rag.models import RagRunResult
from nexodocs_ai.rag.sanitized_grounding import (
    SanitizedRootCauseClass,
    aggregate_classification,
)

DIAGNOSTIC_ID = "grounding-diagnostic-d04"
DIAGNOSTIC_VERSION = "1.0.0"
ARTIFACT_SCHEMA = Path("evals/rag/grounding-diagnostic-d04.schema.json")
REPORT_DIRECTORY = Path("data/run-reports")
SOURCE_PATHS = {
    "fixture_sha256": Path("evals/rag/full-rag-holdout-r02-cases.json"),
    "system_freeze_sha256": Path("evals/rag/full-rag-holdout-r02-system-freeze.json"),
    "threshold_policy_sha256": Path("knowledge_base/index/retrieval-threshold-policy.json"),
    "d03_result_sha256": Path("evals/rag/grounding-diagnostic-d03-result-v1.json"),
}
ALLOWED_CASES = frozenset({"HOLD-P02", "HOLD-P04"})


class D04ErrorCode(StrEnum):
    """Closed public failure codes for the D04 operational boundary."""

    DESTINATION_CONFLICT = "d04_destination_conflict"
    DESTINATION_EXISTS = "d04_destination_exists"
    DESTINATION_INVALID = "d04_destination_invalid"
    ARTIFACT_WRITE_FAILED = "d04_artifact_write_failed"
    USAGE_REPORT_REQUIRED = "d04_usage_report_required"
    PRIVACY_SAFE_REQUIRED = "d04_privacy_safe_required"
    CASE_ID_INVALID = "d04_case_id_invalid"
    DIAGNOSTIC_NOT_APPLICABLE = "d04_diagnostic_not_applicable"
    SOURCE_UNAVAILABLE = "d04_source_unavailable"
    SCHEMA_INVALID = "d04_schema_invalid"
    ARTIFACT_INVALID = "d04_artifact_invalid"


class D04ArtifactError(RuntimeError):
    """Closed operational failure that never includes paths or source content."""

    def __init__(self, code: D04ErrorCode) -> None:
        super().__init__("Sanitized grounding diagnostic unavailable")
        self.code = code.value


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        raise D04ArtifactError(D04ErrorCode.SOURCE_UNAVAILABLE) from None


def _system_commit(root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        raise D04ArtifactError(D04ErrorCode.SOURCE_UNAVAILABLE) from None
    value = completed.stdout.strip()
    if (
        completed.returncode != 0
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise D04ArtifactError(D04ErrorCode.SOURCE_UNAVAILABLE)
    return value


def _optional_total(left: int | None, right: int | None) -> int | None:
    return None if left is None or right is None else left + right


def build_d04_artifact(
    root: Path,
    case_id: str,
    run: RagRunResult,
    usage_report: Path,
) -> dict[str, object]:
    """Build the closed D04 artifact from safe signals and operational counters."""
    if case_id not in ALLOWED_CASES:
        raise D04ArtifactError(D04ErrorCode.CASE_ID_INVALID)
    if (
        run.response.status != "grounding_failed"
        or run.response.reason_code != "grounding_quote_not_in_evidence"
        or not run.sanitized_grounding_signals
        or run.answer_usage is None
    ):
        raise D04ArtifactError(D04ErrorCode.DIAGNOSTIC_NOT_APPLICABLE)
    signals = run.sanitized_grounding_signals
    answer = run.answer_usage
    artifact: dict[str, object] = {
        "diagnostic_id": DIAGNOSTIC_ID,
        "version": DIAGNOSTIC_VERSION,
        "system_commit": _system_commit(root),
        **{key: _sha256(root / path) for key, path in SOURCE_PATHS.items()},
        "case_id": case_id,
        "status": "grounding_failed",
        "safe_error_code": "grounding_quote_not_in_evidence",
        "sanitized_root_cause_class": aggregate_classification(signals).value,
        "quote_failure_count": len(signals),
        "signal_sets": [item.as_dict() for item in signals],
        "retrieval_logical_api_calls": run.retrieval_usage.logical_api_calls,
        "retrieval_physical_attempts": run.retrieval_usage.physical_attempts,
        "answer_logical_api_calls": answer.logical_api_calls,
        "answer_physical_attempts": answer.physical_attempts,
        "retrieval_total_tokens": run.retrieval_usage.total_tokens,
        "answer_total_tokens": answer.total_tokens,
        "total_tokens": _optional_total(run.retrieval_usage.total_tokens, answer.total_tokens),
        "usage_report_sha256": _sha256(usage_report),
        "execution_completed": True,
        "privacy_scan_passed": True,
        "holdout_quality_decision": "NOT_APPLICABLE",
        "r03_created": False,
    }
    validate_d04_artifact(root, artifact)
    return artifact


def _load_schema(root: Path) -> dict[str, object]:
    try:
        value = json.loads((root / ARTIFACT_SCHEMA).read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        raise D04ArtifactError(D04ErrorCode.SCHEMA_INVALID) from None
    if not isinstance(value, dict):
        raise D04ArtifactError(D04ErrorCode.SCHEMA_INVALID)
    return cast(dict[str, object], value)


def validate_d04_artifact(root: Path, artifact: dict[str, object]) -> None:
    """Validate the Draft 2020-12 shape before persistence."""
    schema = _load_schema(root)
    try:
        Draft202012Validator.check_schema(cast(Any, schema))
        errors = list(Draft202012Validator(cast(Any, schema)).iter_errors(cast(Any, artifact)))  # pyright: ignore[reportUnknownMemberType]
    except Exception:
        raise D04ArtifactError(D04ErrorCode.SCHEMA_INVALID) from None
    if errors:
        raise D04ArtifactError(D04ErrorCode.ARTIFACT_INVALID)
    if artifact.get("sanitized_root_cause_class") not in {
        item.value for item in SanitizedRootCauseClass
    }:
        raise D04ArtifactError(D04ErrorCode.ARTIFACT_INVALID)


def _is_reparse_point(path: Path) -> bool:
    if not path.exists() and not path.is_symlink():
        return False
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    return path.is_symlink() or bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def resolve_d04_path(root: Path, candidate: str | Path) -> Path:
    """Resolve a JSON destination strictly under the ignored report directory."""
    try:
        project = root.resolve()
        base = (project / REPORT_DIRECTORY).resolve()
        raw = Path(candidate)
        destination = raw.resolve() if raw.is_absolute() else (project / raw).resolve()
        destination.relative_to(base)
    except OSError, RuntimeError, ValueError:
        raise D04ArtifactError(D04ErrorCode.DESTINATION_INVALID) from None
    if destination == base or destination.suffix.casefold() != ".json":
        raise D04ArtifactError(D04ErrorCode.DESTINATION_INVALID)
    try:
        current = destination.parent
        while current != project:
            if _is_reparse_point(current):
                raise D04ArtifactError(D04ErrorCode.DESTINATION_INVALID)
            current = current.parent
        if _is_reparse_point(destination):
            raise D04ArtifactError(D04ErrorCode.DESTINATION_INVALID)
    except OSError:
        raise D04ArtifactError(D04ErrorCode.DESTINATION_INVALID) from None
    return destination


def ensure_distinct_report_destinations(
    root: Path, usage_candidate: str | Path, diagnostic_candidate: str | Path
) -> tuple[Path, Path]:
    """Reject equivalent usage and D04 paths before any provider is constructed."""
    usage = resolve_d04_path(root, usage_candidate)
    diagnostic = resolve_d04_path(root, diagnostic_candidate)
    try:
        if usage.exists() and diagnostic.exists() and os.path.samefile(usage, diagnostic):
            raise D04ArtifactError(D04ErrorCode.DESTINATION_CONFLICT)
    except D04ArtifactError:
        raise
    except OSError:
        pass
    usage_key = os.path.normcase(os.path.abspath(str(usage))).casefold()
    diagnostic_key = os.path.normcase(os.path.abspath(str(diagnostic))).casefold()
    if usage_key == diagnostic_key:
        raise D04ArtifactError(D04ErrorCode.DESTINATION_CONFLICT)
    return usage, diagnostic


def ensure_d04_destination_available(root: Path, candidate: str | Path) -> Path:
    """Fail before provider calls when an explicit destination already exists."""
    destination = resolve_d04_path(root, candidate)
    try:
        if destination.exists():
            code = (
                D04ErrorCode.DESTINATION_INVALID
                if destination.is_dir()
                else D04ErrorCode.DESTINATION_EXISTS
            )
            raise D04ArtifactError(code)
    except D04ArtifactError:
        raise
    except OSError:
        raise D04ArtifactError(D04ErrorCode.DESTINATION_INVALID) from None
    return destination


def _cleanup_path(path: Path) -> bool:
    """Best-effort cleanup with one retry and no observable OS error text."""
    for _ in range(2):
        try:
            path.unlink(missing_ok=True)
            return True
        except OSError:
            continue
    return False


def write_d04_artifact(root: Path, candidate: str | Path, artifact: dict[str, object]) -> Path:
    """Publish complete UTF-8 JSON exclusively; never leave a partial destination."""
    validate_d04_artifact(root, artifact)
    destination = ensure_d04_destination_available(root, candidate)
    content = (json.dumps(artifact, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    descriptor: int | None = None
    temporary: Path | None = None
    published = False
    failure: D04ArtifactError | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if _is_reparse_point(destination.parent):
            raise D04ArtifactError(D04ErrorCode.DESTINATION_INVALID)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent, prefix=".d04-", suffix=".tmp"
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
        published = True
    except D04ArtifactError as exc:
        failure = exc
    except FileExistsError:
        failure = D04ArtifactError(D04ErrorCode.DESTINATION_EXISTS)
    except IsADirectoryError, NotADirectoryError:
        failure = D04ArtifactError(D04ErrorCode.DESTINATION_INVALID)
    except OSError, ValueError:
        failure = D04ArtifactError(D04ErrorCode.ARTIFACT_WRITE_FAILED)
    if descriptor is not None:
        try:
            os.close(descriptor)
        except OSError:
            if failure is None:
                failure = D04ArtifactError(D04ErrorCode.ARTIFACT_WRITE_FAILED)
    cleanup_succeeded = temporary is None or _cleanup_path(temporary)
    if not cleanup_succeeded and failure is None:
        if published:
            _cleanup_path(destination)
        failure = D04ArtifactError(D04ErrorCode.ARTIFACT_WRITE_FAILED)
    if failure is not None:
        raise failure from None
    return destination
