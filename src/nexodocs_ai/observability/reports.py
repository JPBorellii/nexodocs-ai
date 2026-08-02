"""Closed operational reports that never contain prompts, content, or secrets."""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

from jsonschema import Draft202012Validator

from nexodocs_ai.rag.models import GenerationUsage, RagRunResult
from nexodocs_ai.retrieval.models import EmbeddingRunUsage, IndexingOperationResult

REPORT_SCHEMA_VERSION = "1.0"
REPORT_VERSION = "1.0.0"
REPORT_DIRECTORY = Path("data") / "run-reports"

_FORBIDDEN_KEYS = frozenset(
    {
        "openai_api_key",
        "qdrant_api_key",
        "headers",
        "prompt",
        "context",
        "query",
        "answer",
        "text",
        "vectors",
        "payload",
        "environment",
        "stack_trace",
        "traceback",
        "url",
        "path",
    }
)
_FORBIDDEN_VALUE = re.compile(
    r"(?i)(?:sk-[A-Za-z0-9_-]{16,}|authorization\s*:\s*bearer|https?://|"
    r"^[A-Za-z]:[\\/]|^\\\\|^/(?:home|users|var|tmp)/)"
)

_NULLABLE_INTEGER = {"type": ["integer", "null"], "minimum": 0}
_NULLABLE_STRING = {"type": ["string", "null"], "maxLength": 200}
_REQUEST_IDS_SCHEMA = {
    "type": "array",
    "items": {"type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$"},
    "uniqueItems": True,
}

USAGE_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "provider",
        "model",
        "logical_api_calls",
        "physical_attempts",
        "input_count",
        "prompt_tokens",
        "cached_input_tokens",
        "output_tokens",
        "total_tokens",
        "application_attempts",
        "transport_attempts_observable",
        "request_ids",
    ],
    "properties": {
        "provider": {"type": "string", "minLength": 1, "maxLength": 100},
        "model": {"type": "string", "minLength": 1, "maxLength": 200},
        "logical_api_calls": {"type": "integer", "minimum": 0},
        "physical_attempts": _NULLABLE_INTEGER,
        "input_count": {"type": "integer", "minimum": 0},
        "prompt_tokens": _NULLABLE_INTEGER,
        "cached_input_tokens": _NULLABLE_INTEGER,
        "output_tokens": _NULLABLE_INTEGER,
        "total_tokens": _NULLABLE_INTEGER,
        "application_attempts": {"type": "integer", "minimum": 0},
        "transport_attempts_observable": {"type": "boolean"},
        "request_ids": _REQUEST_IDS_SCHEMA,
    },
}

REPORT_SCHEMA: dict[str, object] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "report_version",
        "run_id",
        "timestamp_utc",
        "duration_ms",
        "operation",
        "provider",
        "model",
        "embedding_dimensions",
        "collection_name",
        "logical_api_calls",
        "physical_attempts",
        "input_count",
        "inserted_points",
        "reused_points",
        "obsolete_points",
        "prompt_tokens",
        "cached_input_tokens",
        "output_tokens",
        "total_tokens",
        "application_attempts",
        "status",
        "safe_error_code",
        "query_character_count",
        "request_ids",
        "retrieval_usage",
        "answer_usage",
    ],
    "properties": {
        "schema_version": {"const": REPORT_SCHEMA_VERSION},
        "report_version": {"const": REPORT_VERSION},
        "run_id": {"type": "string", "pattern": "^[0-9a-f-]{36}$"},
        "timestamp_utc": {"type": "string", "format": "date-time"},
        "duration_ms": {"type": "integer", "minimum": 0},
        "operation": {"enum": ["index", "search", "answer"]},
        "provider": {"type": "string", "minLength": 1, "maxLength": 100},
        "model": {"type": "string", "minLength": 1, "maxLength": 200},
        "embedding_dimensions": _NULLABLE_INTEGER,
        "collection_name": _NULLABLE_STRING,
        "logical_api_calls": {"type": "integer", "minimum": 0},
        "physical_attempts": _NULLABLE_INTEGER,
        "input_count": {"type": "integer", "minimum": 0},
        "inserted_points": _NULLABLE_INTEGER,
        "reused_points": _NULLABLE_INTEGER,
        "obsolete_points": _NULLABLE_INTEGER,
        "prompt_tokens": _NULLABLE_INTEGER,
        "cached_input_tokens": _NULLABLE_INTEGER,
        "output_tokens": _NULLABLE_INTEGER,
        "total_tokens": _NULLABLE_INTEGER,
        "application_attempts": {"type": "integer", "minimum": 0},
        "status": {"type": "string", "minLength": 1, "maxLength": 100},
        "safe_error_code": _NULLABLE_STRING,
        "query_character_count": _NULLABLE_INTEGER,
        "request_ids": _REQUEST_IDS_SCHEMA,
        "retrieval_usage": {"anyOf": [USAGE_SCHEMA, {"type": "null"}]},
        "answer_usage": {"anyOf": [USAGE_SCHEMA, {"type": "null"}]},
    },
}


class ReportError(ValueError):
    """Raised when a local operational report violates its closed boundary."""


@dataclass(frozen=True)
class ApiUsageReport:
    provider: str
    model: str
    logical_api_calls: int
    physical_attempts: int | None
    input_count: int
    prompt_tokens: int | None
    cached_input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    application_attempts: int
    transport_attempts_observable: bool
    request_ids: tuple[str, ...]


@dataclass(frozen=True)
class RealExecutionReport:
    schema_version: str
    report_version: str
    run_id: str
    timestamp_utc: str
    duration_ms: int
    operation: Literal["index", "search", "answer"]
    provider: str
    model: str
    embedding_dimensions: int | None
    collection_name: str | None
    logical_api_calls: int
    physical_attempts: int | None
    input_count: int
    inserted_points: int | None
    reused_points: int | None
    obsolete_points: int | None
    prompt_tokens: int | None
    cached_input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    application_attempts: int
    status: str
    safe_error_code: str | None
    query_character_count: int | None
    request_ids: tuple[str, ...]
    retrieval_usage: ApiUsageReport | None
    answer_usage: ApiUsageReport | None

    def as_dict(self) -> dict[str, object]:
        """Return the closed JSON representation."""
        return cast(
            dict[str, object],
            json.loads(json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)),
        )


def _optional_sum(values: tuple[tuple[int | None, int], ...]) -> int | None:
    available: list[int] = []
    for value, calls in values:
        if value is None:
            if calls:
                return None
            continue
        available.append(value)
    return sum(available)


def _physical_sum(values: tuple[tuple[int | None, int], ...]) -> int | None:
    return _optional_sum(values)


def embedding_api_usage(usage: EmbeddingRunUsage) -> ApiUsageReport:
    """Convert embedding usage to the common safe report shape."""
    return ApiUsageReport(
        usage.provider,
        usage.model,
        usage.logical_api_calls,
        usage.physical_attempts,
        usage.input_count,
        usage.prompt_tokens,
        None,
        None,
        usage.total_tokens,
        0,
        usage.transport_attempts_observable,
        tuple(batch.request_id for batch in usage.batches if batch.request_id is not None),
    )


def generation_api_usage(
    usage: GenerationUsage | None, provider: str, model: str, application_attempts: int
) -> ApiUsageReport:
    """Represent generation usage, including explicit zero-call fallbacks."""
    if usage is None:
        return ApiUsageReport(provider, model, 0, 0, 0, None, None, None, None, 0, True, ())
    return ApiUsageReport(
        usage.provider,
        usage.model,
        usage.logical_api_calls,
        usage.physical_attempts,
        1,
        usage.input_tokens,
        usage.cached_input_tokens,
        usage.output_tokens,
        usage.total_tokens,
        application_attempts,
        usage.transport_attempts_observable,
        () if usage.request_id is None else (usage.request_id,),
    )


def _base_report(
    *,
    duration_ms: int,
    operation: Literal["index", "search", "answer"],
    provider: str,
    model: str,
    embedding_dimensions: int | None,
    collection_name: str | None,
    usage: ApiUsageReport,
    status: str,
    safe_error_code: str | None = None,
    query_character_count: int | None = None,
    inserted_points: int | None = None,
    reused_points: int | None = None,
    obsolete_points: int | None = None,
    retrieval_usage: ApiUsageReport | None = None,
    answer_usage: ApiUsageReport | None = None,
) -> RealExecutionReport:
    return RealExecutionReport(
        REPORT_SCHEMA_VERSION,
        REPORT_VERSION,
        str(uuid4()),
        datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        max(duration_ms, 0),
        operation,
        provider,
        model,
        embedding_dimensions,
        collection_name,
        usage.logical_api_calls,
        usage.physical_attempts,
        usage.input_count,
        inserted_points,
        reused_points,
        obsolete_points,
        usage.prompt_tokens,
        usage.cached_input_tokens,
        usage.output_tokens,
        usage.total_tokens,
        usage.application_attempts,
        status,
        safe_error_code,
        query_character_count,
        usage.request_ids,
        retrieval_usage,
        answer_usage,
    )


def index_report(
    operation: IndexingOperationResult,
    collection_name: str,
    duration_ms: int,
) -> RealExecutionReport:
    usage = embedding_api_usage(operation.embedding_usage)
    result = operation.result
    return _base_report(
        duration_ms=duration_ms,
        operation="index",
        provider=usage.provider,
        model=usage.model,
        embedding_dimensions=operation.embedding_usage.dimensions,
        collection_name=collection_name,
        usage=usage,
        status="succeeded",
        inserted_points=result.inserted_or_updated,
        reused_points=result.reused,
        obsolete_points=result.obsolete,
    )


def search_report(
    usage_value: EmbeddingRunUsage,
    collection_name: str,
    query_character_count: int,
    status: str,
    duration_ms: int,
) -> RealExecutionReport:
    usage = embedding_api_usage(usage_value)
    return _base_report(
        duration_ms=duration_ms,
        operation="search",
        provider=usage.provider,
        model=usage.model,
        embedding_dimensions=usage_value.dimensions,
        collection_name=collection_name,
        usage=usage,
        status=status,
        query_character_count=query_character_count,
    )


def answer_report(
    run: RagRunResult,
    answer_provider: str,
    answer_model: str,
    collection_name: str,
    query_character_count: int,
    duration_ms: int,
) -> RealExecutionReport:
    retrieval = embedding_api_usage(run.retrieval_usage)
    answer = generation_api_usage(
        run.answer_usage, answer_provider, answer_model, run.application_attempts
    )
    combined = ApiUsageReport(
        answer.provider,
        answer.model,
        retrieval.logical_api_calls + answer.logical_api_calls,
        _physical_sum(
            (
                (retrieval.physical_attempts, retrieval.logical_api_calls),
                (answer.physical_attempts, answer.logical_api_calls),
            )
        ),
        retrieval.input_count + answer.input_count,
        _optional_sum(
            (
                (retrieval.prompt_tokens, retrieval.logical_api_calls),
                (answer.prompt_tokens, answer.logical_api_calls),
            )
        ),
        answer.cached_input_tokens,
        answer.output_tokens,
        _optional_sum(
            (
                (retrieval.total_tokens, retrieval.logical_api_calls),
                (answer.total_tokens, answer.logical_api_calls),
            )
        ),
        run.application_attempts,
        retrieval.transport_attempts_observable and answer.transport_attempts_observable,
        (*retrieval.request_ids, *answer.request_ids),
    )
    return _base_report(
        duration_ms=duration_ms,
        operation="answer",
        provider=answer.provider,
        model=answer.model,
        embedding_dimensions=run.retrieval_usage.dimensions,
        collection_name=collection_name,
        usage=combined,
        status=run.response.status,
        safe_error_code=run.response.reason_code,
        query_character_count=query_character_count,
        retrieval_usage=retrieval,
        answer_usage=answer,
    )


def validate_report(report: RealExecutionReport | dict[str, object]) -> dict[str, object]:
    """Validate schema and reject prohibited key/value patterns."""
    data = report.as_dict() if isinstance(report, RealExecutionReport) else report
    errors = sorted(
        Draft202012Validator(REPORT_SCHEMA).iter_errors(cast(Any, data)),  # pyright: ignore[reportUnknownMemberType] - jsonschema is a dynamic boundary.
        key=lambda item: list(item.path),
    )
    if errors:
        raise ReportError(f"Relat\u00f3rio operacional inv\u00e1lido: {errors[0].message}")

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for raw_key, nested in cast(dict[object, object], value).items():
                if not isinstance(raw_key, str):
                    raise ReportError("Chave n\u00e3o textual proibida no relat\u00f3rio")
                key = raw_key
                if key.casefold() in _FORBIDDEN_KEYS:
                    raise ReportError(f"Campo proibido no relat\u00f3rio: {key}")
                visit(nested)
        elif isinstance(value, list | tuple):
            for nested in cast(list[object] | tuple[object, ...], value):
                visit(nested)
        elif isinstance(value, str) and _FORBIDDEN_VALUE.search(value):
            raise ReportError("Valor sens\u00edvel ou ambiental proibido no relat\u00f3rio")

    visit(data)
    return data


def _is_reparse_point(path: Path) -> bool:
    if not path.exists() and not path.is_symlink():
        return False
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    return path.is_symlink() or bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def resolve_report_path(project: Path, candidate: str | Path) -> Path:
    """Resolve a report path strictly below data/run-reports."""
    root = project.resolve()
    base = (root / REPORT_DIRECTORY).resolve()
    raw = Path(candidate)
    destination = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    try:
        destination.relative_to(base)
    except ValueError as exc:
        raise ReportError("--usage-report deve permanecer em data/run-reports") from exc
    if destination == base or destination.suffix.casefold() != ".json":
        raise ReportError("--usage-report deve apontar para um arquivo JSON")
    chain = [root / "data", root / REPORT_DIRECTORY]
    current = destination.parent
    while current != base and current != root:
        chain.append(current)
        current = current.parent
    chain.append(destination)
    if any(_is_reparse_point(path) for path in chain):
        raise ReportError(
            "Symlink ou reparse point n\u00e3o permitido no caminho do relat\u00f3rio"
        )
    return destination


def write_report(
    project: Path,
    candidate: str | Path,
    report: RealExecutionReport,
    *,
    overwrite: bool = False,
) -> Path:
    """Validate and atomically publish one UTF-8 report below the ignored directory."""
    destination = resolve_report_path(project, candidate)
    data = validate_report(report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _is_reparse_point(destination.parent):
        raise ReportError("Diret\u00f3rio de relat\u00f3rio n\u00e3o pode ser reparse point")
    if destination.exists() and not overwrite:
        raise ReportError("Relat\u00f3rio j\u00e1 existe; use sobrescrita expl\u00edcita")
    content = (json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=".usage-", suffix=".tmp", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary_path, destination)
        else:
            os.link(temporary_path, destination)
            temporary_path.unlink()
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination
