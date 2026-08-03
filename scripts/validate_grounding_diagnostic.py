"""Validate the closed D03 grounding-diagnostic contract without external services."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, cast

from _project_bootstrap import bootstrap_project
from jsonschema import Draft202012Validator

SCHEMA = Path("evals/rag/grounding-diagnostic-d03.schema.json")
_FORBIDDEN_KEYS = frozenset(
    {
        "question",
        "query",
        "answer",
        "citation",
        "citations",
        "document",
        "quote",
        "context",
        "prompt",
        "traceback",
        "request_id",
        "timestamp",
        "path",
    }
)
_FORBIDDEN_VALUES = re.compile(
    r"(?i)(?:sk-[A-Za-z0-9_-]{8,}|https?://|^[A-Za-z]:[\\/]|^\\\\|"
    r"^/(?:home|users|var|tmp)/)"
)


class GroundingDiagnosticValidationError(RuntimeError):
    """Raised when the diagnostic schema or a future artifact is unsafe."""


def _load(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GroundingDiagnosticValidationError("invalid_json") from exc
    if not isinstance(value, dict):
        raise GroundingDiagnosticValidationError("object_required")
    return cast(dict[str, object], value)


def _validate_privacy(value: object) -> None:
    if isinstance(value, dict):
        for raw_key, nested in cast(dict[object, object], value).items():
            if not isinstance(raw_key, str) or raw_key.casefold() in _FORBIDDEN_KEYS:
                raise GroundingDiagnosticValidationError("prohibited_field")
            _validate_privacy(nested)
    elif isinstance(value, list):
        for nested in cast(list[object], value):
            _validate_privacy(nested)
    elif isinstance(value, str) and _FORBIDDEN_VALUES.search(value):
        raise GroundingDiagnosticValidationError("prohibited_content")


def validate_grounding_diagnostic(root: Path, artifact: Path | None = None) -> None:
    """Validate the schema and, when supplied, one future D03 artifact."""
    schema = _load(root / SCHEMA)
    try:
        Draft202012Validator.check_schema(cast(Any, schema))
    except Exception as exc:
        raise GroundingDiagnosticValidationError("schema_invalid") from exc

    bootstrap_project()
    from nexodocs_ai.rag.grounding_diagnostics import GROUNDING_SAFE_ERROR_CODES

    properties = schema.get("properties")
    if not isinstance(properties, dict):
        raise GroundingDiagnosticValidationError("schema_properties_invalid")
    safe_code = cast(dict[str, object], properties).get("safe_error_code")
    raw_codes = (
        cast(dict[str, object], safe_code).get("enum") if isinstance(safe_code, dict) else None
    )
    if not isinstance(raw_codes, list) or set(cast(list[object], raw_codes)) != set(
        GROUNDING_SAFE_ERROR_CODES
    ):
        raise GroundingDiagnosticValidationError("safe_error_codes_invalid")
    if artifact is None:
        return
    candidate = artifact if artifact.is_absolute() else root / artifact
    data = _load(candidate)
    if any(Draft202012Validator(cast(Any, schema)).iter_errors(cast(Any, data))):  # pyright: ignore[reportUnknownMemberType] - jsonschema is a dynamic boundary.
        raise GroundingDiagnosticValidationError("schema_validation_failed")
    _validate_privacy(data)


def main() -> int:
    """Run the offline D03 contract validation without executing a diagnostic."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--artifact", type=Path)
    arguments = parser.parse_args()
    root = bootstrap_project() if arguments.root is None else arguments.root.resolve()
    try:
        validate_grounding_diagnostic(root, arguments.artifact)
    except GroundingDiagnosticValidationError:
        print("Grounding diagnostic D03 validation failed.")
        return 1
    print("Grounding diagnostic D03 validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
