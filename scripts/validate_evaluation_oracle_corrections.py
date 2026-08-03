"""Validate the sanitized evaluation-oracle correction artifact."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, cast

from _project_bootstrap import bootstrap_project
from jsonschema import Draft202012Validator

ARTIFACT = Path("evals/rag/evaluation-oracle-corrections-v1.json")
SCHEMA = Path("evals/rag/evaluation-oracle-corrections-v1.schema.json")
_FORBIDDEN_KEYS = frozenset(
    {
        "question",
        "query",
        "answer",
        "citation",
        "citations",
        "quote",
        "excerpt",
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


class OracleCorrectionValidationError(RuntimeError):
    """Raised when the versioned oracle correction violates its contract."""


def _load(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleCorrectionValidationError("invalid_json") from exc
    if not isinstance(value, dict):
        raise OracleCorrectionValidationError("object_required")
    return cast(dict[str, object], value)


def _validate_privacy(value: object) -> None:
    if isinstance(value, dict):
        for raw_key, nested in cast(dict[object, object], value).items():
            if not isinstance(raw_key, str) or raw_key.casefold() in _FORBIDDEN_KEYS:
                raise OracleCorrectionValidationError("prohibited_field")
            _validate_privacy(nested)
    elif isinstance(value, list):
        for nested in cast(list[object], value):
            _validate_privacy(nested)
    elif isinstance(value, str) and _FORBIDDEN_VALUES.search(value):
        raise OracleCorrectionValidationError("prohibited_content")


def validate_oracle_corrections(root: Path) -> None:
    """Validate the closed P05 correction and its sanitized boundary."""
    artifact, schema = _load(root / ARTIFACT), _load(root / SCHEMA)
    try:
        Draft202012Validator.check_schema(cast(Any, schema))
    except Exception as exc:
        raise OracleCorrectionValidationError("schema_invalid") from exc
    if any(Draft202012Validator(cast(Any, schema)).iter_errors(cast(Any, artifact))):  # pyright: ignore[reportUnknownMemberType] - jsonschema is a dynamic boundary.
        raise OracleCorrectionValidationError("schema_validation_failed")
    _validate_privacy(artifact)


def main() -> int:
    """Run the offline correction validation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    arguments = parser.parse_args()
    root = bootstrap_project() if arguments.root is None else arguments.root.resolve()
    try:
        validate_oracle_corrections(root)
    except OracleCorrectionValidationError:
        print("Evaluation oracle correction validation failed.")
        return 1
    print("Evaluation oracle correction validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
