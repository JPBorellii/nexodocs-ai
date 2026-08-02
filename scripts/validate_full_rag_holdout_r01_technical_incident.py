"""Validate the sanitized technical incident for full RAG holdout r01 offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, cast

from _project_bootstrap import bootstrap_project
from jsonschema import Draft202012Validator

ARTIFACT = Path("evals/rag/full-rag-holdout-r01-technical-incident.json")
SCHEMA = Path("evals/rag/full-rag-holdout-r01-technical-incident.schema.json")
_HASH_BINDINGS = {
    "fixture_sha256": Path("evals/rag/full-rag-holdout-r01-cases.json"),
    "system_freeze_sha256": Path("evals/rag/full-rag-holdout-r01-system-freeze.json"),
    "threshold_policy_sha256": Path("knowledge_base/index/retrieval-threshold-policy.json"),
    "index_manifest_sha256": Path("knowledge_base/index/index-manifest.json"),
}
_FORBIDDEN_KEYS = frozenset(
    {"query", "question", "answer", "traceback", "message", "request_id", "timestamp", "path"}
)
_FORBIDDEN_VALUES = re.compile(
    r"(?i)(?:sk-[a-z0-9_-]{16,}|https?://|^[a-z]:[\\/]|^\\\\|^/(?:home|users|var|tmp)/)"
)


class TechnicalIncidentValidationError(RuntimeError):
    """Raised when the incident is malformed, unsafe, or loses its bindings."""


def _load(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TechnicalIncidentValidationError("invalid_json") from exc
    if not isinstance(value, dict):
        raise TechnicalIncidentValidationError("object_required")
    return cast(dict[str, object], value)


def _validate_safe(value: object) -> None:
    if isinstance(value, dict):
        for key, nested in cast(dict[object, object], value).items():
            if not isinstance(key, str) or key.casefold() in _FORBIDDEN_KEYS:
                raise TechnicalIncidentValidationError("prohibited_field")
            _validate_safe(nested)
    elif isinstance(value, list):
        for nested in cast(list[object], value):
            _validate_safe(nested)
    elif isinstance(value, str) and _FORBIDDEN_VALUES.search(value):
        raise TechnicalIncidentValidationError("prohibited_content")


def validate_incident(root: Path) -> None:
    """Validate the closed incident and its immutable versioned hash bindings."""
    artifact, schema = _load(root / ARTIFACT), _load(root / SCHEMA)
    try:
        Draft202012Validator.check_schema(cast(Any, schema))
    except Exception as exc:
        raise TechnicalIncidentValidationError("schema_invalid") from exc
    if any(Draft202012Validator(cast(Any, schema)).iter_errors(cast(Any, artifact))):  # pyright: ignore[reportUnknownMemberType] - jsonschema is a dynamic boundary.
        raise TechnicalIncidentValidationError("schema_validation_failed")
    _validate_safe(artifact)
    for field, relative in _HASH_BINDINGS.items():
        if artifact.get(field) != hashlib.sha256((root / relative).read_bytes()).hexdigest():
            raise TechnicalIncidentValidationError("hash_binding_invalid")


def main() -> int:
    """Run the technical incident validator without external services."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    arguments = parser.parse_args()
    root = bootstrap_project() if arguments.root is None else arguments.root.resolve()
    try:
        validate_incident(root)
    except TechnicalIncidentValidationError:
        print("Full RAG holdout r01 technical incident validation failed.")
        return 1
    print("Full RAG holdout r01 technical incident validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
