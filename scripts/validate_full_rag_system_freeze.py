"""Validate that the full RAG holdout system-under-test remains frozen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from _project_bootstrap import bootstrap_project
from jsonschema import Draft202012Validator

ARTIFACT = Path("evals/rag/full-rag-holdout-r01-system-freeze.json")
SCHEMA = Path("evals/rag/full-rag-holdout-system-freeze.schema.json")


class SystemFreezeValidationError(RuntimeError):
    """Raised when a frozen semantic file differs from its recorded digest."""


def _load(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemFreezeValidationError("invalid_json") from exc
    if not isinstance(value, dict):
        raise SystemFreezeValidationError("object_required")
    return cast(dict[str, object], value)


def validate_system_freeze(root: Path) -> None:
    """Validate the closed freeze artifact and every SHA-256 binding."""
    artifact, schema = _load(root / ARTIFACT), _load(root / SCHEMA)
    try:
        Draft202012Validator.check_schema(cast(Any, schema))
    except Exception as exc:
        raise SystemFreezeValidationError("schema_invalid") from exc
    if any(Draft202012Validator(cast(Any, schema)).iter_errors(cast(Any, artifact))):  # pyright: ignore[reportUnknownMemberType] - jsonschema is a dynamic boundary.
        raise SystemFreezeValidationError("schema_validation_failed")
    files = artifact.get("frozen_files")
    if not isinstance(files, list):
        raise SystemFreezeValidationError("frozen_files_required")
    paths: set[str] = set()
    for raw_item in cast(list[object], files):
        if not isinstance(raw_item, dict):
            raise SystemFreezeValidationError("file_entry_invalid")
        item = cast(dict[str, object], raw_item)
        relative, expected = item.get("path"), item.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str) or relative in paths:
            raise SystemFreezeValidationError("file_entry_invalid")
        paths.add(relative)
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError as exc:
            raise SystemFreezeValidationError("path_invalid") from exc
        if (
            not candidate.is_file()
            or hashlib.sha256(candidate.read_bytes()).hexdigest() != expected
        ):
            raise SystemFreezeValidationError("semantic_file_changed")


def main() -> int:
    """Run the offline full RAG system-freeze validation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    arguments = parser.parse_args()
    root = bootstrap_project() if arguments.root is None else arguments.root.resolve()
    try:
        validate_system_freeze(root)
    except SystemFreezeValidationError:
        print("Full RAG system freeze validation failed.")
        return 1
    print("Full RAG system freeze validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
