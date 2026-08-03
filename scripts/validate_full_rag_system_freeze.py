"""Validate that the full RAG holdout system-under-test remains frozen."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, cast

from _project_bootstrap import bootstrap_project
from jsonschema import Draft202012Validator

_ATTEMPTS = {
    "r01": (
        Path("evals/rag/full-rag-holdout-r01-system-freeze.json"),
        Path("evals/rag/full-rag-holdout-system-freeze.schema.json"),
    ),
    "r02": (
        Path("evals/rag/full-rag-holdout-r02-system-freeze.json"),
        Path("evals/rag/full-rag-holdout-r02-system-freeze.schema.json"),
    ),
}


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


def _content_at_ref(root: Path, relative: str, source_ref: str) -> bytes:
    if not re.fullmatch(r"[a-f0-9]{40}", source_ref):
        raise SystemFreezeValidationError("source_ref_invalid")
    completed = subprocess.run(
        ["git", "show", f"{source_ref}:{Path(relative).as_posix()}"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise SystemFreezeValidationError("historical_file_unavailable")
    return completed.stdout


def validate_system_freeze(root: Path, attempt: str = "r01", source_ref: str | None = None) -> None:
    """Validate the closed freeze against the worktree or an explicit historical commit."""
    artifact_path, schema_path = _ATTEMPTS[attempt]
    artifact, schema = _load(root / artifact_path), _load(root / schema_path)
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
        content = (
            _content_at_ref(root, relative, source_ref)
            if source_ref is not None
            else candidate.read_bytes()
            if candidate.is_file()
            else b""
        )
        if hashlib.sha256(content).hexdigest() != expected:
            raise SystemFreezeValidationError("semantic_file_changed")


def main() -> int:
    """Run the offline full RAG system-freeze validation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--attempt", choices=tuple(_ATTEMPTS), default="r01")
    parser.add_argument("--source-ref")
    arguments = parser.parse_args()
    root = bootstrap_project() if arguments.root is None else arguments.root.resolve()
    try:
        validate_system_freeze(root, arguments.attempt, arguments.source_ref)
    except SystemFreezeValidationError:
        print("Full RAG system freeze validation failed.")
        return 1
    print("Full RAG system freeze validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
