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

_PROJECT_ROOT = bootstrap_project()

from nexodocs_ai.rag.full_rag_holdout_r03_integrity import (  # noqa: E402
    FileContent,
    R03IntegrityError,
    capture_bound_files,
    load_provenance_manifest,
    load_vector_fingerprint,
    verify_system_commit,
)

_ATTEMPTS = {
    "r01": (
        Path("evals/rag/full-rag-holdout-r01-system-freeze.json"),
        Path("evals/rag/full-rag-holdout-system-freeze.schema.json"),
    ),
    "r02": (
        Path("evals/rag/full-rag-holdout-r02-system-freeze.json"),
        Path("evals/rag/full-rag-holdout-r02-system-freeze.schema.json"),
    ),
    "r03": (
        Path("evals/rag/full-rag-holdout-r03-system-freeze.json"),
        Path("evals/rag/full-rag-holdout-r03-system-freeze.schema.json"),
    ),
}
_R03_HISTORICAL_EVIDENCE = (
    (
        "full-rag-holdout-r01-technical-incident",
        "evals/rag/full-rag-holdout-r01-technical-incident.json",
        "4e81e29fad9ad8a94efbcc9905594f9587514419aa39f8d423ba8e43ec0c2359",
    ),
    (
        "full-rag-holdout-r02-adjudication-v1",
        "evals/rag/full-rag-holdout-r02-adjudication-v1.json",
        "3df46e614b8343ea62124dc80d29f279fbbe360fe1887d5266db405332751ce3",
    ),
    (
        "grounding-diagnostic-d03-result-v1",
        "evals/rag/grounding-diagnostic-d03-result-v1.json",
        "62c54027590b00cf3222e3cedc54f4c6be52db7b5ef9b8bc48d355de8f94a64f",
    ),
    (
        "grounding-diagnostic-d04-result-v1",
        "evals/rag/grounding-diagnostic-d04-result-v1.json",
        "4ebabadd22d3d8ffce7a1923ee40a68b57c8d455c509f5ae3d8c41680f9c56d6",
    ),
    (
        "evaluation-oracle-corrections-v1",
        "evals/rag/evaluation-oracle-corrections-v1.json",
        "ed8b4f72d11584c64e1ca2eff7ec6825ae762516c3c5317b8ea33be425d36346",
    ),
)


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


def validate_system_freeze(
    root: Path,
    attempt: str = "r01",
    source_ref: str | None = None,
    *,
    verify_git_provenance: bool = True,
) -> None:
    """Validate the closed freeze against the worktree or an explicit historical commit."""
    artifact_path, schema_path = _ATTEMPTS[attempt]
    artifact, schema = _load(root / artifact_path), _load(root / schema_path)
    try:
        Draft202012Validator.check_schema(cast(Any, schema))
    except Exception as exc:
        raise SystemFreezeValidationError("schema_invalid") from exc
    if any(Draft202012Validator(cast(Any, schema)).iter_errors(cast(Any, artifact))):  # pyright: ignore[reportUnknownMemberType] - jsonschema is a dynamic boundary.
        raise SystemFreezeValidationError("schema_validation_failed")
    if attempt == "r03":
        raw_system_commit = artifact.get("system_commit")
        if not isinstance(raw_system_commit, str):
            raise SystemFreezeValidationError("system_commit_invalid")
        system_commit: str | None = raw_system_commit
    else:
        system_commit = None
    if source_ref is not None and system_commit is not None and source_ref != system_commit:
        raise SystemFreezeValidationError("source_ref_invalid")
    raw_historical = artifact.get("historical_evidence", [])
    if not isinstance(raw_historical, list):
        raise SystemFreezeValidationError("historical_evidence_invalid")
    historical = cast(list[object], raw_historical)
    expected_history = _R03_HISTORICAL_EVIDENCE if attempt == "r03" else ()
    if len(historical) != len(expected_history):
        raise SystemFreezeValidationError("historical_evidence_invalid")
    for raw_item, (artifact_id, expected_path, expected_hash) in zip(
        historical, expected_history, strict=True
    ):
        if not isinstance(raw_item, dict):
            raise SystemFreezeValidationError("historical_evidence_invalid")
        item = cast(dict[str, object], raw_item)
        if item != {"artifact_id": artifact_id, "path": expected_path, "sha256": expected_hash}:
            raise SystemFreezeValidationError("historical_evidence_invalid")
        candidate = (root / expected_path).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError as exc:
            raise SystemFreezeValidationError("path_invalid") from exc
        if (
            not candidate.is_file()
            or hashlib.sha256(candidate.read_bytes()).hexdigest() != expected_hash
        ):
            raise SystemFreezeValidationError("historical_evidence_changed")
    if attempt == "r03":
        provenance = artifact.get("provenance_bindings")
        environment = artifact.get("environment_provenance")
        vector_binding = artifact.get("vector_fingerprint_binding")
        if (
            not isinstance(provenance, dict)
            or not isinstance(environment, dict)
            or not isinstance(vector_binding, dict)
        ):
            raise SystemFreezeValidationError("provenance_binding_invalid")
        provenance_values = cast(dict[str, object], provenance)
        environment_values = cast(dict[str, object], environment)
        vector_values = cast(dict[str, object], vector_binding)
        expected_environment = {
            "system_commit": system_commit,
            "pyproject_path": "pyproject.toml",
            "pyproject_sha256": "0ab89ae7bf22d84727cc12960cf5e495d8c08433002dfdbb53993fd5a58fc9ae",
            "uv_lock_path": "uv.lock",
            "uv_lock_sha256": "bfb0921aedd880bcdac33f1d6838107525ed3fc73d4e5c6db2e0afff6a629f44",
            "execution_contract": "uv run --locked",
            "authority": "preflight_execution_snapshot",
            "python_requirement": "==3.14.*",
            "launcher_marker": "UV_RUN_RECURSION_DEPTH",
            "critical_packages": {
                "jsonschema": "4.26.0",
                "openai": "2.52.0",
                "qdrant-client": "1.18.0",
            },
        }
        if environment_values != expected_environment:
            raise SystemFreezeValidationError("environment_provenance_invalid")
        for path_field, digest_field in (
            ("pyproject_path", "pyproject_sha256"),
            ("uv_lock_path", "uv_lock_sha256"),
        ):
            relative = cast(str, environment_values[path_field])
            expected_digest = cast(str, environment_values[digest_field])
            try:
                content = (root / relative).read_bytes()
            except OSError as exc:
                raise SystemFreezeValidationError("environment_provenance_changed") from exc
            if hashlib.sha256(content).hexdigest() != expected_digest:
                raise SystemFreezeValidationError("environment_provenance_changed")
        manifest_specs = (
            (
                "system_runtime_manifest_path",
                "system_runtime_manifest_sha256",
                "full-rag-holdout-r03-system-runtime-manifest-v1",
            ),
            (
                "evaluation_harness_manifest_path",
                "evaluation_harness_manifest_sha256",
                "full-rag-holdout-r03-evaluation-harness-manifest-v1",
            ),
        )
        system_files: tuple[FileContent, ...] | None = None
        try:
            for path_field, digest_field, artifact_id in manifest_specs:
                relative = provenance_values.get(path_field)
                digest = provenance_values.get(digest_field)
                if not isinstance(relative, str) or not isinstance(digest, str):
                    raise SystemFreezeValidationError("provenance_binding_invalid")
                content = (root / relative).read_bytes()
                if hashlib.sha256(content).hexdigest() != digest:
                    raise SystemFreezeValidationError("provenance_binding_changed")
                manifest = load_provenance_manifest(content, artifact_id)
                captured = capture_bound_files(root, manifest)
                if path_field == "system_runtime_manifest_path":
                    system_files = captured
            vector_relative = vector_values.get("artifact_path")
            vector_digest = vector_values.get("artifact_sha256")
            if not isinstance(vector_relative, str) or not isinstance(vector_digest, str):
                raise SystemFreezeValidationError("vector_binding_invalid")
            vector_content = (root / vector_relative).read_bytes()
            vector = load_vector_fingerprint(vector_content)
            if (
                hashlib.sha256(vector_content).hexdigest() != vector_digest
                or vector.aggregate_sha256 != vector_values.get("aggregate_sha256")
                or vector.vector_encoding != vector_values.get("vector_encoding")
            ):
                raise SystemFreezeValidationError("vector_binding_changed")
            if system_commit is None or system_files is None:
                raise SystemFreezeValidationError("system_commit_invalid")
            if {item.path.as_posix() for item in system_files}.isdisjoint(
                {"pyproject.toml", "uv.lock"}
            ) or not {"pyproject.toml", "uv.lock"}.issubset(
                {item.path.as_posix() for item in system_files}
            ):
                raise SystemFreezeValidationError("environment_manifest_incomplete")
            if verify_git_provenance:
                verify_system_commit(root, system_commit, system_files)
        except (OSError, R03IntegrityError) as exc:
            raise SystemFreezeValidationError("provenance_validation_failed") from exc
        return
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
        if (
            system_commit is not None
            and hashlib.sha256(_content_at_ref(root, relative, system_commit)).hexdigest()
            != expected
        ):
            raise SystemFreezeValidationError("system_commit_mismatch")


def main() -> int:
    """Run the offline full RAG system-freeze validation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--attempt", choices=tuple(_ATTEMPTS), default="r01")
    parser.add_argument("--source-ref")
    arguments = parser.parse_args()
    root = _PROJECT_ROOT if arguments.root is None else arguments.root.resolve()
    try:
        validate_system_freeze(root, arguments.attempt, arguments.source_ref)
    except SystemFreezeValidationError:
        print("Full RAG system freeze validation failed.")
        return 1
    print("Full RAG system freeze validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
