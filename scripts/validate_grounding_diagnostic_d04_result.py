"""Validate a future two-case D04 consolidation without requiring it in CI."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

from _project_bootstrap import bootstrap_project
from jsonschema import Draft202012Validator

SCHEMA_PATH = Path("evals/rag/grounding-diagnostic-d04-result.schema.json")


class D04ResultValidationError(RuntimeError):
    """Closed consolidated validation failure."""


def _load(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise D04ResultValidationError("invalid_json") from exc
    if not isinstance(value, dict):
        raise D04ResultValidationError("object_required")
    return cast(dict[str, object], value)


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise D04ResultValidationError("artifact_unavailable") from exc


def validate_result(
    root: Path,
    result: Path | None = None,
    *,
    artifact_p02: Path | None = None,
    artifact_p04: Path | None = None,
) -> dict[str, object] | None:
    """Validate schema only, or a synthetic/future result and its two artifacts."""
    schema = _load(root / SCHEMA_PATH)
    try:
        Draft202012Validator.check_schema(cast(Any, schema))
    except Exception as exc:
        raise D04ResultValidationError("schema_invalid") from exc
    if result is None:
        return None
    candidate = result if result.is_absolute() else root / result
    data = _load(candidate)
    if list(Draft202012Validator(cast(Any, schema)).iter_errors(cast(Any, data))):  # pyright: ignore[reportUnknownMemberType]
        raise D04ResultValidationError("schema_validation_failed")
    classifications = cast(dict[str, str], data["case_classifications"])
    counts = Counter(classifications.values())
    if data["classification_counts"] != dict(sorted(counts.items())):
        raise D04ResultValidationError("classification_counts_invalid")
    inconclusive = sorted(
        case for case, classification in classifications.items() if classification == "INCONCLUSIVE"
    )
    classified = sorted(case for case in classifications if case not in inconclusive)
    if sorted(cast(list[str], data["inconclusive_cases"])) != inconclusive:
        raise D04ResultValidationError("inconclusive_cases_invalid")
    if sorted(cast(list[str], data["classified_cases"])) != classified:
        raise D04ResultValidationError("classified_cases_invalid")
    if (artifact_p02 is None) != (artifact_p04 is None):
        raise D04ResultValidationError("both_artifacts_required")
    if artifact_p02 is not None and artifact_p04 is not None:
        try:
            from scripts.validate_grounding_diagnostic_d04 import validate_d04
        except ModuleNotFoundError:
            from validate_grounding_diagnostic_d04 import validate_d04
        artifact_paths = {"HOLD-P02": artifact_p02, "HOLD-P04": artifact_p04}
        hashes = cast(dict[str, str], data["artifact_sha256_by_case"])
        for case_id, path in artifact_paths.items():
            resolved = path if path.is_absolute() else root / path
            if hashes[case_id] != _sha256(resolved):
                raise D04ResultValidationError("artifact_hash_mismatch")
            artifact = validate_d04(root, resolved)
            if artifact is None or artifact.get("case_id") != case_id:
                raise D04ResultValidationError("artifact_case_invalid")
            if artifact.get("sanitized_root_cause_class") != classifications[case_id]:
                raise D04ResultValidationError("artifact_classification_invalid")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a future consolidated D04 result")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--artifact-p02", type=Path)
    parser.add_argument("--artifact-p04", type=Path)
    arguments = parser.parse_args()
    root = bootstrap_project() if arguments.root is None else arguments.root.resolve()
    try:
        validate_result(
            root,
            arguments.result,
            artifact_p02=arguments.artifact_p02,
            artifact_p04=arguments.artifact_p04,
        )
    except D04ResultValidationError:
        print("Grounding diagnostic D04 result validation failed.")
        return 1
    print("Grounding diagnostic D04 result validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
