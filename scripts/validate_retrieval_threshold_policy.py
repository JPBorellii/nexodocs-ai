"""Validate the frozen retrieval-threshold policy without runtime services."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, cast

from jsonschema import Draft202012Validator


class PolicyValidationError(RuntimeError):
    """Raised when the frozen policy or its bound artifacts are inconsistent."""


FORBIDDEN_FIELD_PARTS = frozenset(
    {
        "key",
        "url",
        "path",
        "timestamp",
        "query",
        "text",
        "context",
        "answer",
        "prompt",
        "vector",
        "payload",
        "header",
        "request_id",
    }
)
ABSOLUTE_PATH = re.compile(r"(?:^[A-Za-z]:[\\/]|^/|^\\\\)")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PolicyValidationError(f"Invalid JSON artifact: {path.name}") from exc
    if not isinstance(value, dict):
        raise PolicyValidationError(f"JSON artifact must be an object: {path.name}")
    return cast(dict[str, object], value)


def _validate_schema(schema: dict[str, object], policy: dict[str, object]) -> None:
    Draft202012Validator.check_schema(cast(Any, schema))
    errors = sorted(
        Draft202012Validator(cast(Any, schema)).iter_errors(cast(Any, policy)),  # pyright: ignore[reportUnknownMemberType] - jsonschema is a dynamic boundary.
        key=lambda item: list(item.path),
    )
    if errors:
        raise PolicyValidationError(f"Policy schema validation failed: {errors[0].message}")


def _validate_safe_content(value: object) -> None:
    if isinstance(value, dict):
        for key, child in cast(dict[str, object], value).items():
            lowered = key.lower()
            if any(part in lowered for part in FORBIDDEN_FIELD_PARTS) or lowered.endswith("_at"):
                raise PolicyValidationError("Policy contains a prohibited field")
            _validate_safe_content(child)
    elif isinstance(value, list):
        for child in cast(list[object], value):
            _validate_safe_content(child)
    elif isinstance(value, str) and ("://" in value or ABSOLUTE_PATH.search(value)):
        raise PolicyValidationError("Policy contains a prohibited location value")


def _decimal(policy: dict[str, object], key: str) -> Decimal:
    value = policy.get(key)
    if not isinstance(value, int | float):
        raise PolicyValidationError(f"Policy field must be numeric: {key}")
    return Decimal(str(value))


def _validate_math(policy: dict[str, object]) -> None:
    threshold = _decimal(policy, "score_threshold")
    positive = _decimal(policy, "positive_floor")
    negative = _decimal(policy, "negative_ceiling")
    gap = positive - negative
    midpoint = (positive + negative) / Decimal("2")
    candidate = midpoint.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
    operational = candidate.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    if not negative < threshold < positive:
        raise PolicyValidationError(
            "Threshold must be strictly between negative and positive bounds"
        )
    if gap < Decimal("0.03"):
        raise PolicyValidationError("Separation gap must be at least 0.03")
    if positive - threshold <= 0 or threshold - negative <= 0:
        raise PolicyValidationError("Policy margins must be positive")
    expected = {
        "separation_gap": gap,
        "positive_margin": positive - threshold,
        "negative_margin": threshold - negative,
        "candidate_threshold": candidate,
        "score_threshold": operational,
    }
    for key, expected_value in expected.items():
        if _decimal(policy, key) != expected_value:
            raise PolicyValidationError(f"Policy calculation is inconsistent: {key}")


def _validate_hashes(root: Path, policy: dict[str, object]) -> None:
    paths = {
        "index_manifest_sha256": root / "knowledge_base" / "index" / "index-manifest.json",
        "chunks_sha256": root / "knowledge_base" / "processed" / "chunks.jsonl",
        "processed_manifest_sha256": root / "knowledge_base" / "processed" / "manifest.json",
        "index_plan_sha256": root / "knowledge_base" / "index" / "index-plan.json",
    }
    for key, path in paths.items():
        expected = policy.get(key)
        if not isinstance(expected, str) or _sha256(path) != expected:
            raise PolicyValidationError(f"Artifact hash mismatch: {path.name}")


def _validate_manifest(root: Path, policy: dict[str, object]) -> None:
    index_directory = root / "knowledge_base" / "index"
    manifest = _load_object(index_directory / "index-manifest.json")
    schema = _load_object(index_directory / "index-manifest.schema.json")
    _validate_schema(schema, manifest)
    keys = (
        "embedding_provider",
        "embedding_model",
        "embedding_dimensions",
        "collection_name",
        "distance",
        "total_points",
    )
    if any(manifest.get(key) != policy.get(key) for key in keys):
        raise PolicyValidationError("Index manifest metadata is incompatible with policy")


def validate_policy(root: Path) -> None:
    """Validate the policy, mathematical invariants, hashes, and index manifest."""
    index_directory = root / "knowledge_base" / "index"
    policy = _load_object(index_directory / "retrieval-threshold-policy.json")
    schema = _load_object(index_directory / "retrieval-threshold-policy.schema.json")
    _validate_safe_content(policy)
    _validate_schema(schema, policy)
    _validate_math(policy)
    _validate_hashes(root, policy)
    _validate_manifest(root, policy)


def main() -> int:
    """Run the offline frozen-policy validation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    arguments = parser.parse_args()
    try:
        validate_policy(arguments.root.resolve())
    except PolicyValidationError as exc:
        print(f"Retrieval threshold policy validation failed: {exc}")
        return 1
    print("Retrieval threshold policy validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
