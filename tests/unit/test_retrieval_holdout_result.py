"""Tests for the versioned, sanitized retrieval holdout failure evidence."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = ROOT / "scripts" / "validate_retrieval_holdout_result.py"
ARTIFACT = Path("evals/retrieval/holdout-r01-result.json")


def _copy_fixture(destination: Path) -> None:
    files = (
        "evals/retrieval/holdout-r01-result.json",
        "evals/retrieval/holdout-result.schema.json",
        "knowledge_base/index/retrieval-threshold-policy.json",
        "knowledge_base/index/index-manifest.json",
        "knowledge_base/index/index-plan.json",
        "knowledge_base/processed/chunks.jsonl",
        "knowledge_base/processed/manifest.json",
    )
    for relative in files:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)


def _artifact(root: Path) -> dict[str, object]:
    return json.loads((root / ARTIFACT).read_text(encoding="utf-8"))


def _write_artifact(root: Path, artifact: dict[str, object]) -> None:
    (root / ARTIFACT).write_text(json.dumps(artifact), encoding="utf-8")


def _validate(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATOR), "--root", str(root)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_holdout_failure_artifact_is_valid_and_preserves_all_metrics(tmp_path: Path) -> None:
    _copy_fixture(tmp_path)
    completed = _validate(tmp_path)
    artifact = _artifact(tmp_path)

    assert completed.returncode == 0
    assert completed.stdout == "Retrieval holdout result validation passed.\n"
    assert artifact["retrieval_decision"] == "HOLDOUT_FAILED"
    assert artifact["score_threshold"] == pytest.approx(0.46)
    assert artifact["threshold_frozen"] is True
    assert artifact["threshold_changed"] is False
    assert artifact["failed_case_ids"] == ["HOLD-N01", "HOLD-N02", "HOLD-N04"]
    assert artifact["supported_recall_at_5"] == pytest.approx(1.0)
    assert artifact["supported_hit_rate_at_5"] == pytest.approx(1.0)
    assert artifact["supported_top1_accuracy"] == pytest.approx(5 / 6)
    assert artifact["supported_mrr"] == pytest.approx(11 / 12)
    assert artifact["unsupported_fallback_accuracy"] == pytest.approx(3 / 6)
    assert artifact["status_accuracy"] == pytest.approx(9 / 12)
    assert artifact["logical_api_calls"] == 12
    assert artifact["physical_attempts"] == 12
    assert artifact["generation_api_calls"] == 0
    assert artifact["threshold_policy_sha256"] == (
        "7dc913e6b9469ce45a2f1e6c0b6ecc6e9ee3caa7e52da49e36b971c86527eff7"
    )
    assert artifact["index_manifest_sha256"] == (
        "cb9c47e0ef4a62771b15894bcfcbfbb6966af588169ebc17427ed00f14906b79"
    )
    assert artifact["holdout_scalar_separation_gap"] == pytest.approx(-0.00206978)
    assert artifact["threshold_only_separation_possible"] is False


def test_holdout_schema_is_closed(tmp_path: Path) -> None:
    _copy_fixture(tmp_path)
    artifact = _artifact(tmp_path)
    artifact["unexpected"] = "rejected"
    _write_artifact(tmp_path, artifact)

    assert _validate(tmp_path).returncode != 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("query", "sanitized content is forbidden"),
        ("text", "sanitized content is forbidden"),
        ("timestamp_utc", "2026-08-02T00:00:00Z"),
        ("evidence_path", "C:\\restricted\\artifact.json"),
    ],
)
def test_holdout_rejects_prohibited_content(tmp_path: Path, field: str, value: object) -> None:
    _copy_fixture(tmp_path)
    artifact = _artifact(tmp_path)
    artifact[field] = value
    _write_artifact(tmp_path, artifact)

    assert _validate(tmp_path).returncode != 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("retrieval_decision", "HOLDOUT_PASSED"),
        ("score_threshold", 0.53),
        ("threshold_policy_sha256", "0" * 64),
        ("supported_quality_passed", False),
    ],
)
def test_holdout_rejects_a_changed_historical_result(
    tmp_path: Path, field: str, value: object
) -> None:
    _copy_fixture(tmp_path)
    artifact = _artifact(tmp_path)
    artifact[field] = value
    _write_artifact(tmp_path, artifact)

    assert _validate(tmp_path).returncode != 0


def test_holdout_rejects_inconsistent_case_math_and_manifest(tmp_path: Path) -> None:
    _copy_fixture(tmp_path)
    artifact = _artifact(tmp_path)
    outcomes = artifact["case_outcomes"]
    assert isinstance(outcomes, list)
    assert isinstance(outcomes[3], dict)
    outcomes[3]["expected_document_score"] = 0.53
    _write_artifact(tmp_path, artifact)

    assert _validate(tmp_path).returncode != 0

    _copy_fixture(tmp_path)
    (tmp_path / "knowledge_base/index/index-manifest.json").write_text("{}\n", encoding="utf-8")

    assert _validate(tmp_path).returncode != 0
