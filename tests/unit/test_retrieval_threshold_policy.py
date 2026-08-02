"""Tests for the frozen, offline retrieval-threshold policy."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = ROOT / "scripts" / "validate_retrieval_threshold_policy.py"


def _copy_policy_fixture(destination: Path) -> None:
    files = (
        "knowledge_base/index/retrieval-threshold-policy.json",
        "knowledge_base/index/retrieval-threshold-policy.schema.json",
        "knowledge_base/index/index-manifest.json",
        "knowledge_base/index/index-manifest.schema.json",
        "knowledge_base/index/index-plan.json",
        "knowledge_base/processed/chunks.jsonl",
        "knowledge_base/processed/manifest.json",
    )
    for relative in files:
        source = ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _policy_path(root: Path) -> Path:
    return root / "knowledge_base" / "index" / "retrieval-threshold-policy.json"


def _policy(root: Path) -> dict[str, object]:
    return json.loads(_policy_path(root).read_text(encoding="utf-8"))


def _write_policy(root: Path, policy: dict[str, object]) -> None:
    _policy_path(root).write_text(json.dumps(policy), encoding="utf-8")


def _validate(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATOR), "--root", str(root)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_frozen_policy_is_valid_and_has_expected_calibration_values(tmp_path: Path) -> None:
    _copy_policy_fixture(tmp_path)

    completed = _validate(tmp_path)
    policy = _policy(tmp_path)

    assert completed.returncode == 0
    assert completed.stdout == "Retrieval threshold policy validation passed.\n"
    assert policy["threshold_frozen"] is True
    assert policy["score_threshold"] == pytest.approx(0.46)
    assert policy["candidate_threshold"] == pytest.approx(0.458789)
    assert policy["positive_floor"] == pytest.approx(0.56656004)
    assert policy["negative_ceiling"] == pytest.approx(0.35101858)
    assert policy["separation_gap"] == pytest.approx(0.21554146)
    assert policy["positive_margin"] == pytest.approx(0.10656004)
    assert policy["negative_margin"] == pytest.approx(0.10898142)


def test_frozen_policy_math_and_manifest_identity_are_consistent(tmp_path: Path) -> None:
    _copy_policy_fixture(tmp_path)
    policy = _policy(tmp_path)
    manifest = json.loads(
        (tmp_path / "knowledge_base" / "index" / "index-manifest.json").read_text(encoding="utf-8")
    )
    positive = Decimal(str(policy["positive_floor"]))
    negative = Decimal(str(policy["negative_ceiling"]))
    threshold = Decimal(str(policy["score_threshold"]))

    assert positive - negative == Decimal(str(policy["separation_gap"]))
    assert positive - threshold == Decimal(str(policy["positive_margin"]))
    assert threshold - negative == Decimal(str(policy["negative_margin"]))
    assert negative < threshold < positive
    assert Decimal(str(policy["separation_gap"])) >= Decimal("0.03")
    for key in (
        "embedding_provider",
        "embedding_model",
        "embedding_dimensions",
        "collection_name",
        "distance",
        "total_points",
    ):
        assert manifest[key] == policy[key]


def test_policy_rejects_additional_property_and_hash_divergence(tmp_path: Path) -> None:
    _copy_policy_fixture(tmp_path)
    policy = _policy(tmp_path)
    policy["unexpected"] = "rejected"
    _write_policy(tmp_path, policy)

    assert _validate(tmp_path).returncode != 0

    _copy_policy_fixture(tmp_path)
    (tmp_path / "knowledge_base" / "index" / "index-plan.json").write_bytes(b"{}\n")

    assert _validate(tmp_path).returncode != 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("score_threshold", -0.99),
        ("request_id", "opaque"),
        ("frozen_at", "2026-08-02T00:00:00Z"),
        ("evidence_path", "C:\\restricted\\artifact.json"),
    ],
)
def test_policy_rejects_invalid_or_prohibited_content(
    tmp_path: Path, field: str, value: object
) -> None:
    _copy_policy_fixture(tmp_path)
    policy = _policy(tmp_path)
    policy[field] = value
    _write_policy(tmp_path, policy)

    assert _validate(tmp_path).returncode != 0
