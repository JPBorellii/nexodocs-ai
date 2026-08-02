"""Offline contracts for the full RAG holdout fixture and system freeze."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
from jsonschema import Draft202012Validator

from nexodocs_ai.observability.reports import (
    PRIVACY_SAFE_REPORT_SCHEMA,
    ReportError,
    index_report,
    privacy_safe_report,
    validate_privacy_safe_report,
)
from nexodocs_ai.retrieval.models import (
    EmbeddingBatchUsage,
    EmbeddingRunUsage,
    IndexingOperationResult,
    IndexingResult,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_VALIDATOR = ROOT / "scripts/validate_full_rag_holdout_fixture.py"
FREEZE_VALIDATOR = ROOT / "scripts/validate_full_rag_system_freeze.py"
INCIDENT_VALIDATOR = ROOT / "scripts/validate_full_rag_holdout_r01_technical_incident.py"


def _run(script: Path, root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), "--root", str(root)],
        check=False,
        capture_output=True,
        text=True,
    )


def _run_attempt(script: Path, root: Path, attempt: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), "--root", str(root), "--attempt", attempt],
        check=False,
        capture_output=True,
        text=True,
    )


def _copy_fixture(destination: Path) -> Path:
    for relative in (
        "evals/rag/full-rag-holdout-r01-cases.json",
        "evals/rag/full-rag-holdout-cases.schema.json",
    ):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    return destination / "evals/rag/full-rag-holdout-r01-cases.json"


def _copy_freeze(destination: Path) -> Path:
    artifact = json.loads(
        (ROOT / "evals/rag/full-rag-holdout-r01-system-freeze.json").read_text(encoding="utf-8")
    )
    for relative in (
        "evals/rag/full-rag-holdout-r01-system-freeze.json",
        "evals/rag/full-rag-holdout-system-freeze.schema.json",
    ):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    for item in artifact["frozen_files"]:
        source = ROOT / item["path"]
        target = destination / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return destination / artifact["frozen_files"][0]["path"]


def test_fixture_is_valid_and_records_the_exact_twelve_cases(tmp_path: Path) -> None:
    fixture_path = _copy_fixture(tmp_path)
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert _run(FIXTURE_VALIDATOR, tmp_path).returncode == 0
    assert fixture["score_threshold"] == pytest.approx(0.46)
    assert fixture["top_k"] == 5
    assert fixture["embedding_model"] == "text-embedding-3-small"
    assert fixture["answer_model"] == "gpt-5.6-luna"
    assert len(fixture["cases"]) == 12
    assert sum(case["kind"] == "supported" for case in fixture["cases"]) == 6
    assert fixture["cases"][8]["expected_preflight_behavior"] == "clinical_preflight_block"


@pytest.mark.parametrize(
    "field,value",
    [
        ("unexpected", True),
        ("timestamp", "2026-08-02T00:00:00Z"),
        ("path", "C:\\temp\\holdout.json"),
    ],
)
def test_fixture_rejects_closed_schema_and_prohibited_content(
    tmp_path: Path, field: str, value: object
) -> None:
    fixture_path = _copy_fixture(tmp_path)
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    fixture[field] = value
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    assert _run(FIXTURE_VALIDATOR, tmp_path).returncode == 1


@pytest.mark.parametrize("field", ["case_id", "query"])
def test_fixture_rejects_duplicate_ids_and_queries(tmp_path: Path, field: str) -> None:
    fixture_path = _copy_fixture(tmp_path)
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    fixture["cases"][1][field] = fixture["cases"][0][field]
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    assert _run(FIXTURE_VALIDATOR, tmp_path).returncode == 1


def test_r01_system_freeze_preserves_the_pre_correction_boundary(tmp_path: Path) -> None:
    semantic_file = _copy_freeze(tmp_path)
    assert _run(FREEZE_VALIDATOR, tmp_path).returncode == 1
    semantic_file.write_text(semantic_file.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert _run(FREEZE_VALIDATOR, tmp_path).returncode == 1


def test_r01_technical_incident_is_sanitized_and_hash_bound(tmp_path: Path) -> None:
    for relative in (
        "evals/rag/full-rag-holdout-r01-technical-incident.json",
        "evals/rag/full-rag-holdout-r01-technical-incident.schema.json",
        "evals/rag/full-rag-holdout-r01-cases.json",
        "evals/rag/full-rag-holdout-r01-system-freeze.json",
        "knowledge_base/index/retrieval-threshold-policy.json",
        "knowledge_base/index/index-manifest.json",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    incident = tmp_path / "evals/rag/full-rag-holdout-r01-technical-incident.json"
    assert _run(INCIDENT_VALIDATOR, tmp_path).returncode == 0
    payload = json.loads(incident.read_text(encoding="utf-8"))
    assert payload["quality_decision"] == "NOT_EVALUATED"
    assert payload["retrieval_calls_observed"] is None
    payload["message"] = "raw provider message"
    incident.write_text(json.dumps(payload), encoding="utf-8")
    assert _run(INCIDENT_VALIDATOR, tmp_path).returncode == 1


def test_r02_fixture_retains_every_r01_case_and_records_the_predecessor() -> None:
    r01 = json.loads(
        (ROOT / "evals/rag/full-rag-holdout-r01-cases.json").read_text(encoding="utf-8")
    )
    r02 = json.loads(
        (ROOT / "evals/rag/full-rag-holdout-r02-cases.json").read_text(encoding="utf-8")
    )
    assert _run_attempt(FIXTURE_VALIDATOR, ROOT, "r02").returncode == 0
    assert r02["cases"] == r01["cases"]
    assert r02["predecessor"] == "full-rag-holdout-r01"
    assert r02["predecessor_status"] == "TECHNICALLY_FAILED"
    assert r02["predecessor_quality_decision"] == "NOT_EVALUATED"


def test_r02_system_freeze_is_valid() -> None:
    assert _run_attempt(FREEZE_VALIDATOR, ROOT, "r02").returncode == 0


def _privacy_safe_index_report() -> dict[str, object]:
    usage = EmbeddingRunUsage(
        "openai",
        "text-embedding-3-small",
        1536,
        32,
        1,
        1,
        1,
        1,
        2,
        2,
        True,
        (EmbeddingBatchUsage(1, 1, 2, 2, "req-safe", 1),),
    )
    return privacy_safe_report(
        index_report(IndexingOperationResult(IndexingResult(1, 0, 0, 1), usage), "collection", 1)
    )


def test_privacy_safe_schema_is_closed_and_removes_operational_identifiers() -> None:
    Draft202012Validator.check_schema(PRIVACY_SAFE_REPORT_SCHEMA)
    report = _privacy_safe_index_report()
    forbidden = {
        "request_id",
        "request_ids",
        "run_id",
        "timestamp",
        "timestamp_utc",
        "prompt",
        "context",
    }

    def collect_keys(value: object) -> set[str]:
        if isinstance(value, dict):
            mapping = cast(dict[str, object], value)
            return set(mapping) | set().union(*(collect_keys(item) for item in mapping.values()))
        return set()

    assert not forbidden & collect_keys(report)
    assert report["prompt_tokens"] == report["total_tokens"] == 2
    assert report["logical_api_calls"] == report["physical_attempts"] == 1
    report["unexpected"] = True
    with pytest.raises(ReportError):
        validate_privacy_safe_report(report)
