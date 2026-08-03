"""Schema, validator, privacy, atomic-write, and CLI contracts for D04."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest
from jsonschema import Draft202012Validator

from nexodocs_ai.observability.grounding_diagnostic_d04 import (
    D04ArtifactError,
    build_d04_artifact,
    resolve_d04_path,
    write_d04_artifact,
)
from nexodocs_ai.observability.reports import answer_report, privacy_safe_report
from nexodocs_ai.rag.models import (
    EvidenceBlock,
    GenerationUsage,
    RagResponse,
    RagRunResult,
)
from nexodocs_ai.rag.sanitized_grounding import project_quote_failure
from nexodocs_ai.retrieval.models import EmbeddingRunUsage

ROOT = Path(__file__).resolve().parents[2]
D04_SCHEMA = ROOT / "evals/rag/grounding-diagnostic-d04.schema.json"
D04_RESULT_SCHEMA = ROOT / "evals/rag/grounding-diagnostic-d04-result.schema.json"


def _sentinel(label: str) -> str:
    return "_".join(("SENT" + "INEL", "PRIVATE", label))


def _block(text: str) -> EvidenceBlock:
    return EvidenceBlock(
        1,
        "synthetic-chunk",
        "synthetic-document",
        "Synthetic",
        "synthetic.txt",
        "section=synthetic",
        "Synthetic",
        0.9,
        text,
        "a" * 64,
    )


def _run() -> RagRunResult:
    quote = _sentinel("QUOTE")
    evidence = (_block(_sentinel("EVIDENCE_DIFFERENT")),)
    signals = (project_quote_failure(quote, 1, evidence),)
    retrieval = EmbeddingRunUsage("fake", "fake-embedding", 3, 1, 1, 1, 1, 1, 10, 10, True, ())
    generation = GenerationUsage("fake", "fake-answer", 15, 0, 5, 20, None, 1, 1, False, 1, True)
    response = RagResponse(
        "1.0",
        "grounding_failed",
        _sentinel("QUESTION"),
        (),
        reason_code="grounding_quote_not_in_evidence",
        message="synthetic fallback",
    )
    return RagRunResult(response, retrieval, generation, 1, signals)


def _artifact(tmp_path: Path) -> dict[str, object]:
    usage = tmp_path / "usage.json"
    run = _run()
    report = privacy_safe_report(
        answer_report(run, "fake", "fake-answer", "synthetic-collection", 10, 1)
    )
    usage.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    return build_d04_artifact(ROOT, "HOLD-P02", run, usage)


def _schema_root(tmp_path: Path) -> Path:
    destination = tmp_path / "evals/rag"
    destination.mkdir(parents=True)
    shutil.copyfile(D04_SCHEMA, destination / D04_SCHEMA.name)
    return tmp_path


def test_d04_schemas_are_closed_draft_2020_12_and_no_real_result_exists() -> None:
    for path in (D04_SCHEMA, D04_RESULT_SCHEMA):
        schema = json.loads(path.read_text(encoding="utf-8"))
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["additionalProperties"] is False
        Draft202012Validator.check_schema(schema)
    assert not (ROOT / "evals/rag/grounding-diagnostic-d04.json").exists()
    assert not (ROOT / "evals/rag/grounding-diagnostic-d04-result.json").exists()


def test_artifact_builder_contains_only_closed_sanitized_projection(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    schema = json.loads(D04_SCHEMA.read_text(encoding="utf-8"))
    errors = list(Draft202012Validator(schema).iter_errors(cast(Any, artifact)))  # pyright: ignore[reportUnknownMemberType]
    assert not errors
    serialized = json.dumps(artifact, ensure_ascii=False)
    assert artifact["status"] == "grounding_failed"
    assert artifact["safe_error_code"] == "grounding_quote_not_in_evidence"
    assert artifact["retrieval_logical_api_calls"] == artifact["retrieval_physical_attempts"] == 1
    assert artifact["answer_logical_api_calls"] == artifact["answer_physical_attempts"] == 1
    assert artifact["total_tokens"] == 30
    for sentinel in (
        _sentinel("QUESTION"),
        _sentinel("ANSWER"),
        _sentinel("QUOTE"),
        _sentinel("EVIDENCE"),
        _sentinel("PROMPT"),
        "-".join(("sk", "proj", _sentinel("SECRET_NOT_REAL"))),
    ):
        assert sentinel not in serialized


def test_offline_validator_accepts_artifact_and_rejects_incoherence(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/validate_grounding_diagnostic_d04.py"),
            "--artifact",
            str(path),
            "--usage-report",
            str(tmp_path / "usage.json"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    artifact["quote_failure_count"] = 2
    path.write_text(json.dumps(artifact), encoding="utf-8")
    rejected = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/validate_grounding_diagnostic_d04.py"),
            "--artifact",
            str(path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode == 1
    assert rejected.stdout == "Grounding diagnostic D04 validation failed.\n"


def test_atomic_exclusive_write_and_no_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _schema_root(tmp_path)
    artifact = _artifact(tmp_path)
    candidate = "data/run-reports/d04.json"
    written = write_d04_artifact(root, candidate, artifact)
    assert json.loads(written.read_text(encoding="utf-8")) == artifact
    assert not list(written.parent.glob(".d04-*.tmp"))
    with pytest.raises(D04ArtifactError):
        write_d04_artifact(root, candidate, artifact)

    failing = "data/run-reports/failing.json"

    def fail_link(*_: object) -> None:
        raise OSError("synthetic")

    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(D04ArtifactError):
        write_d04_artifact(root, failing, artifact)
    assert not (root / failing).exists()
    assert not list((root / "data/run-reports").glob(".d04-*.tmp"))


@pytest.mark.parametrize(
    "candidate",
    (
        "outside.json",
        "data/outside.json",
        "data/run-reports/../outside.json",
        "data/run-reports/d04.txt",
    ),
)
def test_d04_path_rejects_unsafe_destinations(tmp_path: Path, candidate: str) -> None:
    with pytest.raises(D04ArtifactError):
        resolve_d04_path(tmp_path, candidate)


def test_consolidated_validator_accepts_exactly_two_synthetic_cases(tmp_path: Path) -> None:
    result = {
        "diagnostic_id": "grounding-diagnostic-d04-result",
        "version": "1.0.0",
        "case_ids": ["HOLD-P02", "HOLD-P04"],
        "classification_counts": {"CASE_ONLY_DIFFERENCE": 1, "INCONCLUSIVE": 1},
        "case_classifications": {"HOLD-P02": "CASE_ONLY_DIFFERENCE", "HOLD-P04": "INCONCLUSIVE"},
        "classified_cases": ["HOLD-P02"],
        "inconclusive_cases": ["HOLD-P04"],
        "artifact_sha256_by_case": {"HOLD-P02": "a" * 64, "HOLD-P04": "b" * 64},
        "holdout_quality_decision": "NOT_APPLICABLE",
        "r03_created": False,
        "artifact_integrity_passed": True,
        "privacy_contract_passed": True,
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/validate_grounding_diagnostic_d04_result.py"),
            "--result",
            str(path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    result["classified_cases"] = ["HOLD-P02", "HOLD-P04"]
    path.write_text(json.dumps(result), encoding="utf-8")
    rejected = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/validate_grounding_diagnostic_d04_result.py"),
            "--result",
            str(path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode == 1


def test_cli_contract_is_opt_in_and_invalid_destination_fails_without_provider_calls(
    tmp_path: Path,
) -> None:
    help_result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/answer_question.py"), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0
    assert "--sanitized-grounding-diagnostic-report" in help_result.stdout

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/answer_question.py"),
            "--query",
            "synthetic",
            "--usage-report",
            "data/run-reports/synthetic-usage.json",
            "--privacy-safe-usage-report",
            "--sanitized-grounding-diagnostic-report",
            str(tmp_path / "outside.json"),
            "--sanitized-grounding-diagnostic-case-id",
            "HOLD-P02",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert completed.stderr == ""
    assert json.loads(completed.stdout) == {
        "safe_error_code": "grounding_diagnostic_report_unavailable",
        "status": "diagnostic_failed",
    }
    assert not (tmp_path / "outside.json").exists()
