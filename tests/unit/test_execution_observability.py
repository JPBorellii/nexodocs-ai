"""Closed operational-report and CLI observability contracts."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from nexodocs_ai.observability.reports import (
    REPORT_SCHEMA,
    ReportError,
    index_report,
    resolve_report_path,
    validate_report,
    write_report,
)
from nexodocs_ai.retrieval.models import (
    EmbeddingBatchUsage,
    EmbeddingRunUsage,
    IndexingOperationResult,
    IndexingResult,
)


def _operation() -> IndexingOperationResult:
    usage = EmbeddingRunUsage(
        "openai",
        "text-embedding-3-small",
        1536,
        32,
        44,
        2,
        2,
        2,
        123,
        123,
        True,
        (
            EmbeddingBatchUsage(1, 32, 80, 80, "req_safe-1", 1),
            EmbeddingBatchUsage(2, 12, 43, 43, "req_safe-2", 1),
        ),
    )
    return IndexingOperationResult(IndexingResult(44, 0, 0, 44), usage)


def test_report_schema_is_closed_and_models_are_immutable() -> None:
    Draft202012Validator.check_schema(REPORT_SCHEMA)
    report = index_report(_operation(), "nexodocs_chunks_v1", 25)
    data = validate_report(report)
    serialized = json.dumps(data, sort_keys=True)
    assert data["logical_api_calls"] == data["physical_attempts"] == 2
    assert data["prompt_tokens"] == data["total_tokens"] == 123
    assert "prompt" not in data and "vectors" not in data and "text" not in data
    assert "OPENAI_API_KEY" not in serialized and "sk-" not in serialized
    with pytest.raises(FrozenInstanceError):
        report.status = "changed"  # type: ignore[misc]
    unsafe = dict(data)
    unsafe["prompt"] = "proibido"
    with pytest.raises(ReportError):
        validate_report(unsafe)


def test_report_path_atomic_write_existing_file_and_overwrite(tmp_path: Path) -> None:
    destination = Path("data/run-reports/index.json")
    report = index_report(_operation(), "nexodocs_chunks_v1", 25)
    written = write_report(tmp_path, destination, report)
    assert written == tmp_path / destination
    assert json.loads(written.read_text(encoding="utf-8"))["input_count"] == 44
    assert not list(written.parent.glob(".usage-*.tmp"))
    with pytest.raises(ReportError, match="existe"):
        write_report(tmp_path, destination, report)
    assert write_report(tmp_path, destination, report, overwrite=True) == written


@pytest.mark.parametrize(
    "candidate",
    ["../report.json", "data/outside.json", "data/run-reports/../outside.json", "report.txt"],
)
def test_report_path_rejects_traversal_and_wrong_locations(tmp_path: Path, candidate: str) -> None:
    with pytest.raises(ReportError):
        resolve_report_path(tmp_path, candidate)


def test_report_path_rejects_symlink_when_supported(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    link = data / "run-reports"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation is unavailable on this platform")
    with pytest.raises(ReportError):
        resolve_report_path(tmp_path, "data/run-reports/report.json")


@pytest.mark.parametrize(
    ("script", "arguments"),
    [
        ("index_knowledge_base.py", ["write", "--help"]),
        ("search_knowledge_base.py", ["--help"]),
        ("answer_question.py", ["--help"]),
    ],
)
def test_cli_help_exposes_optional_usage_report(script: str, arguments: list[str]) -> None:
    completed = subprocess.run(
        [sys.executable, str(Path("scripts") / script), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "--usage-report" in completed.stdout
    assert "--overwrite-usage-report" in completed.stdout


def test_privacy_safe_usage_report_requires_a_destination() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(Path("scripts") / "answer_question.py"),
            "--query",
            "sintética",
            "--privacy-safe-usage-report",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "requires --usage-report" in completed.stderr
