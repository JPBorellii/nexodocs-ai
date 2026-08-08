"""Schema, validator, privacy, atomic-write, and CLI contracts for D04."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import pytest
from jsonschema import Draft202012Validator

from nexodocs_ai.observability.grounding_diagnostic_d04 import (
    D04ArtifactError,
    D04ErrorCode,
    build_d04_artifact,
    ensure_distinct_report_destinations,
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


def _artifact(
    tmp_path: Path,
    case_id: str = "HOLD-P02",
    usage_name: str = "usage.json",
) -> dict[str, object]:
    usage = tmp_path / usage_name
    run = _run()
    report = privacy_safe_report(
        answer_report(run, "fake", "fake-answer", "synthetic-collection", 10, 1)
    )
    usage.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    return build_d04_artifact(ROOT, case_id, run, usage)


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
    assert completed.stdout == "Grounding diagnostic D04 operational validation passed.\n"
    artifact["quote_failure_count"] = 2
    path.write_text(json.dumps(artifact), encoding="utf-8")
    rejected = subprocess.run(
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


def test_equivalent_report_destinations_are_rejected_case_insensitively(tmp_path: Path) -> None:
    candidates = (
        ("data/run-reports/same.json", "data/run-reports/same.json"),
        ("data/run-reports/./same.json", "data/run-reports/nested/../same.json"),
        ("data/run-reports/SAME.json", "data/run-reports/same.JSON"),
    )
    for usage, diagnostic in candidates:
        with pytest.raises(D04ArtifactError) as failure:
            ensure_distinct_report_destinations(tmp_path, usage, diagnostic)
        assert failure.value.code == D04ErrorCode.DESTINATION_CONFLICT.value


def test_existing_hard_links_are_rejected_by_samefile(tmp_path: Path) -> None:
    report_directory = tmp_path / "data/run-reports"
    report_directory.mkdir(parents=True)
    usage = report_directory / "usage.json"
    diagnostic = report_directory / "diagnostic.json"
    usage.write_text("synthetic", encoding="utf-8")
    os.link(usage, diagnostic)
    with pytest.raises(D04ArtifactError) as failure:
        ensure_distinct_report_destinations(tmp_path, usage, diagnostic)
    assert failure.value.code == D04ErrorCode.DESTINATION_CONFLICT.value


def test_samefile_oserror_falls_back_to_normalized_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_directory = tmp_path / "data/run-reports"
    report_directory.mkdir(parents=True)
    usage = report_directory / "usage.json"
    diagnostic = report_directory / "diagnostic.json"
    usage.write_text("usage", encoding="utf-8")
    diagnostic.write_text("diagnostic", encoding="utf-8")

    def unavailable(_left: object, _right: object) -> bool:
        raise OSError("SENTINEL_PRIVATE_PATH")

    monkeypatch.setattr(os.path, "samefile", unavailable)
    resolved_usage, resolved_diagnostic = ensure_distinct_report_destinations(
        tmp_path, usage, diagnostic
    )
    assert resolved_usage == usage.resolve()
    assert resolved_diagnostic == diagnostic.resolve()


@pytest.mark.parametrize(
    "operation",
    (
        "mkdir",
        "mkstemp",
        "fdopen",
        "fsync",
        "link",
        "file_exists",
        "is_directory",
        "not_directory",
    ),
)
def test_operational_write_failures_are_closed_and_leave_no_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    root = _schema_root(tmp_path)
    artifact = _artifact(tmp_path)
    sentinel = "SENTINEL_PRIVATE_PATH"
    candidate = "data/run-reports/failure.json"

    def fail_mkdir(*_args: object, **_kwargs: object) -> None:
        raise PermissionError(sentinel)

    def fail_mkstemp(**_kwargs: object) -> tuple[int, str]:
        raise PermissionError(sentinel)

    def fail_fsync(_descriptor: int) -> None:
        raise OSError(sentinel)

    def fail_link(_source: object, _destination: object) -> None:
        raise OSError(sentinel)

    def fail_fdopen(_descriptor: int, _mode: str) -> Any:
        raise PermissionError(sentinel)

    def fail_file_exists(_source: object, _destination: object) -> None:
        raise FileExistsError(sentinel)

    def fail_is_directory(**_kwargs: object) -> tuple[int, str]:
        raise IsADirectoryError(sentinel)

    def fail_not_directory(**_kwargs: object) -> tuple[int, str]:
        raise NotADirectoryError(sentinel)

    if operation == "mkdir":
        monkeypatch.setattr(Path, "mkdir", fail_mkdir)
    elif operation == "mkstemp":
        monkeypatch.setattr(tempfile, "mkstemp", fail_mkstemp)
    elif operation == "fdopen":
        monkeypatch.setattr(os, "fdopen", fail_fdopen)
    elif operation == "fsync":
        monkeypatch.setattr(os, "fsync", fail_fsync)
    elif operation == "link":
        monkeypatch.setattr(os, "link", fail_link)
    elif operation == "file_exists":
        monkeypatch.setattr(os, "link", fail_file_exists)
    elif operation == "is_directory":
        monkeypatch.setattr(tempfile, "mkstemp", fail_is_directory)
    else:
        monkeypatch.setattr(tempfile, "mkstemp", fail_not_directory)

    try:
        write_d04_artifact(root, candidate, artifact)
    except D04ArtifactError as exc:
        rendered_traceback = traceback.format_exc()
        expected_code = {
            "file_exists": D04ErrorCode.DESTINATION_EXISTS.value,
            "is_directory": D04ErrorCode.DESTINATION_INVALID.value,
            "not_directory": D04ErrorCode.DESTINATION_INVALID.value,
        }.get(operation, D04ErrorCode.ARTIFACT_WRITE_FAILED.value)
        assert exc.code == expected_code
        assert sentinel not in str(exc)
        assert sentinel not in repr(exc)
        assert sentinel not in rendered_traceback
    else:
        pytest.fail("D04ArtifactError was not raised")
    assert not (root / candidate).exists()
    assert not list((root / "data/run-reports").glob(".d04-*.tmp"))


def test_cleanup_is_retried_without_replacing_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _schema_root(tmp_path)
    artifact = _artifact(tmp_path)
    original_unlink = Path.unlink
    failures = 0

    def transient_cleanup(path: Path, *, missing_ok: bool = False) -> None:
        nonlocal failures
        if path.name.startswith(".d04-") and failures == 0:
            failures += 1
            raise PermissionError("SENTINEL_PRIVATE_PATH")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", transient_cleanup)
    written = write_d04_artifact(root, "data/run-reports/cleanup.json", artifact)
    assert written.exists()
    assert failures == 1
    assert not list(written.parent.glob(".d04-*.tmp"))


def test_concurrent_publication_has_exactly_one_winner(tmp_path: Path) -> None:
    root = _schema_root(tmp_path)
    artifact = _artifact(tmp_path)

    def publish() -> str:
        try:
            write_d04_artifact(root, "data/run-reports/concurrent.json", artifact)
        except D04ArtifactError:
            return "rejected"
        return "written"

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(publish) for _ in range(2)]
        outcomes = [future.result() for future in futures]
    assert sorted(outcomes) == ["rejected", "written"]


def test_directory_destination_is_rejected_as_invalid(tmp_path: Path) -> None:
    root = _schema_root(tmp_path)
    artifact = _artifact(tmp_path)
    destination = root / "data/run-reports/directory.json"
    destination.mkdir(parents=True)
    with pytest.raises(D04ArtifactError) as failure:
        write_d04_artifact(root, destination, artifact)
    assert failure.value.code == D04ErrorCode.DESTINATION_INVALID.value


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


def test_consolidated_schema_only_accepts_exactly_two_synthetic_cases(tmp_path: Path) -> None:
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
            "--schema-only",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert completed.stdout == "Grounding diagnostic D04 result schema validation passed.\n"
    result["classified_cases"] = ["HOLD-P02", "HOLD-P04"]
    path.write_text(json.dumps(result), encoding="utf-8")
    rejected = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/validate_grounding_diagnostic_d04_result.py"),
            "--result",
            str(path),
            "--schema-only",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode == 1


def test_case_validator_modes_are_explicit_and_usage_is_mandatory(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    script = str(ROOT / "scripts/validate_grounding_diagnostic_d04.py")

    no_sources = subprocess.run(
        [sys.executable, script, "--artifact", str(artifact_path)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert no_sources.returncode == 1
    schema_only = subprocess.run(
        [sys.executable, script, "--schema-only", "--artifact", str(artifact_path)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert schema_only.returncode == 0
    assert schema_only.stdout == "Grounding diagnostic D04 schema validation passed.\n"
    invalid_combination = subprocess.run(
        [
            sys.executable,
            script,
            "--schema-only",
            "--artifact",
            str(artifact_path),
            "--usage-report",
            str(tmp_path / "usage.json"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert invalid_combination.returncode == 1


@pytest.mark.parametrize(
    "mutation",
    ("hash", "status", "safe_error_code", "calls", "attempts", "tokens", "privacy"),
)
def test_case_validator_rejects_incoherent_usage(tmp_path: Path, mutation: str) -> None:
    artifact = _artifact(tmp_path)
    usage_path = tmp_path / "usage.json"
    usage = json.loads(usage_path.read_text(encoding="utf-8"))
    if mutation == "hash":
        usage["duration_ms"] = 2
    elif mutation == "status":
        usage["status"] = "answered"
    elif mutation == "safe_error_code":
        usage["safe_error_code"] = "grounding_quote_mismatch"
    elif mutation == "calls":
        usage["retrieval_usage"]["logical_api_calls"] = 2
    elif mutation == "attempts":
        usage["answer_usage"]["physical_attempts"] = 2
    elif mutation == "tokens":
        usage["answer_usage"]["total_tokens"] = 21
    else:
        usage["query"] = "SENTINEL_PRIVATE_QUESTION"
    usage_path.write_text(json.dumps(usage), encoding="utf-8")
    if mutation != "hash":
        artifact["usage_report_sha256"] = hashlib.sha256(usage_path.read_bytes()).hexdigest()
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/validate_grounding_diagnostic_d04.py"),
            "--artifact",
            str(artifact_path),
            "--usage-report",
            str(usage_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert "SENTINEL" not in completed.stdout + completed.stderr


def test_consolidated_operational_mode_requires_and_binds_both_artifacts(
    tmp_path: Path,
) -> None:
    p02 = _artifact(tmp_path, "HOLD-P02", "p02-usage.json")
    p04 = _artifact(tmp_path, "HOLD-P04", "p04-usage.json")
    p02_path, p04_path = tmp_path / "p02.json", tmp_path / "p04.json"
    p02_path.write_text(json.dumps(p02), encoding="utf-8")
    p04_path.write_text(json.dumps(p04), encoding="utf-8")
    classification = cast(str, p02["sanitized_root_cause_class"])
    result = {
        "diagnostic_id": "grounding-diagnostic-d04-result",
        "version": "1.0.0",
        "case_ids": ["HOLD-P02", "HOLD-P04"],
        "classification_counts": {classification: 2},
        "case_classifications": {"HOLD-P02": classification, "HOLD-P04": classification},
        "classified_cases": ["HOLD-P02", "HOLD-P04"],
        "inconclusive_cases": [],
        "artifact_sha256_by_case": {
            "HOLD-P02": hashlib.sha256(p02_path.read_bytes()).hexdigest(),
            "HOLD-P04": hashlib.sha256(p04_path.read_bytes()).hexdigest(),
        },
        "holdout_quality_decision": "NOT_APPLICABLE",
        "r03_created": False,
        "artifact_integrity_passed": True,
        "privacy_contract_passed": True,
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    script = str(ROOT / "scripts/validate_grounding_diagnostic_d04_result.py")
    base = [sys.executable, script, "--result", str(result_path)]
    assert subprocess.run(base, cwd=ROOT, check=False).returncode == 1
    assert (
        subprocess.run([*base, "--p02-artifact", str(p02_path)], cwd=ROOT, check=False).returncode
        == 1
    )
    completed = subprocess.run(
        [
            *base,
            "--p02-artifact",
            str(p02_path),
            "--p04-artifact",
            str(p04_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert completed.stdout == "Grounding diagnostic D04 result operational validation passed.\n"
    swapped = subprocess.run(
        [
            *base,
            "--p02-artifact",
            str(p04_path),
            "--p04-artifact",
            str(p02_path),
        ],
        cwd=ROOT,
        check=False,
    )
    assert swapped.returncode == 1


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
        "safe_error_code": "d04_destination_invalid",
        "status": "diagnostic_failed",
    }
    assert not (tmp_path / "outside.json").exists()


@pytest.mark.parametrize(
    ("script_name", "expected_option"),
    (
        ("validate_grounding_diagnostic_d04.py", "--usage-report"),
        ("validate_grounding_diagnostic_d04_result.py", "--p02-artifact"),
    ),
)
def test_d04_validator_help_documents_operational_sources_and_schema_only(
    script_name: str,
    expected_option: str,
) -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script_name), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert expected_option in completed.stdout
    assert "--schema-only" in completed.stdout
