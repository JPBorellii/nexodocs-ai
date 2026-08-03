"""Contracts for the consolidated, sanitized D03 result."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

ROOT = Path(__file__).resolve().parents[2]
if TYPE_CHECKING:
    from scripts.validate_grounding_diagnostic_d03_result import (
        D03ResultValidationError,
        verify_local_sources,
    )
else:
    sys.path.insert(0, str(ROOT / "scripts"))
    from validate_grounding_diagnostic_d03_result import (
        D03ResultValidationError,
        verify_local_sources,
    )

RESULT = ROOT / "evals/rag/grounding-diagnostic-d03-result-v1.json"
SCHEMA = ROOT / "evals/rag/grounding-diagnostic-d03-result-v1.schema.json"
D04_CONCEPT_SCHEMA = ROOT / "evals/rag/grounding-diagnostic-d04-signals-concept.schema.json"
SCRIPT = ROOT / "scripts/validate_grounding_diagnostic_d03_result.py"


def _load_result() -> dict[str, object]:
    return cast(dict[str, object], json.loads(RESULT.read_text(encoding="utf-8")))


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _write_candidate(tmp_path: Path, value: dict[str, object]) -> Path:
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def _cases(value: dict[str, object]) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], value["cases"])


def _mutate(kind: str) -> dict[str, object]:
    value = copy.deepcopy(_load_result())
    cases = _cases(value)
    if kind == "additional_field":
        value["unexpected"] = False
    elif kind == "missing_field":
        value.pop("version")
    elif kind == "invalid_enum":
        value["execution_status"] = "FAILED"
    elif kind == "invalid_hash":
        cases[0]["usage_report_sha256"] = "not-a-hash"
    elif kind == "duplicate_case":
        cases[1] = copy.deepcopy(cases[0])
    elif kind == "missing_case":
        cases.pop()
    elif kind == "wrong_order":
        cases[0], cases[1] = cases[1], cases[0]
    elif kind == "wrong_classification":
        cases[0]["classification"] = "NON_REPRODUCED_IN_D03"
    elif kind == "incoherent_status":
        cases[0]["status"] = "answered"
    elif kind == "incoherent_safe_error_code":
        cases[0]["safe_error_code"] = None
    elif kind == "artifact_on_non_reproduced":
        cases[1]["diagnostic_artifact_sha256"] = "a" * 64
    elif kind == "artifact_missing_on_reproduced":
        cases[0]["diagnostic_artifact_sha256"] = None
    elif kind == "calls_diverge":
        cases[0]["retrieval_logical_api_calls"] = 2
    elif kind == "attempts_diverge":
        cases[0]["answer_physical_attempts"] = 2
    elif kind == "wrong_consolidated_total":
        value["grounding_failures_reproduced"] = 3
    elif kind == "wrong_reproduced_list":
        value["reproduced_case_ids"] = ["HOLD-P04", "HOLD-P02"]
    elif kind == "wrong_non_reproduced_list":
        value["non_reproduced_case_ids"] = ["HOLD-N04", "HOLD-P03"]
    elif kind == "wrong_error_counts":
        value["safe_error_code_counts"] = {"grounding_quote_not_in_evidence": 1}
    elif kind == "wrong_threshold":
        value["threshold"] = 0.47
    elif kind == "wrong_top_k":
        value["top_k"] = 4
    elif kind == "wrong_embedding_model":
        value["embedding_model"] = "synthetic-model"
    elif kind == "wrong_answer_model":
        value["answer_model"] = "synthetic-model"
    elif kind == "quality_decision":
        value["holdout_quality_decision"] = "FAILED"
    elif kind == "r03_created":
        value["r03_created"] = True
    elif kind == "prohibited_content":
        value["answer"] = "synthetic raw content"
    elif kind == "absolute_path":
        value["path"] = "C:\\synthetic\\artifact.json"
    elif kind == "timestamp":
        value["timestamp"] = "2026-08-02T12:00:00Z"
    elif kind == "request_id":
        value["request_id"] = "req_synthetic"
    elif kind == "simulated_secret":
        value["unexpected"] = "sk-" + "synthetic" + "123456789"
    else:
        raise AssertionError(f"Unknown mutation: {kind}")
    return value


def test_versioned_result_is_valid_and_help_is_available() -> None:
    assert _run().returncode == 0
    help_result = _run("--help")
    assert help_result.returncode == 0
    assert "--verify-local-sources" in help_result.stdout


@pytest.mark.parametrize(
    "kind",
    [
        "additional_field",
        "missing_field",
        "invalid_enum",
        "invalid_hash",
        "duplicate_case",
        "missing_case",
        "wrong_order",
        "wrong_classification",
        "incoherent_status",
        "incoherent_safe_error_code",
        "artifact_on_non_reproduced",
        "artifact_missing_on_reproduced",
        "calls_diverge",
        "attempts_diverge",
        "wrong_consolidated_total",
        "wrong_reproduced_list",
        "wrong_non_reproduced_list",
        "wrong_error_counts",
        "wrong_threshold",
        "wrong_top_k",
        "wrong_embedding_model",
        "wrong_answer_model",
        "quality_decision",
        "r03_created",
        "prohibited_content",
        "absolute_path",
        "timestamp",
        "request_id",
        "simulated_secret",
    ],
)
def test_result_validator_rejects_contract_mutations(tmp_path: Path, kind: str) -> None:
    candidate = _write_candidate(tmp_path, _mutate(kind))
    completed = _run("--result", str(candidate))
    assert completed.returncode == 1
    assert completed.stdout == "Grounding diagnostic D03 result validation failed.\n"


def test_ci_mode_does_not_require_local_run_reports(tmp_path: Path) -> None:
    eval_directory = tmp_path / "evals/rag"
    eval_directory.mkdir(parents=True)
    (eval_directory / RESULT.name).write_bytes(RESULT.read_bytes())
    (eval_directory / SCHEMA.name).write_bytes(SCHEMA.read_bytes())
    (eval_directory / D04_CONCEPT_SCHEMA.name).write_bytes(D04_CONCEPT_SCHEMA.read_bytes())
    completed = _run("--root", str(tmp_path))
    assert completed.returncode == 0
    assert not (tmp_path / "data").exists()


def _usage_report(status: str, safe_error_code: str | None) -> dict[str, object]:
    retrieval = {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "logical_api_calls": 1,
        "physical_attempts": 1,
        "input_count": 1,
        "prompt_tokens": 10,
        "cached_input_tokens": None,
        "output_tokens": None,
        "total_tokens": 10,
        "application_attempts": 0,
        "transport_attempts_observable": True,
    }
    answer = {
        "provider": "openai",
        "model": "gpt-5.6-luna",
        "logical_api_calls": 1,
        "physical_attempts": 1,
        "input_count": 1,
        "prompt_tokens": 15,
        "cached_input_tokens": 0,
        "output_tokens": 5,
        "total_tokens": 20,
        "application_attempts": 1,
        "transport_attempts_observable": True,
    }
    return {
        "schema_version": "1.0",
        "report_version": "1.0.0",
        "operation": "answer",
        "provider": "openai",
        "model": "gpt-5.6-luna",
        "logical_api_calls": 2,
        "physical_attempts": 2,
        "input_count": 2,
        "inserted_points": None,
        "reused_points": None,
        "obsolete_points": None,
        "prompt_tokens": 25,
        "cached_input_tokens": 0,
        "output_tokens": 5,
        "total_tokens": 30,
        "application_attempts": 1,
        "status": status,
        "safe_error_code": safe_error_code,
        "query_character_count": 20,
        "retrieval_usage": retrieval,
        "answer_usage": answer,
    }


def _write_json(path: Path, value: dict[str, object]) -> str:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_sources(tmp_path: Path) -> dict[str, object]:
    data = copy.deepcopy(_load_result())
    cases: dict[str, dict[str, object]] = {}
    for case in _cases(data):
        case_id = case["case_id"]
        assert isinstance(case_id, str)
        cases[case_id] = case
    for case_id, case in cases.items():
        reproduced = case_id in {"HOLD-P02", "HOLD-P04"}
        status = "grounding_failed" if reproduced else "answered"
        code = "grounding_quote_not_in_evidence" if reproduced else None
        usage_name = f"phase-6a-grounding-diagnostic-d03-{case_id[5:].lower()}-usage.json"
        usage_hash = _write_json(tmp_path / usage_name, _usage_report(status, code))
        case["usage_report_sha256"] = usage_hash
        if not reproduced:
            case["diagnostic_artifact_sha256"] = None
            continue
        artifact: dict[str, object] = {
            "diagnostic_id": "grounding-diagnostic-d03",
            "system_commit": "a" * 40,
            "fixture_sha256": "b" * 64,
            "system_freeze_sha256": "c" * 64,
            "threshold_policy_sha256": "d" * 64,
            "case_id": case_id,
            "status": "grounding_failed",
            "safe_error_code": "grounding_quote_not_in_evidence",
            "retrieval_logical_api_calls": 1,
            "retrieval_physical_attempts": 1,
            "answer_logical_api_calls": 1,
            "answer_physical_attempts": 1,
            "retrieval_total_tokens": 10,
            "answer_total_tokens": 20,
            "total_tokens": 30,
            "usage_report_sha256": usage_hash,
            "execution_completed": True,
            "privacy_scan_passed": True,
        }
        artifact_name = f"phase-6a-grounding-diagnostic-d03-{case_id[5:].lower()}.json"
        case["diagnostic_artifact_sha256"] = _write_json(tmp_path / artifact_name, artifact)
    return data


def test_verify_local_sources_accepts_matching_synthetic_files(tmp_path: Path) -> None:
    data = _synthetic_sources(tmp_path)
    verify_local_sources(ROOT, data, tmp_path)


def test_verify_local_sources_rejects_hash_mismatch(tmp_path: Path) -> None:
    data = _synthetic_sources(tmp_path)
    target = tmp_path / "phase-6a-grounding-diagnostic-d03-p02-usage.json"
    target.write_text("{}", encoding="utf-8")
    with pytest.raises(D03ResultValidationError):
        verify_local_sources(ROOT, data, tmp_path)


def test_verify_local_sources_rejects_missing_file(tmp_path: Path) -> None:
    data = _synthetic_sources(tmp_path)
    target = tmp_path / "phase-6a-grounding-diagnostic-d03-n04-usage.json"
    target.unlink()
    with pytest.raises(D03ResultValidationError):
        verify_local_sources(ROOT, data, tmp_path)
