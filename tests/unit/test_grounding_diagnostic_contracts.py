"""Closed taxonomy, D03 schema, and oracle-correction contracts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
from jsonschema import Draft202012Validator

from nexodocs_ai.rag.grounding_diagnostics import (
    GROUNDING_SAFE_ERROR_CODES,
    GroundingErrorCode,
    safe_grounding_error_code,
    validate_safe_error_code,
)
from nexodocs_ai.rag.models import ValidationError

ROOT = Path(__file__).resolve().parents[2]


def _run(script: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def test_grounding_taxonomy_is_closed_and_generic_fallback_ignores_messages() -> None:
    assert GROUNDING_SAFE_ERROR_CODES == frozenset(code.value for code in GroundingErrorCode)
    raw = ValidationError("raw answer, quote, req_forbidden and traceback")
    assert safe_grounding_error_code(raw) == GroundingErrorCode.VALIDATION_FAILED.value
    with pytest.raises(ValueError):
        validate_safe_error_code("grounding_failed", str(raw))
    assert _run("validate_grounding_error_codes.py").returncode == 0


def test_d03_schema_has_exact_fields_and_no_diagnostic_is_created() -> None:
    path = ROOT / "evals/rag/grounding-diagnostic-d03.schema.json"
    schema = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    properties = cast(dict[str, object], schema["properties"])
    assert set(properties) == set(schema["required"])
    safe_code = cast(dict[str, object], properties["safe_error_code"])
    assert safe_code["enum"] == [code.value for code in GroundingErrorCode]
    assert not (ROOT / "evals/rag/grounding-diagnostic-d03.json").exists()
    assert _run("validate_grounding_diagnostic.py").returncode == 0


def test_d03_validator_accepts_only_sanitized_future_artifacts(tmp_path: Path) -> None:
    valid = {
        "diagnostic_id": "grounding-diagnostic-d03",
        "system_commit": "a" * 40,
        "fixture_sha256": "b" * 64,
        "system_freeze_sha256": "c" * 64,
        "threshold_policy_sha256": "d" * 64,
        "case_id": "HOLD-P02",
        "status": "grounding_failed",
        "safe_error_code": "grounding_missing_citation",
        "retrieval_logical_api_calls": 1,
        "retrieval_physical_attempts": 1,
        "answer_logical_api_calls": 1,
        "answer_physical_attempts": 1,
        "retrieval_total_tokens": 3,
        "answer_total_tokens": 5,
        "total_tokens": 8,
        "usage_report_sha256": "e" * 64,
        "execution_completed": True,
        "privacy_scan_passed": True,
    }
    artifact = tmp_path / "d03.json"
    artifact.write_text(json.dumps(valid), encoding="utf-8")
    assert _run("validate_grounding_diagnostic.py", "--artifact", str(artifact)).returncode == 0
    valid["answer"] = "raw answer"
    artifact.write_text(json.dumps(valid), encoding="utf-8")
    assert _run("validate_grounding_diagnostic.py", "--artifact", str(artifact)).returncode == 1


def test_p05_oracle_correction_is_closed_sanitized_and_directs_r03() -> None:
    path = ROOT / "evals/rag/evaluation-oracle-corrections-v1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    correction = data["corrections"][0]
    assert correction["case_id"] == "HOLD-P05"
    assert correction["original_fact_code"] == "nexo_integral_active_north_unit"
    assert correction["correction_type"] == "EVALUATION_ORACLE_DEFECT"
    assert correction["source_document_id"] == "NSI-TAB-COV-001"
    assert correction["evidence_scope"] == "R02_RETRIEVED_EVIDENCE"
    assert correction["corrected_expectation_code"] == (
        "nexo_integral_not_confirmed_for_north_unit"
    )
    assert correction["future_fixture_directive"] == {
        "fixture_id": "full-rag-holdout-r03",
        "preserve_question": True,
        "use_corrected_expectation": True,
    }
    serialized = json.dumps(data, ensure_ascii=False)
    assert not {"question", "query", "answer", "citation", "quote", "excerpt"} & set(correction)
    assert "sk-" not in serialized and "traceback" not in serialized
    assert _run("validate_evaluation_oracle_corrections.py").returncode == 0
