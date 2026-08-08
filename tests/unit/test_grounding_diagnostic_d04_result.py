"""Contracts for the consolidated, sanitized D04 result v1."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "evals/rag/grounding-diagnostic-d04-result-v1.json"
SCHEMA = ROOT / "evals/rag/grounding-diagnostic-d04-result-v1.schema.json"
SCRIPT = ROOT / "scripts/validate_grounding_diagnostic_d04_result.py"


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


def _nested(value: dict[str, object], key: str) -> dict[str, object]:
    return cast(dict[str, object], value[key])


def _mutate(kind: str) -> dict[str, object]:
    value = copy.deepcopy(_load_result())
    if kind == "additional_field":
        value["unexpected"] = False
    elif kind == "missing_field":
        value.pop("version")
    elif kind == "wrong_commit":
        value["system_commit"] = "a" * 40
    elif kind == "wrong_classification":
        _nested(value, "case_classifications")["HOLD-P02"] = "INCONCLUSIVE"
    elif kind == "wrong_classification_count":
        _nested(value, "classification_counts")["WHITESPACE_ONLY_DIFFERENCE"] = 1
    elif kind == "wrong_quote_failure_count":
        _nested(value, "quote_failure_counts_by_case")["HOLD-P04"] = 2
    elif kind == "wrong_usage_hash":
        _nested(value, "usage_report_sha256_by_case")["HOLD-P02"] = "a" * 64
    elif kind == "wrong_artifact_hash":
        _nested(value, "artifact_sha256_by_case")["HOLD-P04"] = "b" * 64
    elif kind == "wrong_summary_hash":
        value["summary_sha256"] = "c" * 64
    elif kind == "inconclusive":
        value["inconclusive_cases"] = ["HOLD-P02"]
        value["inconclusive_count"] = 1
    elif kind == "integrity_failed":
        value["artifact_integrity_passed"] = False
    elif kind == "privacy_failed":
        value["privacy_contract_passed"] = False
    elif kind == "quality_decision":
        value["holdout_quality_decision"] = "PASSED"
    elif kind == "wrong_root_cause":
        value["root_cause_classification"] = "ROOT_CAUSE_INSUFFICIENT_REQUIRE_D04"
    elif kind == "r03_created":
        value["r03_created"] = True
    elif kind == "prohibited_content":
        value["answer"] = "synthetic raw content"
    elif kind == "absolute_path":
        value["path"] = "C:\\synthetic\\artifact.json"
    elif kind == "simulated_secret":
        value["unexpected"] = "sk-" + "synthetic" + "123456789"
    else:
        raise AssertionError(f"Unknown mutation: {kind}")
    return value


def test_versioned_result_is_valid_and_help_documents_local_verification() -> None:
    completed = _run()
    assert completed.returncode == 0
    assert completed.stdout == "Grounding diagnostic D04 result validation passed.\n"
    help_result = _run("--help")
    assert help_result.returncode == 0
    assert "--verify-local-sources" in help_result.stdout
    assert "--local-source-dir" in help_result.stdout


@pytest.mark.parametrize(
    "kind",
    [
        "additional_field",
        "missing_field",
        "wrong_commit",
        "wrong_classification",
        "wrong_classification_count",
        "wrong_quote_failure_count",
        "wrong_usage_hash",
        "wrong_artifact_hash",
        "wrong_summary_hash",
        "inconclusive",
        "integrity_failed",
        "privacy_failed",
        "quality_decision",
        "wrong_root_cause",
        "r03_created",
        "prohibited_content",
        "absolute_path",
        "simulated_secret",
    ],
)
def test_result_validator_rejects_contract_mutations(tmp_path: Path, kind: str) -> None:
    candidate = _write_candidate(tmp_path, _mutate(kind))
    completed = _run("--result", str(candidate))
    assert completed.returncode == 1
    assert completed.stdout == "Grounding diagnostic D04 result validation failed.\n"


def test_ci_mode_does_not_require_ignored_local_sources(tmp_path: Path) -> None:
    completed = _run("--local-source-dir", str(tmp_path))
    assert completed.returncode == 0
    verified = _run("--verify-local-sources", "--local-source-dir", str(tmp_path))
    assert verified.returncode == 1
    assert verified.stdout == "Grounding diagnostic D04 result validation failed.\n"


def test_versioned_result_contains_only_auditable_closed_values() -> None:
    data = _load_result()
    prohibited = {
        "query",
        "answer",
        "quote",
        "evidence",
        "citation_text",
        "document",
        "chunk",
        "prompt",
        "normalized_content",
        "quote_sha256",
        "evidence_sha256",
    }

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            mapping = cast(dict[str, object], value)
            found = set(mapping)
            for nested in mapping.values():
                found.update(keys(nested))
            return found
        if isinstance(value, list):
            found: set[str] = set()
            for nested in cast(list[object], value):
                found.update(keys(nested))
            return found
        return set()

    assert keys(data).isdisjoint(prohibited)
    assert data["inconclusive_count"] == 0
    assert data["r03_created"] is False
    assert not list((ROOT / "evals/rag").glob("*r03*"))


def test_versioned_schema_is_closed_and_specific_to_the_real_result() -> None:
    schema = cast(dict[str, object], json.loads(SCHEMA.read_text(encoding="utf-8")))
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False
    properties = cast(dict[str, object], schema["properties"])
    assert cast(dict[str, object], properties["system_commit"])["const"] == (
        "9e1a327d19ca10f0c4757de00c3bb2dfa5887fd9"
    )
    assert cast(dict[str, object], properties["root_cause_classification"])["const"] == (
        "ROOT_CAUSE_CONFIRMED_WHITESPACE_CANONICALIZATION_GAP"
    )
