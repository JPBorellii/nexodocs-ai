"""Directed offline tests for the answerability development-calibration harness."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import cast

import pytest

import nexodocs_ai.rag.answerability_calibration as calibration
from nexodocs_ai.rag.answerability_calibration import (
    CHUNKS_RELATIVE_PATH,
    MANIFEST_RELATIVE_PATH,
    ORACLE_RELATIVE_PATH,
    CalibrationFixture,
    CalibrationFixtureError,
    CalibrationResultError,
    Decision,
    ExecutionMode,
    evaluate_calibration,
    load_calibration_fixture,
    map_expected_support_evidence_ids,
    reconstruct_request,
    serialize_calibration_result,
    write_calibration_result,
)
from nexodocs_ai.rag.answerability_prompts import (
    ANSWERABILITY_SYSTEM_PROMPT,
    render_answerability_input,
)
from nexodocs_ai.rag.models import (
    AnswerabilityDecision,
    AnswerabilityProviderError,
    AnswerabilityProviderRefusalError,
    AnswerabilityRequest,
    AnswerabilitySupport,
    GenerationUsage,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _InvalidDecision:
    pass


_INVALID_DECISION = _InvalidDecision()
type _Outcome = Decision | Exception | _InvalidDecision


class _ScriptedProvider:
    provider_name = "scripted"
    model_identifier = "deterministic-test-model"

    def __init__(self, outcomes: list[_Outcome], *, usage: GenerationUsage | None = None) -> None:
        self._outcomes = outcomes
        self._usage = usage
        self.requests: list[AnswerabilityRequest] = []

    def assess(self, request: AnswerabilityRequest) -> AnswerabilityDecision:
        outcome = self._outcomes[len(self.requests)]
        self.requests.append(request)
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, _InvalidDecision):
            return AnswerabilityDecision(cast(Decision, "invalid"), ())
        decision = outcome
        if decision == "insufficient":
            return AnswerabilityDecision(decision, (), self._usage)
        evidence = request.evidence_blocks[0]
        return AnswerabilityDecision(
            decision,
            (AnswerabilitySupport(evidence.evidence_id, evidence.text),),
            self._usage,
        )


class _OpenAIIdentityProvider(_ScriptedProvider):
    provider_name = "openai"


@pytest.fixture(scope="module")
def fixture() -> CalibrationFixture:
    return load_calibration_fixture(PROJECT_ROOT)


def _copy_inputs(tmp_path: Path) -> Path:
    for relative_path in (ORACLE_RELATIVE_PATH, CHUNKS_RELATIVE_PATH, MANIFEST_RELATIVE_PATH):
        destination = tmp_path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT / relative_path, destination)
    return tmp_path


def _json_object(path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def _oracle_cases(oracle: dict[str, object]) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], oracle["cases"])


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _refreeze_test_oracle(monkeypatch: pytest.MonkeyPatch, oracle_path: Path) -> None:
    digest = hashlib.sha256(oracle_path.read_bytes()).hexdigest()
    monkeypatch.setattr(calibration, "EXPECTED_ORACLE_SHA256", digest)


def _matching_outcomes(fixture: CalibrationFixture) -> list[_Outcome]:
    return [case.expected_decision for case in fixture.cases]


def _case_results(result: dict[str, object]) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], result["cases"])


def _metrics(result: dict[str, object]) -> dict[str, object]:
    return cast(dict[str, object], result["metrics"])


def test_valid_fixture_and_corpus_load_with_frozen_counts(fixture: CalibrationFixture) -> None:
    assert fixture.fixture_id == "nexodocs-answerability-calibration-v1"
    assert fixture.fixture_schema_version == "1.0.0"
    assert len(fixture.cases) == 32
    assert len(fixture.evidence_by_chunk_id) == 44
    assert sum(case.expected_decision == "answerable" for case in fixture.cases) == 16
    assert sum(case.expected_decision == "insufficient" for case in fixture.cases) == 16
    assert fixture.corpus_sha256 == (
        "a49ad2a4c3b8dc767277566be60e93cd2a8d3113ffe1c598eec9ccfb759b02c4"
    )
    assert fixture.oracle_sha256 == calibration.EXPECTED_ORACLE_SHA256


def test_structurally_valid_oracle_mutation_fails_frozen_digest(tmp_path: Path) -> None:
    root = _copy_inputs(tmp_path)
    oracle_path = root / ORACLE_RELATIVE_PATH
    oracle = _json_object(oracle_path)
    metadata = cast(dict[str, object], oracle["metadata"])
    metadata["creation_purpose"] = f"{metadata['creation_purpose']} Mutated."
    _write_json(oracle_path, oracle)

    with pytest.raises(CalibrationFixtureError, match="frozen digest"):
        load_calibration_fixture(root)


def test_oracle_bytes_are_read_once(monkeypatch: pytest.MonkeyPatch) -> None:
    oracle_path = PROJECT_ROOT / ORACLE_RELATIVE_PATH
    original_read_bytes = Path.read_bytes
    oracle_reads = 0

    def counting_read_bytes(path: Path) -> bytes:
        nonlocal oracle_reads
        if path == oracle_path:
            oracle_reads += 1
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counting_read_bytes)

    loaded = load_calibration_fixture(PROJECT_ROOT)

    assert loaded.oracle_sha256 == calibration.EXPECTED_ORACLE_SHA256
    assert oracle_reads == 1


def test_canonical_corpus_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    root = _copy_inputs(tmp_path)
    chunks_path = root / CHUNKS_RELATIVE_PATH
    chunks_path.write_bytes(chunks_path.read_bytes() + b" ")

    with pytest.raises(CalibrationFixtureError, match="corpus SHA-256"):
        load_calibration_fixture(root)


def test_manifest_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    root = _copy_inputs(tmp_path)
    manifest_path = root / MANIFEST_RELATIVE_PATH
    manifest = _json_object(manifest_path)
    manifest["chunks_sha256"] = "0" * 64
    _write_json(manifest_path, manifest)

    with pytest.raises(CalibrationFixtureError, match="manifest"):
        load_calibration_fixture(root)


def test_fixture_identity_mismatch_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _copy_inputs(tmp_path)
    oracle_path = root / ORACLE_RELATIVE_PATH
    oracle = _json_object(oracle_path)
    metadata = cast(dict[str, object], oracle["metadata"])
    metadata["fixture_schema_version"] = "2.0.0"
    _write_json(oracle_path, oracle)
    _refreeze_test_oracle(monkeypatch, oracle_path)

    with pytest.raises(CalibrationFixtureError, match="identity or schema version"):
        load_calibration_fixture(root)


def test_unknown_evidence_chunk_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _copy_inputs(tmp_path)
    oracle_path = root / ORACLE_RELATIVE_PATH
    oracle = _json_object(oracle_path)
    first = _oracle_cases(oracle)[0]
    evidence_ids = cast(list[str], first["evidence_chunk_ids"])
    support_ids = cast(list[str], first["expected_support_chunk_ids"])
    provenance = cast(list[dict[str, object]], first["evidence_provenance"])
    original_chunk_id = evidence_ids[0]
    unknown_chunk_id = "nsi-chk-ffffffffffffffffffffffff"
    evidence_ids[0] = unknown_chunk_id
    support_ids[:] = [
        unknown_chunk_id if chunk_id == original_chunk_id else chunk_id for chunk_id in support_ids
    ]
    provenance[0]["chunk_id"] = unknown_chunk_id
    provenance[0]["source_text_sha256"] = "0" * 64
    _write_json(oracle_path, oracle)
    _refreeze_test_oracle(monkeypatch, oracle_path)

    with pytest.raises(CalibrationFixtureError, match="unknown evidence chunk"):
        load_calibration_fixture(root)


def test_duplicate_evidence_reference_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _copy_inputs(tmp_path)
    oracle_path = root / ORACLE_RELATIVE_PATH
    oracle = _json_object(oracle_path)
    first = _oracle_cases(oracle)[0]
    evidence_ids = cast(list[str], first["evidence_chunk_ids"])
    provenance = cast(list[dict[str, object]], first["evidence_provenance"])
    evidence_ids.append(evidence_ids[0])
    provenance.append(dict(provenance[0]))
    _write_json(oracle_path, oracle)
    _refreeze_test_oracle(monkeypatch, oracle_path)

    with pytest.raises(CalibrationFixtureError, match="duplicates"):
        load_calibration_fixture(root)


def test_malformed_fixture_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _copy_inputs(tmp_path)
    oracle_path = root / ORACLE_RELATIVE_PATH
    oracle = _json_object(oracle_path)
    del _oracle_cases(oracle)[0]["query"]
    _write_json(oracle_path, oracle)
    _refreeze_test_oracle(monkeypatch, oracle_path)

    with pytest.raises(CalibrationFixtureError, match="closed shape"):
        load_calibration_fixture(root)


def test_provenance_text_hash_mismatch_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _copy_inputs(tmp_path)
    oracle_path = root / ORACLE_RELATIVE_PATH
    oracle = _json_object(oracle_path)
    first = _oracle_cases(oracle)[0]
    provenance = cast(list[dict[str, object]], first["evidence_provenance"])
    provenance[0]["source_text_sha256"] = "0" * 64
    _write_json(oracle_path, oracle)
    _refreeze_test_oracle(monkeypatch, oracle_path)

    with pytest.raises(CalibrationFixtureError, match="mismatched evidence provenance"):
        load_calibration_fixture(root)


def test_deterministic_evidence_ids_and_scoring_only_support_mapping(
    fixture: CalibrationFixture,
) -> None:
    for case in fixture.cases:
        request = reconstruct_request(fixture, case)
        assert tuple(block.evidence_id for block in request.evidence_blocks) == tuple(
            range(1, len(case.evidence_chunk_ids) + 1)
        )
        assert map_expected_support_evidence_ids(case) == tuple(
            case.evidence_chunk_ids.index(chunk_id) + 1
            for chunk_id in case.expected_support_chunk_ids
        )
        assert not hasattr(request, "expected_decision")
        assert not hasattr(request, "expected_support_chunk_ids")


def test_provider_model_input_contains_only_query_evidence_id_and_text(
    fixture: CalibrationFixture,
) -> None:
    provider = _ScriptedProvider(_matching_outcomes(fixture))

    evaluate_calibration(fixture, provider, execution_mode="deterministic_fake")

    assert len(provider.requests) == 32
    for request in provider.requests:
        rendered = cast(dict[str, object], json.loads(render_answerability_input(request)))
        evidence = cast(list[dict[str, object]], rendered["evidence"])
        assert set(rendered) == {"query", "evidence"}
        assert all(set(block) == {"evidence_id", "text"} for block in evidence)
        assert all(block.chunk_id == "" for block in request.evidence_blocks)
        assert all(block.document_id == "" for block in request.evidence_blocks)
        assert all(block.title == "" for block in request.evidence_blocks)
        assert all(block.source_filename == "" for block in request.evidence_blocks)
        assert all(block.locator == "" for block in request.evidence_blocks)
        assert all(block.text_sha256 == "" for block in request.evidence_blocks)


def test_scripted_provider_cannot_run_as_real_baseline(fixture: CalibrationFixture) -> None:
    provider = _ScriptedProvider(_matching_outcomes(fixture))

    with pytest.raises(CalibrationResultError, match="requires the openai provider"):
        evaluate_calibration(fixture, provider, execution_mode="real_provider_baseline")

    assert provider.requests == []


def test_openai_provider_identity_cannot_run_as_deterministic_fake(
    fixture: CalibrationFixture,
) -> None:
    provider = _OpenAIIdentityProvider(_matching_outcomes(fixture))

    with pytest.raises(CalibrationResultError, match="rejects the openai provider"):
        evaluate_calibration(fixture, provider, execution_mode="deterministic_fake")

    assert provider.requests == []


def test_unknown_execution_mode_fails_before_provider_call(fixture: CalibrationFixture) -> None:
    provider = _ScriptedProvider(_matching_outcomes(fixture))

    with pytest.raises(CalibrationResultError, match="Execution mode is invalid"):
        evaluate_calibration(
            fixture,
            provider,
            execution_mode=cast(ExecutionMode, "unknown_runtime_mode"),
        )

    assert provider.requests == []


def test_semantic_outcomes_and_operational_failures_remain_separate(
    fixture: CalibrationFixture,
) -> None:
    outcomes = _matching_outcomes(fixture)
    answerable_indices = [
        index for index, case in enumerate(fixture.cases) if case.expected_decision == "answerable"
    ]
    insufficient_indices = [
        index
        for index, case in enumerate(fixture.cases)
        if case.expected_decision == "insufficient"
    ]
    outcomes[answerable_indices[0]] = "insufficient"
    outcomes[insufficient_indices[0]] = "answerable"
    outcomes[answerable_indices[1]] = AnswerabilityProviderRefusalError("sanitized refusal")
    outcomes[insufficient_indices[1]] = AnswerabilityProviderError("sanitized failure")
    outcomes[answerable_indices[2]] = _INVALID_DECISION
    outcomes[insufficient_indices[2]] = RuntimeError("must not be persisted")
    provider = _ScriptedProvider(outcomes)

    result = evaluate_calibration(fixture, provider, execution_mode="deterministic_fake")
    metrics = _metrics(result)

    assert metrics["total_cases"] == 32
    assert metrics["passed_cases"] == 26
    assert metrics["failed_cases"] == 6
    assert metrics["overall_decision_accuracy"] == pytest.approx(26 / 32)
    assert metrics["answerable_total"] == 16
    assert metrics["answerable_correct"] == 13
    assert metrics["answerable_recall"] == pytest.approx(13 / 16)
    assert metrics["false_abstention_count"] == 1
    assert metrics["false_abstention_rate"] == pytest.approx(1 / 16)
    assert metrics["insufficient_total"] == 16
    assert metrics["insufficient_correct"] == 13
    assert metrics["insufficient_recall"] == pytest.approx(13 / 16)
    assert metrics["false_answer_count"] == 1
    assert metrics["false_answer_rate"] == pytest.approx(1 / 16)
    assert metrics["provider_refusal_count"] == 1
    assert metrics["provider_failure_count"] == 1
    assert metrics["invalid_output_count"] == 1
    assert metrics["evaluator_failure_count"] == 1
    assert metrics["support_id_agreement_evaluated"] == 14
    statuses = [case["operational_status"] for case in _case_results(result)]
    assert statuses.count("provider_refusal") == 1
    assert statuses.count("provider_failure") == 1
    assert statuses.count("invalid_output") == 1
    assert statuses.count("evaluator_failure") == 1


def test_support_id_agreement_is_secondary_not_semantic_pass(
    fixture: CalibrationFixture,
) -> None:
    provider = _ScriptedProvider(_matching_outcomes(fixture))

    result = evaluate_calibration(fixture, provider, execution_mode="deterministic_fake")

    answerable_results = [
        case for case in _case_results(result) if case["expected_decision"] == "answerable"
    ]
    assert all(case["passed"] is True for case in answerable_results)
    assert any(case["support_id_agreement"] is False for case in answerable_results)
    metrics = _metrics(result)
    agreement_count = sum(case["support_id_agreement"] is True for case in answerable_results)
    assert metrics["support_id_agreement_evaluated"] == 16
    assert metrics["support_id_agreement_count"] == agreement_count
    assert metrics["support_id_agreement_rate"] == pytest.approx(agreement_count / 16)


def test_result_is_development_calibration_and_contains_no_raw_content(
    fixture: CalibrationFixture,
) -> None:
    provider = _ScriptedProvider(_matching_outcomes(fixture))
    result = evaluate_calibration(fixture, provider, execution_mode="deterministic_fake")

    payload = serialize_calibration_result(result)

    assert result["evaluation_role"] == "development_calibration"
    assert result["unseen_holdout"] is False
    assert result["result_schema_version"] == "1.0.0"
    assert result["case_count"] == 32
    fixture_binding = cast(dict[str, object], result["fixture"])
    corpus_binding = cast(dict[str, object], result["corpus"])
    classifier_binding = cast(dict[str, object], result["classifier"])
    provider_binding = cast(dict[str, object], result["provider"])
    assert fixture_binding["fixture_id"] == fixture.fixture_id
    assert fixture_binding["fixture_schema_version"] == fixture.fixture_schema_version
    assert fixture_binding["oracle_artifact_sha256"] == fixture.oracle_sha256
    assert corpus_binding["canonical_corpus_sha256"] == fixture.corpus_sha256
    assert classifier_binding == {
        "prompt_version": "answerability-v1",
        "prompt_sha256": "0b9c69791182b67603342070ea7232ade4d80c2e7818ce4f2cb37ca0c3e27529",
    }
    assert provider_binding == {
        "provider": "scripted",
        "model": "deterministic-test-model",
    }
    assert "query" not in {key for case in _case_results(result) for key in case}
    assert "quote" not in {key for case in _case_results(result) for key in case}
    assert ANSWERABILITY_SYSTEM_PROMPT not in payload
    assert all(case.query not in payload for case in fixture.cases)
    assert all(evidence.text not in payload for evidence in fixture.evidence_by_chunk_id.values())
    for forbidden_key in (
        "raw_query",
        "raw_evidence",
        "supporting_quote",
        "system_prompt",
        "rendered_prompt",
        "api_key",
        "environment",
        "raw_sdk_response",
        "raw_refusal_text",
    ):
        assert forbidden_key not in payload


def test_result_keeps_safe_usage_and_omits_provider_request_id(
    fixture: CalibrationFixture,
) -> None:
    request_id = "provider-request-id-must-not-be-persisted"
    usage = GenerationUsage(
        "scripted",
        "deterministic-test-model",
        10,
        2,
        3,
        13,
        request_id,
        1,
        1,
        False,
        1,
        True,
    )
    provider = _ScriptedProvider(_matching_outcomes(fixture), usage=usage)

    result = evaluate_calibration(fixture, provider, execution_mode="deterministic_fake")
    payload = serialize_calibration_result(result)
    first_usage = cast(dict[str, object], _case_results(result)[0]["usage"])

    assert first_usage["input_tokens"] == 10
    assert first_usage["cached_input_tokens"] == 2
    assert first_usage["total_tokens"] == 13
    assert request_id not in payload
    assert "request_id" not in payload


def test_repeated_scripted_runs_are_byte_equivalent(fixture: CalibrationFixture) -> None:
    first = evaluate_calibration(
        fixture,
        _ScriptedProvider(_matching_outcomes(fixture)),
        execution_mode="deterministic_fake",
    )
    second = evaluate_calibration(
        fixture,
        _ScriptedProvider(_matching_outcomes(fixture)),
        execution_mode="deterministic_fake",
    )

    assert serialize_calibration_result(first) == serialize_calibration_result(second)


@pytest.mark.parametrize(
    "tamper",
    ["passed", "support_id_agreement", "aggregate_metric", "duplicate_case_id"],
)
def test_result_serialization_rejects_semantic_tampering(
    fixture: CalibrationFixture, tamper: str
) -> None:
    result = evaluate_calibration(
        fixture,
        _ScriptedProvider(_matching_outcomes(fixture)),
        execution_mode="deterministic_fake",
    )
    cases = _case_results(result)
    if tamper == "passed":
        cases[0]["passed"] = not cast(bool, cases[0]["passed"])
    elif tamper == "support_id_agreement":
        cases[0]["support_id_agreement"] = not cast(bool, cases[0]["support_id_agreement"])
    elif tamper == "aggregate_metric":
        metrics = _metrics(result)
        metrics["passed_cases"] = cast(int, metrics["passed_cases"]) - 1
    else:
        cases[1]["case_id"] = cases[0]["case_id"]

    with pytest.raises(CalibrationResultError, match="internally inconsistent"):
        serialize_calibration_result(result)


def test_result_serialization_rejects_answerable_without_observed_support(
    fixture: CalibrationFixture,
) -> None:
    result = evaluate_calibration(
        fixture,
        _ScriptedProvider(_matching_outcomes(fixture)),
        execution_mode="deterministic_fake",
    )
    answerable = next(
        case
        for case in _case_results(result)
        if case["expected_decision"] == "answerable" and case["support_id_agreement"] is False
    )
    answerable["observed_decision"] = "answerable"
    answerable["observed_supporting_evidence_ids"] = []
    answerable["passed"] = True
    answerable["support_id_agreement"] = False

    with pytest.raises(CalibrationResultError, match="internally inconsistent"):
        serialize_calibration_result(result)


def test_result_serialization_rejects_answerable_without_expected_support(
    fixture: CalibrationFixture,
) -> None:
    result = evaluate_calibration(
        fixture,
        _ScriptedProvider(_matching_outcomes(fixture)),
        execution_mode="deterministic_fake",
    )
    answerable = next(
        case
        for case in _case_results(result)
        if case["expected_decision"] == "answerable" and case["support_id_agreement"] is False
    )
    answerable["expected_decision"] = "answerable"
    answerable["expected_support_evidence_ids"] = []
    answerable["support_id_agreement"] = False
    metrics = _metrics(result)
    evaluated = cast(int, metrics["support_id_agreement_evaluated"]) - 1
    agreement_count = cast(int, metrics["support_id_agreement_count"])
    metrics["support_id_agreement_evaluated"] = evaluated
    metrics["support_id_agreement_rate"] = agreement_count / evaluated

    with pytest.raises(CalibrationResultError, match="internally inconsistent"):
        serialize_calibration_result(result)


def test_result_writer_rejects_semantically_tampered_result(
    fixture: CalibrationFixture, tmp_path: Path
) -> None:
    result = evaluate_calibration(
        fixture,
        _ScriptedProvider(_matching_outcomes(fixture)),
        execution_mode="deterministic_fake",
    )
    metrics = _metrics(result)
    metrics["passed_cases"] = cast(int, metrics["passed_cases"]) - 1
    output_path = tmp_path / "tampered-result.json"

    with pytest.raises(CalibrationResultError, match="internally inconsistent"):
        write_calibration_result(result, output_path)

    assert not output_path.exists()


def test_result_writer_creates_only_one_temporary_artifact(
    fixture: CalibrationFixture, tmp_path: Path
) -> None:
    result = evaluate_calibration(
        fixture,
        _ScriptedProvider(_matching_outcomes(fixture)),
        execution_mode="deterministic_fake",
    )
    output_path = tmp_path / "answerability-calibration-result.json"

    write_calibration_result(result, output_path)

    assert json.loads(output_path.read_text(encoding="utf-8")) == result
    with pytest.raises(CalibrationResultError, match="Unable to create"):
        write_calibration_result(result, output_path)
