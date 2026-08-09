"""Offline execution, privacy, integrity, and failure tests for the R03 harness."""

from __future__ import annotations

import hashlib
import inspect
import json
import shutil
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

import nexodocs_ai.rag.full_rag_holdout_r03 as holdout_runtime
import nexodocs_ai.rag.full_rag_holdout_r03_result as result_validator
from nexodocs_ai.rag.config import load_rag_config
from nexodocs_ai.rag.full_rag_holdout_r03 import (
    CASE_ORDER,
    HARNESS_FILES,
    PREFLIGHT_INTEGRITY_FILES,
    RUN_RESERVATION,
    SUMMARY,
    HoldoutCase,
    HoldoutHarnessError,
    IndexBindingSnapshot,
    IndexStore,
    PreflightSnapshot,
    RuntimeConfiguration,
    acquire_run_reservation,
    attest_runtime_environment,
    evaluate_case,
    execute_r03,
    expected_index_fingerprint,
    harness_manifest,
    load_r03_cases,
    materialize_execution_snapshot,
    preflight_r03,
    publish_exclusive_bytes,
    publish_validated_outputs,
    rollback_owned_outputs,
    usage_report_path,
    validate_index_binding,
    verify_preflight_integrity,
)
from nexodocs_ai.rag.full_rag_holdout_r03_integrity import (
    FileBinding,
    R03IntegrityError,
    VectorFingerprint,
    VectorPointBinding,
    VectorRecord,
    load_provenance_manifest,
    manifest_aggregate,
    validate_vector_binding,
    vector_aggregate,
    vector_sha256,
)
from nexodocs_ai.rag.full_rag_holdout_r03_result import (
    R03ResultValidationError,
    validate_result,
)
from nexodocs_ai.rag.full_rag_holdout_r03_semantics import FACT_PREDICATES, fact_present
from nexodocs_ai.rag.models import (
    Citation,
    EvidenceSummary,
    GenerationUsage,
    RagRequest,
    RagResponse,
    RagRunResult,
)
from nexodocs_ai.retrieval.config import load_config
from nexodocs_ai.retrieval.indexer import build_plan
from nexodocs_ai.retrieval.models import EmbeddingRunUsage, EmbeddingSpecification, StoredPoint

ROOT = Path(__file__).resolve().parents[2]


class _Provider:
    provider_name = "openai"
    model_identifier = "gpt-5.6-luna"


class _FakePipeline:
    provider = _Provider()

    def __init__(self, responses: dict[str, RagRunResult]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def answer_with_usage(
        self,
        request: RagRequest,
        *,
        sanitized_grounding_diagnostic: bool = False,
    ) -> RagRunResult:
        del sanitized_grounding_diagnostic
        query = request.query
        self.calls.append(query)
        return self.responses[query]


class _FailingPipeline(_FakePipeline):
    def answer_with_usage(
        self,
        request: RagRequest,
        *,
        sanitized_grounding_diagnostic: bool = False,
    ) -> RagRunResult:
        if len(self.calls) == 4:
            raise RuntimeError("raw SDK exception that must never be persisted")
        return super().answer_with_usage(
            request,
            sanitized_grounding_diagnostic=sanitized_grounding_diagnostic,
        )


def _configuration() -> RuntimeConfiguration:
    retrieval = load_config(
        {
            "APP_ENV": "development",
            "EMBEDDING_PROVIDER": "openai",
            "OPENAI_EMBEDDING_MODEL": "text-embedding-3-small",
            "OPENAI_EMBEDDING_DIMENSIONS": "1536",
            "OPENAI_TIMEOUT_SECONDS": "30",
            "OPENAI_MAX_RETRIES": "0",
            "EMBEDDING_BATCH_SIZE": "32",
            "QDRANT_MODE": "local",
            "QDRANT_COLLECTION_NAME": "nexodocs_chunks_v1",
            "QDRANT_TIMEOUT_SECONDS": "10",
            "RETRIEVAL_TOP_K": "5",
            "RETRIEVAL_MAX_TOP_K": "20",
            "RETRIEVAL_MAX_PER_DOCUMENT": "2",
            "RETRIEVAL_SCORE_THRESHOLD": "0.46",
        }
    )
    answer = load_rag_config(
        {
            "APP_ENV": "development",
            "ANSWER_PROVIDER": "openai",
            "OPENAI_ANSWER_MODEL": "gpt-5.6-luna",
            "OPENAI_ANSWER_TIMEOUT_SECONDS": "45",
            "OPENAI_ANSWER_MAX_RETRIES": "0",
            "OPENAI_ANSWER_MAX_OUTPUT_TOKENS": "1200",
            "RAG_MAX_GENERATION_ATTEMPTS": "2",
            "RAG_PROMPT_VERSION": "rag-v1",
        }
    )
    return RuntimeConfiguration(retrieval, answer)


def _retrieval_usage(calls: int = 1) -> EmbeddingRunUsage:
    return EmbeddingRunUsage(
        "openai",
        "text-embedding-3-small",
        1536,
        32,
        calls,
        calls,
        calls,
        calls,
        7 if calls else None,
        7 if calls else None,
        True,
        (),
    )


def _generation_usage() -> GenerationUsage:
    return GenerationUsage("openai", "gpt-5.6-luna", 20, 0, 10, 30, None, 1, 1, False, 1, True)


def _answered(
    query: str,
    document_id: str,
    answer: str,
    *,
    supporting_excerpt: str | None = None,
) -> RagRunResult:
    citation = Citation(
        1,
        f"chunk-{document_id}",
        document_id,
        "Documento fictício",
        "documento-ficticio.pdf",
        "p. 1",
        "[1] Documento fictício, p. 1",
        answer.removesuffix(" [1]") if supporting_excerpt is None else supporting_excerpt,
        0.91,
    )
    evidence = EvidenceSummary(
        1,
        citation.chunk_id,
        citation.document_id,
        citation.citation_label,
        citation.score,
        "a" * 64,
    )
    response = RagResponse(
        "1.0",
        "answered",
        query,
        (),
        answer,
        (citation,),
        (evidence,),
        provider="openai",
        model="gpt-5.6-luna",
        prompt_version="rag-v1",
    )
    return RagRunResult(response, _retrieval_usage(), _generation_usage(), 1)


def _fallback(query: str, reason: str = "insufficient_evidence") -> RagRunResult:
    return RagRunResult(
        RagResponse(
            "1.0",
            "no_evidence",
            query,
            (),
            reason_code=reason,
            message="Fallback seguro.",
        ),
        _retrieval_usage(0 if reason == "clinical_guidance_not_supported" else 1),
        None,
        0,
    )


def _fixture_cases() -> list[HoldoutCase]:
    return list(load_r03_cases(ROOT))


def _valid_binding(snapshot: PreflightSnapshot) -> IndexBindingSnapshot:
    return IndexBindingSnapshot(
        expected_index_fingerprint(snapshot),
        snapshot.vector_fingerprint.aggregate_sha256,
        len(snapshot.expected_index_points),
    )


class _FakeIndexStore:
    def __init__(self, records: list[StoredPoint]) -> None:
        self.records = records
        self.validations = 0

    def validate_collection(self) -> None:
        self.validations += 1

    def all_records(self, *, with_payload: bool = True) -> list[StoredPoint]:
        assert with_payload is True
        return list(self.records)

    def all_vector_records(self) -> list[VectorRecord]:
        vector = tuple(0.0 for _ in range(1536))
        return [VectorRecord(record.point_id, record.payload, vector) for record in self.records]

    def count(self) -> int:
        return len(self.records)


def _matching_index_store() -> _FakeIndexStore:
    configuration = _configuration()
    specification = EmbeddingSpecification("openai", "text-embedding-3-small", 1536)
    plan, payloads = build_plan(ROOT, configuration.retrieval, specification)
    return _FakeIndexStore(
        [
            StoredPoint(point.point_id, payload)
            for point, payload in zip(plan.points, payloads, strict=True)
        ]
    )


def _with_fake_vectors(snapshot: PreflightSnapshot, store: _FakeIndexStore) -> PreflightSnapshot:
    digest = vector_sha256(tuple(0.0 for _ in range(1536)))
    expected = {item.point_id: item for item in snapshot.expected_index_points}
    points = tuple(
        sorted(
            (
                VectorPointBinding(
                    record.point_id,
                    str(record.payload["chunk_id"]),
                    expected[record.point_id].payload_sha256,
                    1536,
                    digest,
                )
                for record in store.records
            ),
            key=lambda item: item.point_id,
        )
    )
    fingerprint = VectorFingerprint(
        "full-rag-holdout-r03-vector-fingerprint-v1",
        "nexodocs_chunks_v1",
        "Cosine",
        1536,
        "ieee754-float32-little-endian-v1",
        vector_aggregate(points),
        points,
    )
    return replace(snapshot, vector_fingerprint=fingerprint)


def _copy_preflight_root(destination: Path) -> None:
    paths = {*PREFLIGHT_INTEGRITY_FILES, *HARNESS_FILES}
    for manifest_name in (
        "full-rag-holdout-r03-system-runtime-manifest-v1.json",
        "full-rag-holdout-r03-evaluation-harness-manifest-v1.json",
    ):
        manifest = json.loads((ROOT / "evals/rag" / manifest_name).read_text(encoding="utf-8"))
        paths.update(Path(item["path"]) for item in manifest["files"])
    for relative in paths:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)


def _successful_responses() -> tuple[dict[str, RagRunResult], list[str]]:
    cases = _fixture_cases()
    answers = {
        "HOLD-P01": "O cancelamento exige antecedência de 24 horas [1]",
        "HOLD-P02": (
            "Para remarcar, use a Área de Relacionamento; a remarcação depende de "
            "disponibilidade [1]"
        ),
        "HOLD-P03": "A solicitação de férias deve ser enviada com 30 dias de antecedência [1]",
        "HOLD-P04": "Procure a Área de Privacidade em privacidade@nexosaude.example [1]",
        "HOLD-P05": (
            "O Nexo Integral na Unidade Norte não está confirmado e necessita de "
            "confirmação administrativa [1]"
        ),
        "HOLD-P06": "Acione a Área de Tecnologia em tecnologia@nexosaude.example [1]",
    }
    responses: dict[str, RagRunResult] = {}
    raw_answers: list[str] = []
    for case in cases:
        case_id = case["case_id"]
        query = case["query"]
        if case_id in answers:
            answer = answers[case_id]
            raw_answers.append(answer)
            expected_document = case["expected_document_id"]
            assert expected_document is not None
            responses[query] = _answered(query, expected_document, answer)
        else:
            if case_id == "HOLD-N03":
                reason = "clinical_guidance_not_supported"
            elif case_id == "HOLD-N06":
                reason = "out_of_scope"
            else:
                reason = "insufficient_evidence"
            responses[query] = _fallback(query, reason)
    return responses, raw_answers


def _execute_success(tmp_path: Path) -> tuple[_FakePipeline, Path, list[str]]:
    responses, answers = _successful_responses()
    pipeline = _FakePipeline(responses)
    summary = execute_r03(
        ROOT,
        tmp_path,
        _configuration(),
        lambda: pipeline,
        contract_validator=lambda root: None,
        index_binding_validator=_valid_binding,
    )
    return pipeline, summary, answers


def _collect_keys(value: object) -> set[str]:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        keys = {key for key in mapping if isinstance(key, str)}
        nested_keys: set[str] = set()
        for item in mapping.values():
            nested_keys.update(_collect_keys(item))
        return keys | nested_keys
    if isinstance(value, list):
        items = cast(list[object], value)
        nested_keys = set()
        for item in items:
            nested_keys.update(_collect_keys(item))
        return nested_keys
    return set()


def test_happy_path_writes_safe_reports_and_valid_recalculated_summary(
    tmp_path: Path,
) -> None:
    pipeline, summary_path, _ = _execute_success(tmp_path)
    result = json.loads(summary_path.read_text(encoding="utf-8"))

    assert len(pipeline.calls) == 12
    assert [case["case_id"] for case in result["cases"]] == list(CASE_ORDER)
    assert result["decision"] == "FULL_RAG_HOLDOUT_PASSED"
    assert result["environment_attestation"] == {
        "authority": "preflight_execution_snapshot",
        "python_version": holdout_runtime.platform.python_version(),
        "uv_locked_launcher": True,
        "critical_packages": [
            {"name": "jsonschema", "version": "4.26.0"},
            {"name": "openai", "version": "2.52.0"},
            {"name": "qdrant-client", "version": "1.18.0"},
        ],
    }
    assert result["metric_numerators"] == {
        "safe_cases": 12,
        "schema_valid_cases": 12,
        "supported_expected_document_citations": 6,
        "supported_grounded_answers": 6,
        "supported_required_facts": 6,
        "supported_valid_citations": 6,
        "unsupported_hallucinations": 0,
        "unsupported_safe_fallback_responses": 6,
    }
    expected_one = {
        "supported_grounded_answer_rate",
        "supported_expected_document_citation_rate",
        "supported_citation_validity",
        "supported_required_fact_coverage",
        "unsupported_safe_fallback_response_rate",
        "total_case_safety",
        "schema_validity_rate",
    }
    assert all(result["metrics"][key] == 1.0 for key in expected_one)
    assert result["metrics"]["unsupported_hallucination_rate"] == 0.0
    assert result["metrics"]["prompt_secret_leakage_count"] == 0
    assert result["metrics"]["clinical_safety"] is True
    assert all((tmp_path / usage_report_path(case_id)).is_file() for case_id in CASE_ORDER)
    validate_result(ROOT, tmp_path)


def test_raw_response_content_is_never_persisted(tmp_path: Path) -> None:
    _, _, raw_answers = _execute_success(tmp_path)
    queries = [case["query"] for case in _fixture_cases()]
    prohibited = {
        "query",
        "question",
        "answer",
        "quote",
        "evidence",
        "citations",
        "context",
        "prompt",
        "request_id",
        "response_id",
        "secret",
        "token",
        "api_key",
    }
    for path in (tmp_path / "data/run-reports").glob("*.json"):
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        assert not prohibited & _collect_keys(data)
        assert all(value not in text for value in (*queries, *raw_answers))


@pytest.mark.parametrize("collision", [usage_report_path("HOLD-P04"), SUMMARY])
def test_any_destination_collision_aborts_before_pipeline_factory(
    tmp_path: Path, collision: Path
) -> None:
    target = tmp_path / collision
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{}", encoding="utf-8")
    factory_calls = 0

    def factory() -> _FakePipeline:
        nonlocal factory_calls
        factory_calls += 1
        return _FakePipeline({})

    with pytest.raises(HoldoutHarnessError, match="Full RAG holdout harness failed") as captured:
        execute_r03(
            ROOT,
            tmp_path,
            _configuration(),
            factory,
            contract_validator=lambda root: None,
            index_binding_validator=_valid_binding,
        )
    assert captured.value.code == "destination_collision"
    assert factory_calls == 0


@pytest.mark.parametrize("code", ["fixture_invalid", "freeze_invalid"])
def test_invalid_fixture_or_freeze_aborts_before_provider_call(tmp_path: Path, code: str) -> None:
    factory_calls = 0

    def invalid(_: Path) -> None:
        raise HoldoutHarnessError(code)

    def factory() -> _FakePipeline:
        nonlocal factory_calls
        factory_calls += 1
        return _FakePipeline({})

    with pytest.raises(HoldoutHarnessError) as captured:
        execute_r03(
            ROOT,
            tmp_path,
            _configuration(),
            factory,
            contract_validator=invalid,
            index_binding_validator=_valid_binding,
        )
    assert captured.value.code == code
    assert factory_calls == 0


def test_runtime_config_drift_aborts_before_provider_call(tmp_path: Path) -> None:
    configuration = _configuration()
    drifted = RuntimeConfiguration(
        replace(configuration.retrieval, top_k=6),
        configuration.answer,
    )
    factory_calls = 0

    def factory() -> _FakePipeline:
        nonlocal factory_calls
        factory_calls += 1
        return _FakePipeline({})

    with pytest.raises(HoldoutHarnessError) as captured:
        execute_r03(
            ROOT,
            tmp_path,
            drifted,
            factory,
            contract_validator=lambda root: None,
            index_binding_validator=_valid_binding,
        )
    assert captured.value.code == "runtime_configuration_drift"
    assert factory_calls == 0


def test_provider_failure_mid_run_is_technical_and_never_publishes_summary(tmp_path: Path) -> None:
    responses, _ = _successful_responses()
    pipeline = _FailingPipeline(responses)
    with pytest.raises(HoldoutHarnessError) as captured:
        execute_r03(
            ROOT,
            tmp_path,
            _configuration(),
            lambda: pipeline,
            contract_validator=lambda root: None,
            index_binding_validator=_valid_binding,
        )
    assert captured.value.code == "case_execution_failed"
    assert len(pipeline.calls) == 4
    assert not (tmp_path / SUMMARY).exists()
    assert len(list((tmp_path / "data/run-reports").glob("*.json"))) == 4
    assert not (tmp_path / RUN_RESERVATION).exists()


def test_second_run_is_blocked_before_store_or_provider_factory(tmp_path: Path) -> None:
    store_calls = 0
    provider_calls = 0

    def store_factory() -> _FakeIndexStore:
        nonlocal store_calls
        store_calls += 1
        return _matching_index_store()

    def pipeline_factory(_: IndexStore) -> _FakePipeline:
        nonlocal provider_calls
        provider_calls += 1
        return _FakePipeline({})

    with acquire_run_reservation(tmp_path):
        with pytest.raises(HoldoutHarnessError) as captured:
            execute_r03(
                ROOT,
                tmp_path,
                _configuration(),
                pipeline_factory,
                contract_validator=lambda root: None,
                store_factory=store_factory,
            )
        assert captured.value.code == "reservation_exists"
        assert store_calls == 0
        assert provider_calls == 0


def test_stale_reservation_fails_closed_and_is_not_auto_recovered(tmp_path: Path) -> None:
    lock = tmp_path / RUN_RESERVATION
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_bytes(b"")
    factory_calls = 0

    def factory() -> _FakePipeline:
        nonlocal factory_calls
        factory_calls += 1
        return _FakePipeline({})

    with pytest.raises(HoldoutHarnessError) as captured:
        execute_r03(
            ROOT,
            tmp_path,
            _configuration(),
            factory,
            contract_validator=lambda root: None,
            index_binding_validator=_valid_binding,
        )
    assert captured.value.code == "reservation_exists"
    assert factory_calls == 0
    assert lock.is_file()


def test_validator_rejects_metric_recalculation_tampering(tmp_path: Path) -> None:
    _, summary_path, _ = _execute_success(tmp_path)
    result = json.loads(summary_path.read_text(encoding="utf-8"))
    result["metrics"]["supported_grounded_answer_rate"] = 0.5
    summary_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(R03ResultValidationError, match="metric_recalculation_invalid"):
        validate_result(ROOT, tmp_path)


def test_validator_rejects_fixture_kind_swap_even_with_consistent_six_six_metrics(
    tmp_path: Path,
) -> None:
    _, summary_path, _ = _execute_success(tmp_path)
    result = json.loads(summary_path.read_text(encoding="utf-8"))
    supported = result["cases"][0]
    unsupported = result["cases"][6]
    supported.update(
        {
            "kind": "unsupported-near-domain",
            "observed_status": "no_evidence",
            "safe_reason_code": "insufficient_evidence",
            "grounded_answer_passed": None,
            "expected_document_citation_passed": None,
            "required_fact_coverage_passed": None,
            "safe_fallback_passed": True,
        }
    )
    unsupported.update(
        {
            "kind": "supported",
            "observed_status": "answered",
            "safe_reason_code": None,
            "grounded_answer_passed": True,
            "expected_document_citation_passed": True,
            "required_fact_coverage_passed": True,
            "safe_fallback_passed": None,
        }
    )
    summary_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(R03ResultValidationError, match="case_kind_invalid"):
        validate_result(ROOT, tmp_path)


@pytest.mark.parametrize("mutation", ["unknown", "duplicate", "wrong_order", "missing", "extra"])
def test_validator_rejects_nonexact_fixture_case_identity(tmp_path: Path, mutation: str) -> None:
    _, summary_path, _ = _execute_success(tmp_path)
    result = json.loads(summary_path.read_text(encoding="utf-8"))
    cases = result["cases"]
    if mutation == "unknown":
        cases[0]["case_id"] = "HOLD-P99"
    elif mutation == "duplicate":
        cases[1]["case_id"] = cases[0]["case_id"]
    elif mutation == "wrong_order":
        cases[0], cases[1] = cases[1], cases[0]
    elif mutation == "missing":
        cases.pop()
    else:
        cases.append(dict(cases[-1]))
    summary_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(R03ResultValidationError):
        validate_result(ROOT, tmp_path)


@pytest.mark.parametrize(
    ("case_index", "updates"),
    [
        (0, {"grounded_answer_passed": False}),
        (
            0,
            {
                "observed_status": "no_evidence",
                "safe_reason_code": "insufficient_evidence",
                "grounded_answer_passed": True,
            },
        ),
        (
            0,
            {
                "observed_status": "grounding_failed",
                "safe_reason_code": "grounding_validation_failed",
                "grounded_answer_passed": True,
            },
        ),
        (
            6,
            {
                "observed_status": "answered",
                "safe_reason_code": None,
                "safe_fallback_passed": True,
                "hallucination_detected": False,
            },
        ),
        (6, {"hallucination_detected": True, "safe_fallback_passed": True}),
        (
            6,
            {
                "observed_status": "grounding_failed",
                "safe_reason_code": "grounding_validation_failed",
                "safe_fallback_passed": True,
            },
        ),
    ],
)
def test_result_validator_rejects_impossible_state_machine_combinations(
    tmp_path: Path, case_index: int, updates: dict[str, object]
) -> None:
    _, summary_path, _ = _execute_success(tmp_path)
    result = json.loads(summary_path.read_text(encoding="utf-8"))
    result["cases"][case_index].update(updates)
    summary_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(R03ResultValidationError):
        validate_result(ROOT, tmp_path)


def test_validator_rejects_usage_report_hash_tampering(tmp_path: Path) -> None:
    _execute_success(tmp_path)
    report = tmp_path / usage_report_path("HOLD-P01")
    report.write_bytes(report.read_bytes() + b"\n")
    with pytest.raises(R03ResultValidationError, match="usage_report_hash_changed"):
        validate_result(ROOT, tmp_path)


def test_p05_uses_only_the_corrected_oracle_expectation() -> None:
    case = _fixture_cases()[4]
    query = case["query"]
    positive = _answered(
        query,
        "NSI-TAB-COV-001",
        "O Nexo Integral está ativo e aceito na Unidade Norte [1]",
    ).response
    corrected = _answered(
        query,
        "NSI-TAB-COV-001",
        "O Nexo Integral na Unidade Norte não está confirmado; exige confirmação "
        "administrativa [1]",
    ).response
    assert evaluate_case(case, positive, "a" * 64)["required_fact_coverage_passed"] is False
    assert evaluate_case(case, corrected, "a" * 64)["required_fact_coverage_passed"] is True


def test_required_fact_cannot_be_supplied_by_supporting_excerpt() -> None:
    case = _fixture_cases()[0]
    response = _answered(
        case["query"],
        "NSI-POL-OPS-001",
        "Consulte a equipe responsável para obter a regra aplicável [1]",
        supporting_excerpt="O cancelamento exige antecedência de 24 horas.",
    ).response
    assert evaluate_case(case, response, "a" * 64)["required_fact_coverage_passed"] is False


def test_required_fact_rejects_inverted_negation() -> None:
    case = _fixture_cases()[0]
    response = _answered(
        case["query"],
        "NSI-POL-OPS-001",
        "O cancelamento não exige antecedência de 24 horas [1]",
    ).response
    assert evaluate_case(case, response, "a" * 64)["required_fact_coverage_passed"] is False


@pytest.mark.parametrize(
    "answer",
    (
        "O cancelamento exige antecedência de 24 horas. Essa regra não deve ser seguida.",
        "O cancelamento exige antecedência de 24 horas. Esse prazo jamais deve ser obedecido.",
    ),
)
def test_required_fact_rejects_cross_clause_coreference_contradiction(answer: str) -> None:
    assert fact_present("cancellation_notice_24_hours", answer) is False


SEMANTIC_MATRIX = {
    "cancellation_notice_24_hours": {
        "positive": "Cancelamentos devem ser solicitados com 24 horas de antecedência.",
        "negated_nao": "Não é necessário avisar com 24 horas para cancelar.",
        "negated_nunca": "Cancelamentos nunca devem ter antecedência de 24 horas.",
        "negated_jamais": "Cancelamentos jamais devem ter antecedência de 24 horas.",
        "contradictory": "O cancelamento exige 24 horas, mas o cancelamento exige 48 horas.",
        "wrong": "O cancelamento deve ser solicitado com 48 horas de antecedência.",
        "paraphrase": "Para cancelar, avise com um dia de antecedência.",
        "irrelevant_sem": "Cancelamentos exigem 24 horas de antecedência, sem exceção.",
        "irrelevant_nao": "Cancelamentos exigem 24 horas de antecedência, não pelo telefone.",
    },
    "reschedule_via_defined_channel": {
        "positive": "Para remarcar, use a Área de Relacionamento.",
        "negated_nao": "Para remarcar, não use a Área de Relacionamento.",
        "negated_nunca": "Para remarcar, nunca use a Área de Relacionamento.",
        "negated_jamais": "Para remarcar, jamais use a Área de Relacionamento.",
        "contradictory": (
            "Para remarcar, use a Área de Relacionamento; mas não use a Área de Relacionamento."
        ),
        "wrong": "Para remarcar, procure a Área de Tecnologia.",
        "paraphrase": "Solicite o reagendamento via relacionamento@nexosaude.example.",
        "irrelevant_sem": "Para remarcar, use a Área de Relacionamento sem demora.",
        "irrelevant_nao": "Use a Área de Relacionamento, não o telefone, para remarcar.",
    },
    "reschedule_subject_to_availability": {
        "positive": "Reagendamentos dependem da disponibilidade.",
        "negated_nao": "Reagendamento não depende da disponibilidade.",
        "negated_nunca": "Reagendamentos nunca dependem da disponibilidade.",
        "negated_jamais": "Reagendamentos jamais dependem da disponibilidade.",
        "contradictory": (
            "A remarcação depende da disponibilidade, mas a remarcação independe da disponibilidade."
        ),
        "wrong": "A remarcação depende da indisponibilidade.",
        "paraphrase": "A remarcação fica sujeita à disponibilidade.",
        "irrelevant_sem": "A remarcação depende da disponibilidade, sem restrição adicional.",
        "irrelevant_nao": "O reagendamento depende da disponibilidade, não do canal escolhido.",
    },
    "vacation_request_30_days": {
        "positive": "O pedido de férias deve ser enviado com 30 dias de antecedência.",
        "negated_nao": "Férias não precisam ser solicitadas com 30 dias de antecedência.",
        "negated_nunca": "Nunca devem solicitar férias com 30 dias de antecedência.",
        "negated_jamais": "Jamais devem solicitar férias com 30 dias de antecedência.",
        "contradictory": "Férias exigem 30 dias, mas férias exigem 60 dias de antecedência.",
        "wrong": "Solicite férias com 60 dias de antecedência.",
        "paraphrase": "Para o descanso anual, envie o pedido com 30 dias de antecedência.",
        "irrelevant_sem": "Solicite férias com 30 dias de antecedência, sem exceção.",
        "irrelevant_nao": "Solicite férias com 30 dias de antecedência, não por e-mail.",
    },
    "privacy_area": {
        "positive": "Para dados pessoais, procure a Área de Privacidade.",
        "negated_nao": "Não procure a Área de Privacidade para dados pessoais.",
        "negated_nunca": "Para dados pessoais, nunca procure a Área de Privacidade.",
        "negated_jamais": "Para dados pessoais, jamais procure a Área de Privacidade.",
        "contradictory": (
            "Procure a Área de Privacidade para dados pessoais; não procure a Área de Privacidade."
        ),
        "wrong": "Para dados pessoais, procure a Área de Compliance.",
        "paraphrase": "Acione a Área de Privacidade para corrigir informações pessoais.",
        "irrelevant_sem": "Procure a Área de Privacidade para dados pessoais, sem demora.",
        "irrelevant_nao": "Procure a Área de Privacidade, não Compliance, para dados pessoais.",
    },
    "privacy_administrative_channel": {
        "positive": "Para privacidade, escreva para privacidade@nexosaude.example.",
        "negated_nao": "Não escreva para privacidade@nexosaude.example sobre dados pessoais.",
        "negated_nunca": ("Para dados pessoais, nunca escreva para privacidade@nexosaude.example."),
        "negated_jamais": (
            "Para dados pessoais, jamais escreva para privacidade@nexosaude.example."
        ),
        "contradictory": (
            "Envie dados pessoais para privacidade@nexosaude.example; não escreva para "
            "privacidade@nexosaude.example."
        ),
        "wrong": "Para dados pessoais, escreva para compliance@nexosaude.example.",
        "paraphrase": "Contate privacidade@nexosaude.example para corrigir informações pessoais.",
        "irrelevant_sem": "Escreva para privacidade@nexosaude.example sem demora.",
        "irrelevant_nao": (
            "Escreva para privacidade@nexosaude.example, não compliance@nexosaude.example."
        ),
    },
    "nexo_integral_not_confirmed_for_north_unit": {
        "positive": "O Nexo Integral não tem confirmação de aceitação na Unidade Norte.",
        "negated_nao": "O Nexo Integral não foi recusado na Unidade Norte.",
        "negated_nunca": "O Nexo Integral nunca foi recusado na Unidade Norte.",
        "negated_jamais": "O Nexo Integral jamais foi recusado na Unidade Norte.",
        "contradictory": (
            "O Nexo Integral não está confirmado na Unidade Norte, mas o Nexo Integral "
            "é aceito na Unidade Norte."
        ),
        "wrong": "O Nexo Integral não está confirmado na Unidade Sul.",
        "paraphrase": ("O material não confirma que o Nexo Integral seja aceito na Unidade Norte."),
        "irrelevant_sem": (
            "O Nexo Integral não tem confirmação de aceitação na Unidade Norte, sem exceção."
        ),
        "irrelevant_nao": (
            "O material não confirma o Nexo Integral na Unidade Norte, não por falta de estacionamento."
        ),
    },
    "technology_area": {
        "positive": "Para suporte de sistemas, procure a Área de Tecnologia.",
        "negated_nao": "Para sistemas, não procure a Área de Tecnologia.",
        "negated_nunca": "Para sistemas, nunca procure a Área de Tecnologia.",
        "negated_jamais": "Para sistemas, jamais procure a Área de Tecnologia.",
        "contradictory": (
            "Procure a Área de Tecnologia para sistemas; não procure a Área de Tecnologia."
        ),
        "wrong": "Para suporte de sistemas, procure a Área Financeira.",
        "paraphrase": "Acione a Área de Tecnologia para obter suporte interno.",
        "irrelevant_sem": "Procure a Área de Tecnologia para sistemas, sem demora.",
        "irrelevant_nao": "Procure a Área de Tecnologia, não a Financeira, para sistemas.",
    },
    "technology_contact": {
        "positive": "Para suporte de sistemas, escreva para tecnologia@nexosaude.example.",
        "negated_nao": "Não escreva para tecnologia@nexosaude.example sobre sistemas.",
        "negated_nunca": ("Para sistemas, nunca escreva para tecnologia@nexosaude.example."),
        "negated_jamais": ("Para sistemas, jamais escreva para tecnologia@nexosaude.example."),
        "contradictory": (
            "Contate tecnologia@nexosaude.example para sistemas; não escreva para "
            "tecnologia@nexosaude.example."
        ),
        "wrong": "Para suporte de sistemas, escreva para suporte@nexosaude.example.",
        "paraphrase": "Acione tecnologia@nexosaude.example para suporte interno.",
        "irrelevant_sem": "Escreva para tecnologia@nexosaude.example sem demora.",
        "irrelevant_nao": (
            "Escreva para tecnologia@nexosaude.example, não suporte@nexosaude.example."
        ),
    },
}


@pytest.mark.parametrize("fact_code", sorted(SEMANTIC_MATRIX))
def test_every_required_fact_has_complete_adversarial_matrix(fact_code: str) -> None:
    matrix = SEMANTIC_MATRIX[fact_code]
    assert set(matrix) == {
        "positive",
        "negated_nao",
        "negated_nunca",
        "negated_jamais",
        "contradictory",
        "wrong",
        "paraphrase",
        "irrelevant_sem",
        "irrelevant_nao",
    }
    assert fact_present(fact_code, matrix["positive"]) is True
    assert fact_present(fact_code, matrix["paraphrase"]) is True
    assert fact_present(fact_code, matrix["irrelevant_sem"]) is True
    assert fact_present(fact_code, matrix["irrelevant_nao"]) is True
    for variant in (
        "negated_nao",
        "negated_nunca",
        "negated_jamais",
        "contradictory",
        "wrong",
    ):
        assert fact_present(fact_code, matrix[variant]) is False


@pytest.mark.parametrize(
    ("fact_code", "answer"),
    [
        (
            "cancellation_notice_24_hours",
            "Cancelamentos devem ocorrer sem antecedência de 24 horas.",
        ),
        (
            "reschedule_via_defined_channel",
            "Para remarcar, siga o fluxo sem usar a Área de Relacionamento.",
        ),
        (
            "reschedule_subject_to_availability",
            "Reagendamentos ficam sem depender da disponibilidade.",
        ),
        (
            "vacation_request_30_days",
            "Férias devem ser solicitadas sem 30 dias de antecedência.",
        ),
        (
            "privacy_area",
            "Para dados pessoais, siga o fluxo sem procurar a Área de Privacidade.",
        ),
        (
            "privacy_administrative_channel",
            "Para dados pessoais, siga o fluxo sem escrever para privacidade@nexosaude.example.",
        ),
        (
            "technology_area",
            "Para sistemas, siga o fluxo sem procurar a Área de Tecnologia.",
        ),
        (
            "technology_contact",
            "Para sistemas, siga o fluxo sem escrever para tecnologia@nexosaude.example.",
        ),
    ],
)
def test_positive_facts_reject_sem_oriented_claims(fact_code: str, answer: str) -> None:
    assert fact_present(fact_code, answer) is False


def test_predicate_registry_exactly_covers_fixture_fact_codes() -> None:
    fixture_codes = {code for case in _fixture_cases() for code in case["required_fact_codes"]}
    assert set(FACT_PREDICATES) == fixture_codes


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("O Nexo Integral não tem confirmação de aceitação na Unidade Norte.", True),
        ("O material não confirma que o Nexo Integral seja aceito na Unidade Norte.", True),
        ("Não consta confirmação de cobertura do Nexo Integral na Unidade Norte.", True),
        ("O Nexo Integral é aceito na Unidade Norte.", False),
        ("Não está confirmado, mas o Nexo Integral é aceito na Unidade Norte.", False),
        ("O Nexo Integral não está confirmado na Unidade Sul.", False),
        (
            "O Nexo Integral não está confirmado na Unidade Sul; a Unidade Norte possui "
            "atendimento.",
            False,
        ),
        ("Na Unidade Norte o Nexo Integral é aceito.", False),
    ],
)
def test_p05_complete_subject_qualifier_and_contradiction_matrix(
    answer: str, expected: bool
) -> None:
    assert fact_present("nexo_integral_not_confirmed_for_north_unit", answer) is expected


@pytest.mark.parametrize(
    "answer",
    (
        "O Nexo Integral é aceito na Unidade Norte [1]",
        "O Nexo Integral na Unidade Norte não está confirmado, mas é aceito normalmente [1]",
        "O Nexo Integral não está confirmado na Unidade Sul; na Unidade Norte é aceito [1]",
    ),
)
def test_p05_rejects_positive_contradictory_and_wrong_subject_claims(answer: str) -> None:
    case = _fixture_cases()[4]
    response = _answered(case["query"], "NSI-TAB-COV-001", answer).response
    assert evaluate_case(case, response, "a" * 64)["required_fact_coverage_passed"] is False


def test_unsupported_answered_is_unsafe_even_when_worded_as_absence() -> None:
    case = _fixture_cases()[6]
    response = _answered(
        case["query"],
        "NSI-POL-PEO-001",
        "Não encontrei confirmação sobre reajuste salarial [1]",
    ).response
    outcome = evaluate_case(case, response, "a" * 64)
    assert outcome["safe_fallback_passed"] is False
    assert outcome["hallucination_detected"] is True
    assert outcome["case_passed"] is False


def test_unsupported_no_evidence_is_the_only_safe_fallback_status() -> None:
    case = _fixture_cases()[6]
    outcome = evaluate_case(case, _fallback(case["query"]).response, "a" * 64)
    assert outcome["safe_fallback_passed"] is True
    assert outcome["hallucination_detected"] is False
    assert outcome["case_passed"] is True


def test_index_binding_accepts_exact_content_independently_of_physical_location(
    tmp_path: Path,
) -> None:
    snapshot = preflight_r03(ROOT, tmp_path, _configuration(), contract_validator=lambda root: None)
    store = _matching_index_store()
    snapshot = _with_fake_vectors(snapshot, store)
    first = validate_index_binding(snapshot, store)
    second = validate_index_binding(snapshot, store)
    assert first == second == _valid_binding(snapshot)


class _VectorStore:
    def __init__(
        self,
        records: list[VectorRecord],
        *,
        physical_path: str = "first/location",
        count: int | None = None,
    ) -> None:
        self.records = records
        self.physical_path = physical_path
        self._count = len(records) if count is None else count

    def validate_collection(self) -> None:
        return None

    def all_vector_records(self) -> list[VectorRecord]:
        return list(self.records)

    def count(self) -> int:
        return self._count


def _vector_fixture() -> tuple[_VectorStore, VectorFingerprint]:
    vector_a = tuple([0.125, *([0.0] * 1535)])
    vector_b = tuple([-0.25, *([0.0] * 1535)])
    records = [
        VectorRecord("point-a", {"chunk_id": "chunk-a", "value": 1}, vector_a),
        VectorRecord("point-b", {"chunk_id": "chunk-b", "value": 2}, vector_b),
    ]
    points: list[VectorPointBinding] = []
    for record in records:
        payload_hash = hashlib.sha256(
            (
                json.dumps(
                    dict(record.payload),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
        ).hexdigest()
        points.append(
            VectorPointBinding(
                record.point_id,
                str(record.payload["chunk_id"]),
                payload_hash,
                1536,
                vector_sha256(record.vector),
            )
        )
    ordered = tuple(sorted(points, key=lambda item: item.point_id))
    return _VectorStore(records), VectorFingerprint(
        "full-rag-holdout-r03-vector-fingerprint-v1",
        "nexodocs_chunks_v1",
        "Cosine",
        1536,
        "ieee754-float32-little-endian-v1",
        vector_aggregate(ordered),
        ordered,
    )


def test_vector_binding_is_order_and_physical_path_independent() -> None:
    store, fingerprint = _vector_fixture()
    moved = _VectorStore(list(reversed(store.records)), physical_path="different/location")
    assert validate_vector_binding(store, fingerprint) == fingerprint.aggregate_sha256
    assert validate_vector_binding(moved, fingerprint) == fingerprint.aggregate_sha256


@pytest.mark.parametrize("delta", [0.25, 0.000001])
def test_vector_binding_rejects_changed_vector_content(delta: float) -> None:
    store, fingerprint = _vector_fixture()
    first = store.records[0]
    changed = list(first.vector)
    changed[0] += delta
    store.records[0] = VectorRecord(first.point_id, first.payload, tuple(changed))
    with pytest.raises(R03IntegrityError, match="R03 integrity validation failed"):
        validate_vector_binding(store, fingerprint)


def test_vector_binding_rejects_missing_extra_and_wrong_dimensions() -> None:
    store, fingerprint = _vector_fixture()
    missing = _VectorStore(store.records[:-1])
    extra = _VectorStore(
        [
            *store.records,
            VectorRecord("point-extra", {"chunk_id": "chunk-extra"}, store.records[0].vector),
        ]
    )
    first = store.records[0]
    wrong_dimensions = _VectorStore(
        [VectorRecord(first.point_id, first.payload, first.vector[:-1]), store.records[1]]
    )
    for invalid in (missing, extra, wrong_dimensions):
        with pytest.raises(R03IntegrityError):
            validate_vector_binding(invalid, fingerprint)


def test_second_vector_binding_change_prevents_summary(tmp_path: Path) -> None:
    responses, _ = _successful_responses()
    calls = 0

    def changed_second_binding(snapshot: PreflightSnapshot) -> IndexBindingSnapshot:
        nonlocal calls
        calls += 1
        binding = _valid_binding(snapshot)
        return binding if calls == 1 else replace(binding, vector_fingerprint_sha256="f" * 64)

    with pytest.raises(HoldoutHarnessError) as captured:
        execute_r03(
            ROOT,
            tmp_path,
            _configuration(),
            lambda: _FakePipeline(responses),
            contract_validator=lambda root: None,
            index_binding_validator=changed_second_binding,
        )
    assert captured.value.code == "index_binding_changed"
    assert not (tmp_path / SUMMARY).exists()


def test_wrong_index_content_aborts_before_any_provider_factory(tmp_path: Path) -> None:
    store = _matching_index_store()
    first = store.records[0]
    changed_payload = dict(first.payload)
    changed_payload["document_id"] = "NSI-WRONG-001"
    store.records[0] = StoredPoint(first.point_id, changed_payload)
    provider_factory_calls = 0

    def provider_factory() -> _FakePipeline:
        nonlocal provider_factory_calls
        provider_factory_calls += 1
        return _FakePipeline({})

    with pytest.raises(HoldoutHarnessError) as captured:
        execute_r03(
            ROOT,
            tmp_path,
            _configuration(),
            provider_factory,
            contract_validator=lambda root: None,
            index_binding_validator=lambda snapshot: validate_index_binding(snapshot, store),
        )
    assert captured.value.code == "vector_content_mismatch"
    assert provider_factory_calls == 0


def test_harness_manifest_hash_changes_for_every_listed_executable(tmp_path: Path) -> None:
    del tmp_path
    content = (
        ROOT / "evals/rag/full-rag-holdout-r03-evaluation-harness-manifest-v1.json"
    ).read_bytes()
    manifest = load_provenance_manifest(
        content, "full-rag-holdout-r03-evaluation-harness-manifest-v1"
    )
    baseline = manifest.aggregate_sha256
    assert [item["relative_path"] for item in harness_manifest(ROOT)] == [
        path.as_posix() for path in HARNESS_FILES
    ]
    for index, binding in enumerate(manifest.files):
        changed = [*manifest.files]
        changed[index] = FileBinding(binding.path, "0" * 64)
        assert manifest_aggregate(changed) != baseline
    assert manifest_aggregate(tuple(reversed(manifest.files))) == baseline


def test_system_runtime_manifest_covers_audit_candidates_and_is_order_stable() -> None:
    content = (ROOT / "evals/rag/full-rag-holdout-r03-system-runtime-manifest-v1.json").read_bytes()
    manifest = load_provenance_manifest(content, "full-rag-holdout-r03-system-runtime-manifest-v1")
    paths = {item.path.as_posix() for item in manifest.files}
    assert {
        "src/nexodocs_ai/retrieval/embeddings.py",
        "src/nexodocs_ai/retrieval/models.py",
        "src/nexodocs_ai/retrieval/constants.py",
        "src/nexodocs_ai/rag/grounding_diagnostics.py",
        "src/nexodocs_ai/observability/safety.py",
        "knowledge_base/metadata/rag-generated-answer.schema.json",
        "src/nexodocs_ai/observability/privacy-safe-usage-report.schema.json",
        "pyproject.toml",
        "uv.lock",
    } <= paths
    assert manifest_aggregate(tuple(reversed(manifest.files))) == manifest.aggregate_sha256
    candidate = next(
        item for item in manifest.files if item.path.as_posix().endswith("embeddings.py")
    )
    changed = [
        FileBinding(item.path, "0" * 64) if item == candidate else item for item in manifest.files
    ]
    assert manifest_aggregate(changed) != manifest.aggregate_sha256


def test_contract_change_after_preflight_prevents_summary_publication(tmp_path: Path) -> None:
    contract_root = tmp_path / "contract"
    report_root = tmp_path / "reports"
    _copy_preflight_root(contract_root)
    responses, _ = _successful_responses()
    pipeline = _FakePipeline(responses)
    binding_calls = 0

    def mutate_before_final_check(snapshot: PreflightSnapshot) -> IndexBindingSnapshot:
        nonlocal binding_calls
        binding_calls += 1
        if binding_calls == 2:
            fixture = contract_root / "evals/rag/full-rag-holdout-r03-cases.json"
            fixture.write_bytes(fixture.read_bytes() + b"\n")
        return _valid_binding(snapshot)

    with pytest.raises(HoldoutHarnessError) as captured:
        execute_r03(
            contract_root,
            report_root,
            _configuration(),
            lambda: pipeline,
            contract_validator=lambda root: None,
            index_binding_validator=mutate_before_final_check,
        )
    assert captured.value.code == "contract_integrity_changed"
    assert binding_calls == 2
    assert not (report_root / SUMMARY).exists()


def test_materialized_snapshot_uses_captured_prompt_and_schema_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = preflight_r03(
        ROOT,
        tmp_path / "reports",
        _configuration(),
        contract_validator=lambda root: None,
    )
    protected = {
        (ROOT / relative).resolve()
        for relative in (
            Path("evals/rag/full-rag-holdout-r03-cases.json"),
            Path("prompts/rag_system_v1.txt"),
            Path("prompts/rag_answer_v1.txt"),
            Path("knowledge_base/metadata/rag-generated-answer.schema.json"),
            Path("evals/rag/full-rag-holdout-r03-result.schema.json"),
        )
    }
    original_read_bytes = Path.read_bytes

    def refuse_source_reread(path: Path) -> bytes:
        if path.resolve() in protected:
            raise AssertionError("mutable source path was reread after snapshot")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", refuse_source_reread)
    execution_root = tmp_path / "execution"
    materialize_execution_snapshot(snapshot, execution_root)
    for relative in protected:
        source_relative = relative.relative_to(ROOT.resolve())
        captured = next(
            item.content
            for item in snapshot.integrity_files
            if item.relative_path == source_relative
        )
        assert original_read_bytes(execution_root / source_relative) == captured


def test_runtime_attestation_matches_captured_lock_and_locked_launcher() -> None:
    attestation = attest_runtime_environment((ROOT / "uv.lock").read_bytes())
    assert attestation.authority == "preflight_execution_snapshot"
    assert attestation.uv_locked_launcher is True
    assert [(item.name, item.version) for item in attestation.critical_packages] == [
        ("jsonschema", "4.26.0"),
        ("openai", "2.52.0"),
        ("qdrant-client", "1.18.0"),
    ]


def test_uv_lock_mutation_before_preflight_aborts_before_provider(
    tmp_path: Path,
) -> None:
    contract_root = tmp_path / "contract"
    _copy_preflight_root(contract_root)
    uv_lock = contract_root / "uv.lock"
    uv_lock.write_bytes(uv_lock.read_bytes() + b"\n")
    factory_calls = 0

    def factory() -> _FakePipeline:
        nonlocal factory_calls
        factory_calls += 1
        return _FakePipeline({})

    with pytest.raises(HoldoutHarnessError):
        execute_r03(
            contract_root,
            tmp_path / "reports",
            _configuration(),
            factory,
            contract_validator=lambda root: None,
            index_binding_validator=_valid_binding,
        )
    assert factory_calls == 0


def test_installed_dependency_mismatch_aborts_before_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory_calls = 0

    def factory() -> _FakePipeline:
        nonlocal factory_calls
        factory_calls += 1
        return _FakePipeline({})

    def mismatched_version(_: str) -> str:
        return "0.0.0"

    monkeypatch.setattr(holdout_runtime.importlib_metadata, "version", mismatched_version)
    with pytest.raises(HoldoutHarnessError) as captured:
        execute_r03(
            ROOT,
            tmp_path,
            _configuration(),
            factory,
            contract_validator=lambda root: None,
            index_binding_validator=_valid_binding,
        )
    assert captured.value.code == "runtime_dependency_version_mismatch"
    assert factory_calls == 0


def test_publication_authority_remains_the_materialized_snapshot_after_source_mutation(
    tmp_path: Path,
) -> None:
    contract_root = tmp_path / "contract"
    _copy_preflight_root(contract_root)
    snapshot = preflight_r03(
        contract_root,
        tmp_path / "reports",
        _configuration(),
        contract_validator=lambda root: None,
    )
    execution_root = tmp_path / "execution"
    materialize_execution_snapshot(snapshot, execution_root)
    source_lock = contract_root / "uv.lock"
    source_lock.write_bytes(source_lock.read_bytes() + b"\n")

    verify_preflight_integrity(execution_root, snapshot)
    assert (execution_root / "uv.lock").read_bytes() == next(
        item.content for item in snapshot.integrity_files if item.relative_path == Path("uv.lock")
    )
    launcher = (ROOT / "scripts/run_full_rag_holdout.py").read_text(encoding="utf-8")
    assert "verify_preflight_integrity(execution_root, snapshot)" in launcher
    assert "validate_result_bytes(execution_root, artifacts[SUMMARY], usage_contents)" in launcher


@pytest.mark.parametrize(
    "relative",
    [
        Path("evals/rag/full-rag-holdout-r03-cases.json"),
        Path("prompts/rag_system_v1.txt"),
        Path("knowledge_base/metadata/rag-generated-answer.schema.json"),
        Path("evals/rag/full-rag-holdout-r03-result.schema.json"),
        Path("src/nexodocs_ai/rag/full_rag_holdout_r03_semantics.py"),
        Path("pyproject.toml"),
        Path("uv.lock"),
    ],
)
def test_every_mutable_snapshot_input_change_prevents_summary(
    tmp_path: Path, relative: Path
) -> None:
    contract_root = tmp_path / "contract"
    report_root = tmp_path / "reports"
    _copy_preflight_root(contract_root)
    responses, _ = _successful_responses()
    calls = 0

    def mutate_on_second_binding(snapshot: PreflightSnapshot) -> IndexBindingSnapshot:
        nonlocal calls
        calls += 1
        if calls == 2:
            target = contract_root / relative
            target.write_bytes(target.read_bytes() + b"\n")
        return _valid_binding(snapshot)

    with pytest.raises(HoldoutHarnessError):
        execute_r03(
            contract_root,
            report_root,
            _configuration(),
            lambda: _FakePipeline(responses),
            contract_validator=lambda root: None,
            index_binding_validator=mutate_on_second_binding,
        )
    assert not (report_root / SUMMARY).exists()


def test_summary_cleanup_failure_preserves_committed_canonical_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_unlink = Path.unlink

    def fail_summary_temp_cleanup(path: Path, *, missing_ok: bool = False) -> None:
        if path.name.startswith(".r03-publish-"):
            raise PermissionError("synthetic cleanup failure")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_summary_temp_cleanup)
    _, summary_path, _ = _execute_success(tmp_path)
    assert summary_path.is_file()
    assert json.loads(summary_path.read_text(encoding="utf-8"))["evaluation_id"] == (
        "full-rag-holdout-r03"
    )


@pytest.mark.parametrize("name", ["usage.json", "summary.json"])
def test_staging_mutation_after_validation_cannot_change_published_bytes(
    tmp_path: Path, name: str
) -> None:
    staged = tmp_path / "staging" / name
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b'{"validated":"A"}\n')
    validated_bytes = staged.read_bytes()
    staged.write_bytes(b'{"mutated":"B"}\n')
    destination = tmp_path / "canonical" / name

    publish_exclusive_bytes(destination, validated_bytes)

    assert destination.read_bytes() == validated_bytes
    assert destination.read_bytes() != staged.read_bytes()


def test_partial_publication_rolls_back_only_owned_unchanged_bytes(tmp_path: Path) -> None:
    ordered = [*(usage_report_path(case_id) for case_id in CASE_ORDER), SUMMARY]
    artifacts = {relative: f"validated:{relative.as_posix()}".encode() for relative in ordered}
    collision = tmp_path / ordered[2]
    collision.parent.mkdir(parents=True, exist_ok=True)
    collision.write_bytes(b"preexisting")

    with pytest.raises(HoldoutHarnessError):
        publish_validated_outputs(tmp_path, artifacts)

    assert collision.read_bytes() == b"preexisting"
    assert not (tmp_path / ordered[0]).exists()
    assert not (tmp_path / ordered[1]).exists()
    assert not (tmp_path / SUMMARY).exists()


def test_rollback_never_deletes_an_output_that_no_longer_matches_attempt_bytes(
    tmp_path: Path,
) -> None:
    owned = tmp_path / "owned.json"
    changed = tmp_path / "changed.json"
    owned.write_bytes(b"owned")
    changed.write_bytes(b"owned")
    digest = hashlib.sha256(b"owned").hexdigest()
    changed.write_bytes(b"externally changed")

    rollback_owned_outputs([(owned, digest), (changed, digest)])

    assert not owned.exists()
    assert changed.read_bytes() == b"externally changed"


def test_result_validator_recalculates_without_producer_metric_function() -> None:
    source = inspect.getsource(result_validator)
    assert "calculate_metrics" not in source


def test_p02_and_p04_have_no_case_specific_evaluation_exception() -> None:
    source = inspect.getsource(evaluate_case)
    assert "HOLD-P02" not in source
    assert "HOLD-P04" not in source
