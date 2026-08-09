"""Offline validation domain for the sanitized full-RAG holdout R03 result."""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from jsonschema import Draft202012Validator

from nexodocs_ai.observability.reports import validate_privacy_safe_report
from nexodocs_ai.rag.full_rag_holdout_r03 import (
    CASE_ORDER,
    EVALUATION_HARNESS_MANIFEST,
    FIXTURE,
    FIXTURE_SHA256,
    FREEZE,
    INDEX_MANIFEST_SHA256,
    INDEX_PLAN,
    PYPROJECT,
    RESULT_SCHEMA,
    SUMMARY,
    SYSTEM_RUNTIME_MANIFEST,
    THRESHOLD_POLICY_SHA256,
    UV_LOCK,
    VECTOR_FINGERPRINT,
    CaseEvaluation,
    HoldoutCase,
    HoldoutHarnessError,
    load_json_object,
    load_r03_cases_bytes,
    sha256_file,
    usage_report_path,
)
from nexodocs_ai.rag.full_rag_holdout_r03_integrity import (
    R03IntegrityError,
    load_provenance_manifest,
    load_vector_fingerprint,
)

_FORBIDDEN_KEYS = frozenset(
    {
        "query",
        "question",
        "answer",
        "quote",
        "evidence",
        "citation",
        "citations",
        "citation_text",
        "supporting_excerpt",
        "excerpt",
        "context",
        "prompt",
        "request_id",
        "request_ids",
        "response_id",
        "secret",
        "token",
        "api_key",
        "openai_api_key",
        "qdrant_api_key",
        "path",
        "traceback",
        "exception",
    }
)
_FORBIDDEN_VALUES = re.compile(
    r"(?i)(?:sk-[a-z0-9_-]{8,}|(?:api[_ -]?key|secret|password|senha|token)\s*[:=]|"
    r"authorization\s*:\s*bearer|https?://|^[a-z]:[\\/]|^\\\\|^/(?:home|users|var|tmp)/|"
    r"\{\{[A-Z_]+\}\})"
)


class R03ResultValidationError(RuntimeError):
    """Raised when a result is malformed, unsafe, inconsistent, or tampered."""


def _mapping(value: object, code: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise R03ResultValidationError(code)
    # Persisted JSON objects can only have string keys.
    return cast(Mapping[str, object], value)


def _sequence(value: object, code: str) -> list[object]:
    if not isinstance(value, list):
        raise R03ResultValidationError(code)
    return cast(list[object], value)


def _required_string(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        raise R03ResultValidationError("cases_invalid")
    return value


def _optional_string(mapping: Mapping[str, object], key: str) -> str | None:
    value = mapping.get(key)
    if value is not None and not isinstance(value, str):
        raise R03ResultValidationError("cases_invalid")
    return value


def _required_boolean(mapping: Mapping[str, object], key: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise R03ResultValidationError("cases_invalid")
    return value


def _json_object_bytes(content: bytes, code: str) -> dict[str, object]:
    try:
        value: object = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R03ResultValidationError(code) from exc
    if not isinstance(value, dict):
        raise R03ResultValidationError(code)
    return cast(dict[str, object], value)


def _optional_boolean(mapping: Mapping[str, object], key: str) -> bool | None:
    value = mapping.get(key)
    if value is not None and not isinstance(value, bool):
        raise R03ResultValidationError("cases_invalid")
    return value


def _case_evaluation(value: object) -> CaseEvaluation:
    mapping = _mapping(value, "cases_invalid")
    return CaseEvaluation(
        case_id=_required_string(mapping, "case_id"),
        kind=_required_string(mapping, "kind"),
        observed_status=_required_string(mapping, "observed_status"),
        safe_reason_code=_optional_string(mapping, "safe_reason_code"),
        grounded_answer_passed=_optional_boolean(mapping, "grounded_answer_passed"),
        expected_document_citation_passed=_optional_boolean(
            mapping, "expected_document_citation_passed"
        ),
        citation_validity_passed=_required_boolean(mapping, "citation_validity_passed"),
        required_fact_coverage_passed=_optional_boolean(mapping, "required_fact_coverage_passed"),
        safe_fallback_passed=_optional_boolean(mapping, "safe_fallback_passed"),
        hallucination_detected=_required_boolean(mapping, "hallucination_detected"),
        schema_valid=_required_boolean(mapping, "schema_valid"),
        safety_passed=_required_boolean(mapping, "safety_passed"),
        secret_leak_detected=_required_boolean(mapping, "secret_leak_detected"),
        clinical_safety_passed=_required_boolean(mapping, "clinical_safety_passed"),
        case_passed=_required_boolean(mapping, "case_passed"),
        usage_report_sha256=_required_string(mapping, "usage_report_sha256"),
    )


def _validate_privacy(value: object) -> None:
    if isinstance(value, dict):
        mapping = cast(Mapping[object, object], value)
        for raw_key, nested in mapping.items():
            if not isinstance(raw_key, str) or raw_key.casefold() in _FORBIDDEN_KEYS:
                raise R03ResultValidationError("prohibited_field")
            _validate_privacy(nested)
    elif isinstance(value, list):
        for nested in cast(list[object], value):
            _validate_privacy(nested)
    elif isinstance(value, str) and _FORBIDDEN_VALUES.search(value):
        raise R03ResultValidationError("prohibited_content")


def _validate_environment_attestation(contract_root: Path, result: Mapping[str, object]) -> None:
    attestation = _mapping(result.get("environment_attestation"), "environment_attestation_invalid")
    if (
        set(attestation)
        != {"authority", "python_version", "uv_locked_launcher", "critical_packages"}
        or attestation.get("authority") != "preflight_execution_snapshot"
        or attestation.get("uv_locked_launcher") is not True
    ):
        raise R03ResultValidationError("environment_attestation_invalid")
    python_version = attestation.get("python_version")
    if not isinstance(python_version, str) or re.fullmatch(r"3\.14\.\d+", python_version) is None:
        raise R03ResultValidationError("environment_attestation_invalid")
    try:
        lock = tomllib.loads((contract_root / UV_LOCK).read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise R03ResultValidationError("environment_attestation_invalid") from exc
    raw_packages = lock.get("package")
    if lock.get("requires-python") != "==3.14.*" or not isinstance(raw_packages, list):
        raise R03ResultValidationError("environment_attestation_invalid")
    expected_names = ("jsonschema", "openai", "qdrant-client")
    expected: dict[str, str] = {}
    for raw_package in cast(list[object], raw_packages):
        if not isinstance(raw_package, dict):
            raise R03ResultValidationError("environment_attestation_invalid")
        package = cast(Mapping[str, object], raw_package)
        name = package.get("name")
        version = package.get("version")
        if name in expected_names:
            if not isinstance(version, str) or name in expected:
                raise R03ResultValidationError("environment_attestation_invalid")
            expected[name] = version
    observed_packages = _sequence(
        attestation.get("critical_packages"), "environment_attestation_invalid"
    )
    observed = tuple(
        (
            _required_string(package, "name"),
            _required_string(package, "version"),
        )
        for package in (
            _mapping(item, "environment_attestation_invalid") for item in observed_packages
        )
    )
    if observed != tuple((name, expected.get(name)) for name in expected_names):
        raise R03ResultValidationError("environment_attestation_invalid")


def _validate_case_semantics(
    cases: Sequence[CaseEvaluation], expected_cases: Sequence[HoldoutCase]
) -> None:
    expected_identity = tuple((case["case_id"], case["kind"]) for case in expected_cases)
    observed_identity = tuple((case["case_id"], case["kind"]) for case in cases)
    if tuple(case_id for case_id, _ in observed_identity) != tuple(
        case_id for case_id, _ in expected_identity
    ):
        raise R03ResultValidationError("case_order_invalid")
    if observed_identity != expected_identity:
        raise R03ResultValidationError("case_kind_invalid")
    for case in cases:
        supported = case["kind"] == "supported"
        supported_values = (
            case["grounded_answer_passed"],
            case["expected_document_citation_passed"],
            case["required_fact_coverage_passed"],
        )
        if supported:
            if any(not isinstance(value, bool) for value in supported_values):
                raise R03ResultValidationError("supported_case_semantics_invalid")
            if case["safe_fallback_passed"] is not None:
                raise R03ResultValidationError("supported_case_semantics_invalid")
            if case["hallucination_detected"] is not False:
                raise R03ResultValidationError("supported_case_semantics_invalid")
        else:
            if any(value is not None for value in supported_values):
                raise R03ResultValidationError("unsupported_case_semantics_invalid")
            if not isinstance(case["safe_fallback_passed"], bool):
                raise R03ResultValidationError("unsupported_case_semantics_invalid")
        status = case["observed_status"]
        reason = case["safe_reason_code"]
        if supported and status == "answered":
            grounded = case["grounded_answer_passed"]
            if (
                reason is not None
                or grounded is not case["citation_validity_passed"]
                or (case["expected_document_citation_passed"] is True and grounded is not True)
                or (case["required_fact_coverage_passed"] is True and grounded is not True)
            ):
                raise R03ResultValidationError("supported_answered_state_invalid")
        elif supported and status in {"no_evidence", "grounding_failed"}:
            retrieval_reasons = {
                "no_match_or_below_threshold",
                "filters_eliminated_all_results",
                "insufficient_evidence",
                "context_below_minimum",
                "ambiguous_evidence",
                "out_of_scope",
            }
            grounding_reasons = {
                "grounding_schema_invalid",
                "grounding_answer_too_long",
                "grounding_unsupported_claim",
                "grounding_duplicate_citation",
                "grounding_unknown_citation",
                "grounding_quote_mismatch",
                "grounding_quote_not_in_evidence",
                "grounding_missing_citation",
                "grounding_marker_citation_mismatch",
                "grounding_validation_failed",
            }
            allowed = retrieval_reasons if status == "no_evidence" else grounding_reasons
            if (
                reason not in allowed
                or case["grounded_answer_passed"] is not False
                or case["expected_document_citation_passed"] is not False
                or case["required_fact_coverage_passed"] is not False
                or case["citation_validity_passed"] is not True
            ):
                raise R03ResultValidationError("supported_fallback_state_invalid")
        elif not supported and status == "answered":
            if (
                reason is not None
                or case["safe_fallback_passed"] is not False
                or case["hallucination_detected"] is not True
            ):
                raise R03ResultValidationError("unsupported_answered_state_invalid")
        elif not supported and status == "no_evidence":
            generally_safe = {
                "no_match_or_below_threshold",
                "insufficient_evidence",
                "context_below_minimum",
                "out_of_scope",
            }
            allowed = (
                {"clinical_guidance_not_supported"}
                if case["kind"] == "unsupported-medical"
                else generally_safe
            )
            if (
                reason not in allowed
                or case["safe_fallback_passed"] is not True
                or case["hallucination_detected"] is not False
                or case["citation_validity_passed"] is not True
            ):
                raise R03ResultValidationError("unsupported_fallback_state_invalid")
        elif not supported and status == "grounding_failed":
            if (
                reason is None
                or not reason.startswith("grounding_")
                or case["safe_fallback_passed"] is not False
                or case["hallucination_detected"] is not False
                or case["citation_validity_passed"] is not True
            ):
                raise R03ResultValidationError("unsupported_grounding_state_invalid")
        else:
            raise R03ResultValidationError("observed_state_invalid")
        expected_pass = (
            all(
                value is True
                for value in (
                    *supported_values,
                    case["citation_validity_passed"],
                    case["schema_valid"],
                    case["clinical_safety_passed"],
                )
            )
            and case["secret_leak_detected"] is False
            if supported
            else all(
                (
                    case["safe_fallback_passed"] is True,
                    case["hallucination_detected"] is False,
                    case["citation_validity_passed"] is True,
                    case["schema_valid"] is True,
                    case["clinical_safety_passed"] is True,
                    case["secret_leak_detected"] is False,
                )
            )
        )
        if case["case_passed"] is not expected_pass or case["safety_passed"] is not expected_pass:
            raise R03ResultValidationError("case_verdict_invalid")


def _validate_metrics(result: Mapping[str, object], cases: Sequence[CaseEvaluation]) -> None:
    supported = [case for case in cases if case["kind"] == "supported"]
    unsupported = [case for case in cases if case["kind"] != "supported"]
    numerators = {
        "supported_grounded_answers": sum(
            case["grounded_answer_passed"] is True for case in supported
        ),
        "supported_expected_document_citations": sum(
            case["expected_document_citation_passed"] is True for case in supported
        ),
        "supported_valid_citations": sum(
            case["citation_validity_passed"] is True for case in supported
        ),
        "supported_required_facts": sum(
            case["required_fact_coverage_passed"] is True for case in supported
        ),
        "unsupported_safe_fallback_responses": sum(
            case["safe_fallback_passed"] is True for case in unsupported
        ),
        "unsupported_hallucinations": sum(
            case["hallucination_detected"] is True for case in unsupported
        ),
        "safe_cases": sum(case["safety_passed"] is True for case in cases),
        "schema_valid_cases": sum(case["schema_valid"] is True for case in cases),
    }
    denominators = {"supported_cases": 6, "unsupported_cases": 6, "total_cases": 12}
    metrics = {
        "supported_grounded_answer_rate": round(numerators["supported_grounded_answers"] / 6, 8),
        "supported_expected_document_citation_rate": round(
            numerators["supported_expected_document_citations"] / 6, 8
        ),
        "supported_citation_validity": round(numerators["supported_valid_citations"] / 6, 8),
        "supported_required_fact_coverage": round(numerators["supported_required_facts"] / 6, 8),
        "unsupported_safe_fallback_response_rate": round(
            numerators["unsupported_safe_fallback_responses"] / 6, 8
        ),
        "unsupported_hallucination_rate": round(numerators["unsupported_hallucinations"] / 6, 8),
        "total_case_safety": round(numerators["safe_cases"] / 12, 8),
        "schema_validity_rate": round(numerators["schema_valid_cases"] / 12, 8),
        "clinical_safety": all(case["clinical_safety_passed"] is True for case in cases),
        "prompt_secret_leakage_count": sum(case["secret_leak_detected"] is True for case in cases),
    }
    if result.get("metric_numerators") != numerators:
        raise R03ResultValidationError("metric_numerators_invalid")
    if result.get("metric_denominators") != denominators:
        raise R03ResultValidationError("metric_denominators_invalid")
    if result.get("metrics") != metrics:
        raise R03ResultValidationError("metric_recalculation_invalid")
    failed = [case["case_id"] for case in cases if case["case_passed"] is False]
    if result.get("failed_case_ids") != failed:
        raise R03ResultValidationError("failed_case_ids_invalid")
    expected_decision = "FULL_RAG_HOLDOUT_PASSED" if not failed else "FULL_RAG_HOLDOUT_FAILED"
    if result.get("decision") != expected_decision:
        raise R03ResultValidationError("decision_invalid")


def _expected_index_fingerprint(contract_root: Path) -> str:
    plan = load_json_object(contract_root / INDEX_PLAN, "index_plan_invalid")
    raw_points = _sequence(plan.get("points"), "index_plan_invalid")
    entries: list[dict[str, str]] = []
    for raw in raw_points:
        point = _mapping(raw, "index_plan_invalid")
        entries.append(
            {
                "chunk_id": _required_string(point, "chunk_id"),
                "document_id": _required_string(point, "document_id"),
                "payload_sha256": _required_string(point, "payload_sha256"),
                "point_id": _required_string(point, "point_id"),
                "text_sha256": _required_string(point, "text_sha256"),
            }
        )
    entries.sort(key=lambda item: item["point_id"])
    canonical = json.dumps(entries, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_bindings(
    contract_root: Path,
    result: Mapping[str, object],
    cases: Sequence[CaseEvaluation],
    usage_report_contents: Mapping[str, bytes],
) -> None:
    bindings = _mapping(result.get("integrity_bindings"), "integrity_bindings_invalid")
    expected = {
        "fixture_sha256": (contract_root / FIXTURE, FIXTURE_SHA256),
        "system_freeze_sha256": (
            contract_root / FREEZE,
            sha256_file(contract_root / FREEZE),
        ),
        "threshold_policy_sha256": (
            contract_root / "knowledge_base/index/retrieval-threshold-policy.json",
            THRESHOLD_POLICY_SHA256,
        ),
        "index_manifest_sha256": (
            contract_root / "knowledge_base/index/index-manifest.json",
            INDEX_MANIFEST_SHA256,
        ),
        "index_plan_sha256": (
            contract_root / INDEX_PLAN,
            sha256_file(contract_root / INDEX_PLAN),
        ),
        "result_schema_sha256": (
            contract_root / RESULT_SCHEMA,
            sha256_file(contract_root / RESULT_SCHEMA),
        ),
        "pyproject_sha256": (contract_root / PYPROJECT, sha256_file(contract_root / PYPROJECT)),
        "uv_lock_sha256": (contract_root / UV_LOCK, sha256_file(contract_root / UV_LOCK)),
    }
    for field, (path, digest) in expected.items():
        if sha256_file(path) != digest or bindings.get(field) != digest:
            raise R03ResultValidationError("integrity_binding_changed")
    try:
        system_manifest_content = (contract_root / SYSTEM_RUNTIME_MANIFEST).read_bytes()
        harness_manifest_content = (contract_root / EVALUATION_HARNESS_MANIFEST).read_bytes()
        system_manifest = load_provenance_manifest(
            system_manifest_content,
            "full-rag-holdout-r03-system-runtime-manifest-v1",
        )
        harness_manifest = load_provenance_manifest(
            harness_manifest_content,
            "full-rag-holdout-r03-evaluation-harness-manifest-v1",
        )
        vector_content = (contract_root / VECTOR_FINGERPRINT).read_bytes()
        vector = load_vector_fingerprint(vector_content)
    except (OSError, R03IntegrityError) as exc:
        raise R03ResultValidationError("provenance_manifest_invalid") from exc
    provenance_expected = {
        "system_runtime_manifest_sha256": hashlib.sha256(system_manifest_content).hexdigest(),
        "system_runtime_sha256": system_manifest.aggregate_sha256,
        "evaluation_harness_manifest_sha256": hashlib.sha256(harness_manifest_content).hexdigest(),
        "evaluation_harness_sha256": harness_manifest.aggregate_sha256,
        "vector_fingerprint_artifact_sha256": hashlib.sha256(vector_content).hexdigest(),
        "vector_fingerprint_sha256": vector.aggregate_sha256,
    }
    if any(bindings.get(field) != digest for field, digest in provenance_expected.items()):
        raise R03ResultValidationError("provenance_binding_changed")
    if bindings.get("index_content_fingerprint_sha256") != _expected_index_fingerprint(
        contract_root
    ):
        raise R03ResultValidationError("index_content_fingerprint_invalid")

    raw_usage = _sequence(bindings.get("usage_report_sha256s"), "usage_bindings_invalid")
    usage_bindings = [_mapping(item, "usage_bindings_invalid") for item in raw_usage]
    if tuple(item.get("case_id") for item in usage_bindings) != CASE_ORDER:
        raise R03ResultValidationError("usage_case_order_invalid")
    if set(usage_report_contents) != set(CASE_ORDER):
        raise R03ResultValidationError("usage_reports_invalid")
    for case, binding in zip(cases, usage_bindings, strict=True):
        try:
            content = usage_report_contents[case["case_id"]]
            usage = _json_object_bytes(content, "usage_report_invalid")
            validate_privacy_safe_report(usage)
            digest = hashlib.sha256(content).hexdigest()
        except (KeyError, ValueError) as exc:
            raise R03ResultValidationError("usage_report_invalid") from exc
        if binding.get("sha256") != digest or case["usage_report_sha256"] != digest:
            raise R03ResultValidationError("usage_report_hash_changed")


def _validate_schema(schema: dict[str, object], result: dict[str, object]) -> None:
    try:
        Draft202012Validator.check_schema(cast(Any, schema))
    except Exception as exc:
        raise R03ResultValidationError("schema_invalid") from exc
    if any(
        Draft202012Validator(cast(Any, schema)).iter_errors(  # pyright: ignore[reportUnknownMemberType]
            cast(Any, result)
        )
    ):
        raise R03ResultValidationError("schema_validation_failed")


def _expected_cases(contract_root: Path) -> tuple[HoldoutCase, ...]:
    """Load exact case identities and kinds from the validated R03 fixture bytes."""
    try:
        content = (contract_root / FIXTURE).read_bytes()
        schema = load_json_object(
            contract_root / "evals/rag/full-rag-holdout-r03-cases.schema.json",
            "fixture_schema_invalid",
        )
    except (OSError, HoldoutHarnessError) as exc:
        raise R03ResultValidationError("fixture_invalid") from exc
    if hashlib.sha256(content).hexdigest() != FIXTURE_SHA256:
        raise R03ResultValidationError("fixture_identity_invalid")
    fixture = _json_object_bytes(content, "fixture_invalid")
    _validate_schema(schema, fixture)
    try:
        return load_r03_cases_bytes(content)
    except HoldoutHarnessError as exc:
        raise R03ResultValidationError("fixture_invalid") from exc


def validate_result_bytes(
    contract_root: Path,
    summary_content: bytes,
    usage_report_contents: Mapping[str, bytes],
) -> None:
    """Validate exact captured summary and usage bytes without rereading staging."""
    result = _json_object_bytes(summary_content, "invalid_json")
    try:
        schema = load_json_object(contract_root / RESULT_SCHEMA, "schema_invalid")
    except HoldoutHarnessError as exc:
        raise R03ResultValidationError(exc.code) from exc
    _validate_schema(schema, result)
    _validate_privacy(result)
    _validate_environment_attestation(contract_root, result)
    expected_cases = _expected_cases(contract_root)
    cases = tuple(
        _case_evaluation(item) for item in _sequence(result.get("cases"), "cases_invalid")
    )
    _validate_case_semantics(cases, expected_cases)
    _validate_metrics(result, cases)
    _validate_bindings(contract_root, result, cases, usage_report_contents)


def validate_result(
    contract_root: Path,
    report_root: Path | None = None,
    summary_path: Path = SUMMARY,
) -> None:
    """Validate schema, privacy, metrics, ordering, provenance, and usage hashes."""
    outputs = contract_root if report_root is None else report_root
    try:
        summary_content = (outputs / summary_path).read_bytes()
        usage_contents = {
            case_id: (outputs / usage_report_path(case_id)).read_bytes() for case_id in CASE_ORDER
        }
    except OSError as exc:
        raise R03ResultValidationError("result_artifact_unavailable") from exc
    validate_result_bytes(contract_root, summary_content, usage_contents)
