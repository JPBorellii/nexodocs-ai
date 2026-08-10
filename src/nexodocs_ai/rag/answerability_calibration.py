"""Deterministic development-calibration harness for answerability providers."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Never, cast

from jsonschema import Draft202012Validator

from .answerability_prompts import ANSWERABILITY_PROMPT_VERSION, answerability_prompt_sha256
from .models import (
    AnswerabilityDecision,
    AnswerabilityProvider,
    AnswerabilityProviderError,
    AnswerabilityProviderRefusalError,
    AnswerabilityRequest,
    EvidenceBlock,
    GenerationUsage,
)

EXPECTED_FIXTURE_ID = "nexodocs-answerability-calibration-v1"
EXPECTED_FIXTURE_SCHEMA_VERSION = "1.0.0"
EXPECTED_ORACLE_SHA256 = "cb39f40f1d13cb09ecd042d81d99299eb2bcb0657602c47c5dcd2cc7d0d4d216"
EXPECTED_CORPUS_SHA256 = "a49ad2a4c3b8dc767277566be60e93cd2a8d3113ffe1c598eec9ccfb759b02c4"
EXPECTED_CHUNK_SCHEMA_VERSION = "1.0"
EXPECTED_CASE_COUNT = 32
EXPECTED_CHUNK_COUNT = 44
EXPECTED_DECISION_COUNT = 16
RESULT_SCHEMA_VERSION = "1.0.0"
EVALUATION_ROLE = "development_calibration"

ORACLE_RELATIVE_PATH = Path("evals/rag/answerability-calibration-v1.json")
CHUNKS_RELATIVE_PATH = Path("knowledge_base/processed/chunks.jsonl")
MANIFEST_RELATIVE_PATH = Path("knowledge_base/processed/manifest.json")

type Decision = Literal["answerable", "insufficient"]
type ExecutionMode = Literal["deterministic_fake", "real_provider_baseline"]
type OperationalStatus = Literal[
    "success", "provider_refusal", "provider_failure", "invalid_output", "evaluator_failure"
]


class CalibrationError(ValueError):
    """Base error for a closed calibration-harness failure."""


class CalibrationFixtureError(CalibrationError):
    """The oracle or canonical corpus failed validation."""


class CalibrationResultError(CalibrationError):
    """An evaluation result violated the privacy-safe result contract."""


@dataclass(frozen=True)
class CalibrationCase:
    """Minimum data needed to execute and score one calibration case."""

    case_id: str
    query: str
    expected_decision: Decision
    evidence_chunk_ids: tuple[str, ...]
    expected_support_chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class CanonicalEvidence:
    """Canonical evidence fields used by the isolated classifier boundary."""

    chunk_id: str
    text: str
    text_sha256: str


@dataclass(frozen=True)
class CalibrationFixture:
    """Validated fixture and canonical evidence bound to immutable digests."""

    fixture_id: str
    fixture_schema_version: str
    oracle_sha256: str
    corpus_sha256: str
    cases: tuple[CalibrationCase, ...]
    evidence_by_chunk_id: dict[str, CanonicalEvidence]


def _fail(message: str) -> Never:
    raise CalibrationFixtureError(message)


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        _fail(f"{context} must be an object with string keys")
    raw_mapping = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in raw_mapping):
        _fail(f"{context} must be an object with string keys")
    return cast(dict[str, object], raw_mapping)


def _closed_mapping(value: object, expected_keys: set[str], context: str) -> dict[str, object]:
    mapping = _mapping(value, context)
    if set(mapping) != expected_keys:
        _fail(f"{context} has an invalid closed shape")
    return mapping


def _array(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        _fail(f"{context} must be an array")
    return cast(list[object], value)


def _nonempty_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{context} must be a non-empty string")
    return value


def _string_array(value: object, context: str, *, allow_empty: bool) -> tuple[str, ...]:
    raw = _array(value, context)
    if not allow_empty and not raw:
        _fail(f"{context} must not be empty")
    if not all(isinstance(item, str) and item.strip() for item in raw):
        _fail(f"{context} must contain non-empty strings")
    result = tuple(cast(list[str], raw))
    if len(set(result)) != len(result):
        _fail(f"{context} must not contain duplicates")
    return result


def _integer(value: object, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        _fail(f"{context} must be an integer")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(path: Path, context: str) -> dict[str, object]:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CalibrationFixtureError(f"Unable to read valid {context} JSON") from exc
    return _mapping(raw, context)


def _parse_json_bytes(value: bytes, context: str) -> dict[str, object]:
    try:
        raw: object = json.loads(value.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CalibrationFixtureError(f"Unable to read valid {context} JSON") from exc
    return _mapping(raw, context)


def _validate_metadata(value: object) -> tuple[str, str]:
    metadata = _closed_mapping(
        value,
        {
            "fixture_id",
            "fixture_schema_version",
            "created_on",
            "language",
            "creation_purpose",
            "decision_boundary",
            "classifier_input_fields",
            "human_review_only_fields",
            "independence_declaration",
            "exposed_holdout_used",
            "development_calibration_data",
            "future_holdout",
        },
        "metadata",
    )
    fixture_id = _nonempty_string(metadata["fixture_id"], "metadata.fixture_id")
    schema_version = _nonempty_string(
        metadata["fixture_schema_version"], "metadata.fixture_schema_version"
    )
    if fixture_id != EXPECTED_FIXTURE_ID or schema_version != EXPECTED_FIXTURE_SCHEMA_VERSION:
        _fail("Fixture identity or schema version does not match the frozen contract")
    for field in ("created_on", "language", "creation_purpose", "independence_declaration"):
        _nonempty_string(metadata[field], f"metadata.{field}")
    boundary = _closed_mapping(
        metadata["decision_boundary"], {"answerable", "insufficient"}, "decision_boundary"
    )
    _nonempty_string(boundary["answerable"], "decision_boundary.answerable")
    _nonempty_string(boundary["insufficient"], "decision_boundary.insufficient")
    if metadata["classifier_input_fields"] != ["query", "evidence_chunk_ids"]:
        _fail("Classifier input fields do not match the frozen contract")
    if metadata["human_review_only_fields"] != [
        "expected_decision",
        "expected_support_chunk_ids",
        "rationale_for_human_review",
        "coverage_tags",
    ]:
        _fail("Human-review-only fields do not match the frozen contract")
    if (
        metadata["exposed_holdout_used"] is not False
        or metadata["development_calibration_data"] is not True
        or metadata["future_holdout"] is not False
    ):
        _fail("Fixture evaluation role flags do not match development calibration")
    return fixture_id, schema_version


def _validate_corpus_provenance(value: object) -> None:
    provenance = _closed_mapping(
        value,
        {
            "artifact_path",
            "sha256",
            "manifest_path",
            "manifest_declared_sha256",
            "hash_verified",
            "canonical_chunk_schema_version",
            "canonical_chunk_count",
        },
        "corpus_provenance",
    )
    expected_values: dict[str, object] = {
        "artifact_path": CHUNKS_RELATIVE_PATH.as_posix(),
        "sha256": EXPECTED_CORPUS_SHA256,
        "manifest_path": MANIFEST_RELATIVE_PATH.as_posix(),
        "manifest_declared_sha256": EXPECTED_CORPUS_SHA256,
        "hash_verified": True,
        "canonical_chunk_schema_version": EXPECTED_CHUNK_SCHEMA_VERSION,
        "canonical_chunk_count": EXPECTED_CHUNK_COUNT,
    }
    if provenance != expected_values:
        _fail("Oracle corpus provenance does not match the frozen contract")


def _parse_cases(value: object) -> tuple[tuple[CalibrationCase, ...], Counter[str], int, int]:
    raw_cases = _array(value, "cases")
    if len(raw_cases) != EXPECTED_CASE_COUNT:
        _fail("Fixture must contain exactly 32 cases")
    cases: list[CalibrationCase] = []
    categories: Counter[str] = Counter()
    collective_count = 0
    adversarial_count = 0
    for index, value_case in enumerate(raw_cases):
        context = f"cases[{index}]"
        item = _closed_mapping(
            value_case,
            {
                "case_id",
                "category",
                "query",
                "expected_decision",
                "evidence_chunk_ids",
                "expected_support_chunk_ids",
                "evidence_provenance",
                "requires_collective_multi_chunk_support",
                "coverage_tags",
                "rationale_for_human_review",
            },
            context,
        )
        case_id = _nonempty_string(item["case_id"], f"{context}.case_id")
        category = _nonempty_string(item["category"], f"{context}.category")
        query = _nonempty_string(item["query"], f"{context}.query")
        raw_decision = item["expected_decision"]
        if raw_decision not in {"answerable", "insufficient"}:
            _fail(f"{context}.expected_decision is invalid")
        expected_decision = cast(Decision, raw_decision)
        evidence_ids = _string_array(
            item["evidence_chunk_ids"], f"{context}.evidence_chunk_ids", allow_empty=False
        )
        support_ids = _string_array(
            item["expected_support_chunk_ids"],
            f"{context}.expected_support_chunk_ids",
            allow_empty=True,
        )
        if not set(support_ids).issubset(evidence_ids):
            _fail(f"{context}.expected_support_chunk_ids must reference supplied evidence")
        if (expected_decision == "answerable") != bool(support_ids):
            _fail(f"{context} has support IDs inconsistent with its expected decision")
        raw_provenance = _array(item["evidence_provenance"], f"{context}.evidence_provenance")
        provenance_ids: list[str] = []
        for provenance_index, raw_entry in enumerate(raw_provenance):
            entry = _closed_mapping(
                raw_entry,
                {"chunk_id", "source_text_sha256"},
                f"{context}.evidence_provenance[{provenance_index}]",
            )
            provenance_ids.append(
                _nonempty_string(entry["chunk_id"], f"{context}.evidence_provenance.chunk_id")
            )
            digest = _nonempty_string(
                entry["source_text_sha256"],
                f"{context}.evidence_provenance.source_text_sha256",
            )
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                _fail(f"{context}.evidence_provenance has an invalid SHA-256")
        if tuple(provenance_ids) != evidence_ids:
            _fail(f"{context}.evidence_provenance must exactly match evidence order")
        collective = item["requires_collective_multi_chunk_support"]
        if not isinstance(collective, bool):
            _fail(f"{context}.requires_collective_multi_chunk_support must be boolean")
        tags = _string_array(item["coverage_tags"], f"{context}.coverage_tags", allow_empty=False)
        _nonempty_string(
            item["rationale_for_human_review"], f"{context}.rationale_for_human_review"
        )
        categories[category] += 1
        collective_count += int(collective)
        adversarial_count += int("adversarial_evidence" in tags)
        cases.append(CalibrationCase(case_id, query, expected_decision, evidence_ids, support_ids))
    if len({case.case_id for case in cases}) != len(cases):
        _fail("Fixture case IDs must be unique")
    decisions = Counter(case.expected_decision for case in cases)
    if decisions != Counter(
        answerable=EXPECTED_DECISION_COUNT, insufficient=EXPECTED_DECISION_COUNT
    ):
        _fail("Fixture must contain the frozen 16/16 decision balance")
    return tuple(cases), categories, collective_count, adversarial_count


def _validate_aggregate_counts(
    value: object, categories: Counter[str], collective_count: int, adversarial_count: int
) -> None:
    aggregate = _closed_mapping(
        value,
        {
            "total_cases",
            "expected_decisions",
            "requires_collective_multi_chunk_support",
            "adversarial_evidence_cases",
            "category_counts",
        },
        "aggregate_counts",
    )
    decisions = _closed_mapping(
        aggregate["expected_decisions"], {"answerable", "insufficient"}, "expected_decisions"
    )
    category_counts = _mapping(aggregate["category_counts"], "category_counts")
    parsed_categories = {
        key: _integer(count, f"category_counts.{key}") for key, count in category_counts.items()
    }
    if (
        aggregate["total_cases"] != EXPECTED_CASE_COUNT
        or decisions
        != {
            "answerable": EXPECTED_DECISION_COUNT,
            "insufficient": EXPECTED_DECISION_COUNT,
        }
        or aggregate["requires_collective_multi_chunk_support"] != collective_count
        or aggregate["adversarial_evidence_cases"] != adversarial_count
        or parsed_categories != dict(categories)
    ):
        _fail("Aggregate counts do not match the fixture cases")


def _load_canonical_evidence(path: Path) -> dict[str, CanonicalEvidence]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CalibrationFixtureError("Unable to read canonical chunks") from exc
    if _sha256_bytes(raw) != EXPECTED_CORPUS_SHA256:
        _fail("Canonical corpus SHA-256 does not match the frozen digest")
    if raw.startswith(b"\xef\xbb\xbf") or not raw.endswith(b"\n"):
        _fail("Canonical chunks must be UTF-8 without BOM and end with LF")
    try:
        lines = raw.decode("utf-8").splitlines()
        parsed: list[object] = [json.loads(line) for line in lines]
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CalibrationFixtureError("Canonical chunks are malformed") from exc
    if len(parsed) != EXPECTED_CHUNK_COUNT or any(not line for line in lines):
        _fail("Canonical corpus must contain exactly 44 non-empty chunks")
    result: dict[str, CanonicalEvidence] = {}
    for index, raw_chunk in enumerate(parsed):
        chunk = _mapping(raw_chunk, f"chunks[{index}]")
        for field in ("chunk_id", "text", "text_sha256", "schema_version"):
            if field not in chunk:
                _fail(f"chunks[{index}] is missing {field}")
        chunk_id = _nonempty_string(chunk["chunk_id"], f"chunks[{index}].chunk_id")
        text = _nonempty_string(chunk["text"], f"chunks[{index}].text")
        text_sha256 = _nonempty_string(chunk["text_sha256"], f"chunks[{index}].text_sha256")
        if chunk["schema_version"] != EXPECTED_CHUNK_SCHEMA_VERSION:
            _fail(f"chunks[{index}] has an invalid schema version")
        if _sha256_bytes(text.encode("utf-8")) != text_sha256:
            _fail(f"chunks[{index}] has a mismatched text SHA-256")
        if chunk_id in result:
            _fail("Canonical corpus contains a duplicate chunk ID")
        result[chunk_id] = CanonicalEvidence(chunk_id, text, text_sha256)
    return result


def _validate_manifest(path: Path) -> None:
    manifest = _read_json(path, "processed manifest")
    if manifest.get("chunks_sha256") != EXPECTED_CORPUS_SHA256:
        _fail("Processed manifest does not declare the frozen corpus SHA-256")
    if manifest.get("schema_version") != EXPECTED_CHUNK_SCHEMA_VERSION:
        _fail("Processed manifest schema version is invalid")
    if manifest.get("total_chunks") != EXPECTED_CHUNK_COUNT:
        _fail("Processed manifest chunk count is invalid")


def _validate_case_evidence(
    raw_cases: object,
    cases: tuple[CalibrationCase, ...],
    evidence_by_chunk_id: dict[str, CanonicalEvidence],
) -> None:
    raw_case_list = _array(raw_cases, "cases")
    for case_index, case in enumerate(cases):
        raw_case = _mapping(raw_case_list[case_index], f"cases[{case_index}]")
        raw_provenance = _array(raw_case["evidence_provenance"], "evidence_provenance")
        for evidence_index, chunk_id in enumerate(case.evidence_chunk_ids):
            evidence = evidence_by_chunk_id.get(chunk_id)
            if evidence is None:
                _fail(f"Case {case.case_id} references an unknown evidence chunk")
            provenance = _mapping(raw_provenance[evidence_index], "evidence_provenance entry")
            if provenance["source_text_sha256"] != evidence.text_sha256:
                _fail(f"Case {case.case_id} has mismatched evidence provenance")


def load_calibration_fixture(repository_root: Path) -> CalibrationFixture:
    """Load and fail-closed validate the frozen oracle and canonical corpus."""
    oracle_path = repository_root / ORACLE_RELATIVE_PATH
    chunks_path = repository_root / CHUNKS_RELATIVE_PATH
    manifest_path = repository_root / MANIFEST_RELATIVE_PATH
    try:
        raw_oracle_bytes = oracle_path.read_bytes()
    except OSError as exc:
        raise CalibrationFixtureError("Unable to read answerability calibration oracle") from exc
    oracle_sha256 = _sha256_bytes(raw_oracle_bytes)
    if oracle_sha256 != EXPECTED_ORACLE_SHA256:
        _fail("Oracle artifact SHA-256 does not match the frozen digest")
    oracle = _closed_mapping(
        _parse_json_bytes(raw_oracle_bytes, "answerability calibration oracle"),
        {"metadata", "corpus_provenance", "aggregate_counts", "cases"},
        "oracle",
    )
    fixture_id, schema_version = _validate_metadata(oracle["metadata"])
    _validate_corpus_provenance(oracle["corpus_provenance"])
    cases, categories, collective_count, adversarial_count = _parse_cases(oracle["cases"])
    _validate_aggregate_counts(
        oracle["aggregate_counts"], categories, collective_count, adversarial_count
    )
    evidence_by_chunk_id = _load_canonical_evidence(chunks_path)
    _validate_manifest(manifest_path)
    _validate_case_evidence(oracle["cases"], cases, evidence_by_chunk_id)
    return CalibrationFixture(
        fixture_id,
        schema_version,
        oracle_sha256,
        EXPECTED_CORPUS_SHA256,
        cases,
        evidence_by_chunk_id,
    )


def reconstruct_request(fixture: CalibrationFixture, case: CalibrationCase) -> AnswerabilityRequest:
    """Build model-safe evidence without reading or projecting human labels."""
    blocks: list[EvidenceBlock] = []
    for evidence_id, chunk_id in enumerate(case.evidence_chunk_ids, 1):
        canonical = fixture.evidence_by_chunk_id.get(chunk_id)
        if canonical is None:
            raise CalibrationFixtureError("Validated fixture lost canonical evidence")
        blocks.append(
            EvidenceBlock(
                evidence_id=evidence_id,
                chunk_id="",
                document_id="",
                title="",
                source_filename="",
                locator="",
                citation_label="",
                score=0.0,
                text=canonical.text,
                text_sha256="",
            )
        )
    return AnswerabilityRequest(case.query, tuple(blocks), 1)


def map_expected_support_evidence_ids(case: CalibrationCase) -> tuple[int, ...]:
    """Map human support labels for scoring only, after provider execution."""
    chunk_id_to_evidence_id = {
        chunk_id: evidence_id for evidence_id, chunk_id in enumerate(case.evidence_chunk_ids, 1)
    }
    return tuple(chunk_id_to_evidence_id[chunk_id] for chunk_id in case.expected_support_chunk_ids)


def _valid_nonnegative(value: int | None) -> bool:
    return value is None or (not isinstance(value, bool) and value >= 0)


def _usage_payload(usage: GenerationUsage | None) -> dict[str, object] | None:
    if usage is None:
        return None
    numeric = (
        usage.input_tokens,
        usage.cached_input_tokens,
        usage.output_tokens,
        usage.total_tokens,
        usage.physical_attempts,
    )
    if not all(_valid_nonnegative(value) for value in numeric):
        raise CalibrationResultError("Provider usage contains invalid numeric metadata")
    if usage.logical_api_calls < 0 or usage.application_attempt < 1:
        raise CalibrationResultError("Provider usage contains invalid attempt metadata")
    return {
        "input_tokens": usage.input_tokens,
        "cached_input_tokens": usage.cached_input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "logical_api_calls": usage.logical_api_calls,
        "physical_attempts": usage.physical_attempts,
        "refusal_detected": usage.refusal_detected,
        "application_attempt": usage.application_attempt,
        "transport_attempts_observable": usage.transport_attempts_observable,
    }


def _validate_decision(
    decision: object, request: AnswerabilityRequest
) -> tuple[Decision, tuple[int, ...], dict[str, object] | None]:
    if not isinstance(decision, AnswerabilityDecision):
        raise CalibrationResultError("Provider returned an invalid decision object")
    if decision.decision not in {"answerable", "insufficient"}:
        raise CalibrationResultError("Provider returned an invalid semantic decision")
    available = {block.evidence_id: block.text for block in request.evidence_blocks}
    observed_ids: list[int] = []
    for support in decision.supporting_evidence:
        if support.evidence_id not in available or support.evidence_id in observed_ids:
            raise CalibrationResultError("Provider returned invalid supporting evidence IDs")
        if not support.quote.strip():
            raise CalibrationResultError("Provider returned an invalid supporting quote")
        if support.quote not in available[support.evidence_id]:
            raise CalibrationResultError("Provider returned a non-literal supporting quote")
        observed_ids.append(support.evidence_id)
    if (decision.decision == "answerable") != bool(observed_ids):
        raise CalibrationResultError("Provider decision and support are inconsistent")
    return decision.decision, tuple(observed_ids), _usage_payload(decision.usage)


def _failure_case(
    case: CalibrationCase,
    expected_support_ids: tuple[int, ...],
    status: OperationalStatus,
    usage: GenerationUsage | None,
) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "expected_decision": case.expected_decision,
        "observed_decision": None,
        "operational_status": status,
        "passed": False,
        "expected_support_evidence_ids": list(expected_support_ids),
        "observed_supporting_evidence_ids": [],
        "support_id_agreement": None,
        "usage": _usage_payload(usage),
    }


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _metrics(case_results: list[dict[str, object]]) -> dict[str, object]:
    total = len(case_results)
    passed = sum(result["passed"] is True for result in case_results)
    answerable = [result for result in case_results if result["expected_decision"] == "answerable"]
    insufficient = [
        result for result in case_results if result["expected_decision"] == "insufficient"
    ]
    answerable_correct = sum(result["passed"] is True for result in answerable)
    insufficient_correct = sum(result["passed"] is True for result in insufficient)
    false_abstentions = sum(
        result["operational_status"] == "success" and result["observed_decision"] == "insufficient"
        for result in answerable
    )
    false_answers = sum(
        result["operational_status"] == "success" and result["observed_decision"] == "answerable"
        for result in insufficient
    )
    support_eligible = [
        result
        for result in answerable
        if result["operational_status"] == "success"
        and bool(cast(list[object], result["expected_support_evidence_ids"]))
    ]
    support_agreements = sum(result["support_id_agreement"] is True for result in support_eligible)
    status_counts = Counter(cast(str, result["operational_status"]) for result in case_results)
    return {
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": total - passed,
        "overall_decision_accuracy": _ratio(passed, total),
        "answerable_total": len(answerable),
        "answerable_correct": answerable_correct,
        "answerable_recall": _ratio(answerable_correct, len(answerable)),
        "false_abstention_count": false_abstentions,
        "false_abstention_rate": _ratio(false_abstentions, len(answerable)),
        "insufficient_total": len(insufficient),
        "insufficient_correct": insufficient_correct,
        "insufficient_recall": _ratio(insufficient_correct, len(insufficient)),
        "false_answer_count": false_answers,
        "false_answer_rate": _ratio(false_answers, len(insufficient)),
        "provider_refusal_count": status_counts["provider_refusal"],
        "provider_failure_count": status_counts["provider_failure"],
        "invalid_output_count": status_counts["invalid_output"],
        "evaluator_failure_count": status_counts["evaluator_failure"],
        "support_id_agreement_evaluated": len(support_eligible),
        "support_id_agreement_count": support_agreements,
        "support_id_agreement_rate": _ratio(support_agreements, len(support_eligible)),
    }


def calibration_result_schema() -> dict[str, object]:
    """Load the closed, versioned calibration-result JSON Schema."""
    path = (
        Path(__file__).resolve().parents[3]
        / "evals"
        / "rag"
        / "answerability-calibration-result-v1.schema.json"
    )
    try:
        return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CalibrationResultError("Unable to load calibration result schema") from exc


def validate_calibration_result(result: dict[str, object]) -> None:
    """Validate a privacy-safe result against its closed artifact contract."""
    validator = Draft202012Validator(cast(Any, calibration_result_schema()))
    errors = sorted(
        validator.iter_errors(cast(Any, result)),  # pyright: ignore[reportUnknownMemberType]
        key=lambda error: list(error.path),
    )
    if errors:
        raise CalibrationResultError("Calibration result violates the closed schema")
    case_results = cast(list[dict[str, object]], result["cases"])
    if len(case_results) != EXPECTED_CASE_COUNT:
        raise CalibrationResultError("Calibration result is internally inconsistent")
    case_ids = [cast(str, case["case_id"]) for case in case_results]
    if len(set(case_ids)) != len(case_ids):
        raise CalibrationResultError("Calibration result is internally inconsistent")
    for case in case_results:
        status = cast(str, case["operational_status"])
        observed_decision = case["observed_decision"]
        expected_decision = case["expected_decision"]
        observed_support = cast(list[int], case["observed_supporting_evidence_ids"])
        expected_support = cast(list[int], case["expected_support_evidence_ids"])
        if (expected_decision == "answerable") is not bool(expected_support):
            raise CalibrationResultError("Calibration result is internally inconsistent")
        if status == "success":
            if observed_decision not in {"answerable", "insufficient"}:
                raise CalibrationResultError("Calibration result is internally inconsistent")
            if case["passed"] is not (observed_decision == expected_decision):
                raise CalibrationResultError("Calibration result is internally inconsistent")
            if (observed_decision == "answerable") is not bool(observed_support):
                raise CalibrationResultError("Calibration result is internally inconsistent")
            if case["support_id_agreement"] is not (observed_support == expected_support):
                raise CalibrationResultError("Calibration result is internally inconsistent")
        elif (
            observed_decision is not None
            or case["passed"] is not False
            or observed_support
            or case["support_id_agreement"] is not None
        ):
            raise CalibrationResultError("Calibration result is internally inconsistent")
    if result["metrics"] != _metrics(case_results):
        raise CalibrationResultError("Calibration result is internally inconsistent")


def _validate_execution_mode_provider(execution_mode: str, provider_name: str) -> None:
    if execution_mode not in {"deterministic_fake", "real_provider_baseline"}:
        raise CalibrationResultError("Execution mode is invalid")
    if execution_mode == "real_provider_baseline" and provider_name != "openai":
        raise CalibrationResultError("Real provider baseline requires the openai provider")
    if execution_mode == "deterministic_fake" and provider_name == "openai":
        raise CalibrationResultError("Deterministic fake mode rejects the openai provider")


def evaluate_calibration(
    fixture: CalibrationFixture,
    provider: AnswerabilityProvider,
    *,
    execution_mode: ExecutionMode,
) -> dict[str, object]:
    """Execute injected provider decisions and score only after every call returns."""
    provider_name = provider.provider_name
    model_identifier = provider.model_identifier
    if not provider_name.strip() or not model_identifier.strip():
        raise CalibrationResultError("Provider identity must be explicit")
    _validate_execution_mode_provider(execution_mode, provider_name)
    case_results: list[dict[str, object]] = []
    for case in fixture.cases:
        request = reconstruct_request(fixture, case)
        try:
            raw_decision = provider.assess(request)
        except AnswerabilityProviderRefusalError as exc:
            expected_support_ids = map_expected_support_evidence_ids(case)
            case_results.append(
                _failure_case(case, expected_support_ids, "provider_refusal", exc.usage)
            )
            continue
        except AnswerabilityProviderError as exc:
            expected_support_ids = map_expected_support_evidence_ids(case)
            case_results.append(
                _failure_case(case, expected_support_ids, "provider_failure", exc.usage)
            )
            continue
        except Exception:
            expected_support_ids = map_expected_support_evidence_ids(case)
            case_results.append(
                _failure_case(case, expected_support_ids, "evaluator_failure", None)
            )
            continue
        expected_support_ids = map_expected_support_evidence_ids(case)
        try:
            observed_decision, observed_support_ids, usage = _validate_decision(
                raw_decision, request
            )
        except CalibrationResultError:
            case_results.append(_failure_case(case, expected_support_ids, "invalid_output", None))
            continue
        case_results.append(
            {
                "case_id": case.case_id,
                "expected_decision": case.expected_decision,
                "observed_decision": observed_decision,
                "operational_status": "success",
                "passed": observed_decision == case.expected_decision,
                "expected_support_evidence_ids": list(expected_support_ids),
                "observed_supporting_evidence_ids": list(observed_support_ids),
                "support_id_agreement": observed_support_ids == expected_support_ids,
                "usage": usage,
            }
        )
    result: dict[str, object] = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "evaluation_role": EVALUATION_ROLE,
        "unseen_holdout": False,
        "fixture": {
            "fixture_id": fixture.fixture_id,
            "fixture_schema_version": fixture.fixture_schema_version,
            "oracle_artifact_sha256": fixture.oracle_sha256,
        },
        "corpus": {"canonical_corpus_sha256": fixture.corpus_sha256},
        "classifier": {
            "prompt_version": ANSWERABILITY_PROMPT_VERSION,
            "prompt_sha256": answerability_prompt_sha256(),
        },
        "provider": {"provider": provider_name, "model": model_identifier},
        "execution_mode": execution_mode,
        "case_count": len(fixture.cases),
        "cases": case_results,
        "metrics": _metrics(case_results),
    }
    validate_calibration_result(result)
    return result


def serialize_calibration_result(result: dict[str, object]) -> str:
    """Return deterministic JSON containing only the validated safe result fields."""
    validate_calibration_result(result)
    return json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_calibration_result(result: dict[str, object], output_path: Path) -> None:
    """Create, without overwriting, one validated machine-readable result artifact."""
    payload = serialize_calibration_result(result)
    try:
        with output_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
    except OSError as exc:
        raise CalibrationResultError("Unable to create calibration result artifact") from exc
