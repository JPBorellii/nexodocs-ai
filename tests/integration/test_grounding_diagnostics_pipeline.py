"""Offline grounding diagnostics with fake providers and in-memory Qdrant."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from nexodocs_ai.knowledge_base.generator import repository_root
from nexodocs_ai.observability.reports import answer_report, privacy_safe_report, write_report
from nexodocs_ai.rag.answer_provider import DeterministicFakeAnswerProvider
from nexodocs_ai.rag.config import RagConfig, load_rag_config
from nexodocs_ai.rag.grounding_diagnostics import GroundingErrorCode
from nexodocs_ai.rag.models import (
    AnswerGenerationRequest,
    GeneratedAnswer,
    GeneratedCitationReference,
    GenerationUsage,
    ProviderError,
    ProviderRefusalError,
    RagRequest,
)
from nexodocs_ai.rag.pipeline import RagPipeline
from nexodocs_ai.rag.serialization import (
    cli_exit_code,
    cli_response_data,
    privacy_safe_report_required,
)
from nexodocs_ai.retrieval.config import load_config
from nexodocs_ai.retrieval.embeddings import DeterministicFakeEmbeddingProvider
from nexodocs_ai.retrieval.indexer import incremental_index_with_usage
from nexodocs_ai.retrieval.qdrant_store import QdrantStore, create_client
from nexodocs_ai.retrieval.retriever import Retriever

AnswerFactory = Callable[[AnswerGenerationRequest], GeneratedAnswer]


def _usage() -> GenerationUsage:
    return GenerationUsage(
        "diagnostic-fake",
        "diagnostic-fake-v1",
        20,
        3,
        7,
        27,
        "req_fake_grounding",
        1,
        1,
        False,
        1,
        True,
    )


class _ScriptedProvider:
    provider_name = "diagnostic-fake"
    model_identifier = "diagnostic-fake-v1"

    def __init__(self, factory: AnswerFactory) -> None:
        self.factory = factory
        self.calls = 0

    def generate(self, request: AnswerGenerationRequest) -> GeneratedAnswer:
        self.calls += 1
        return self.factory(request)


class _FailingProvider:
    provider_name = "diagnostic-fake"
    model_identifier = "diagnostic-fake-v1"

    def __init__(self, refusal: bool) -> None:
        self.refusal = refusal
        self.calls = 0

    def generate(self, request: AnswerGenerationRequest) -> GeneratedAnswer:
        self.calls += 1
        error = ProviderRefusalError if self.refusal else ProviderError
        raise error("raw provider message req_forbidden", _usage())


@pytest.fixture(scope="module")
def retriever() -> Retriever:
    config = load_config(
        {
            "APP_ENV": "test",
            "EMBEDDING_PROVIDER": "fake",
            "OPENAI_EMBEDDING_DIMENSIONS": "192",
            "QDRANT_MODE": "memory",
        }
    )
    provider = DeterministicFakeEmbeddingProvider(config.embedding_dimensions)
    store = QdrantStore(create_client(config), config.collection_name, provider.dimensions)
    indexed = incremental_index_with_usage(repository_root(), config, provider, store)
    assert indexed.result.inserted_or_updated == 44
    return Retriever(provider, store)


def _rag_config() -> RagConfig:
    return load_rag_config(
        {"APP_ENV": "test", "ANSWER_PROVIDER": "fake", "RAG_MIN_CONTEXT_CHARACTERS": "1"}
    )


def _first(request: AnswerGenerationRequest) -> tuple[int, str]:
    block = request.evidence_blocks[0]
    return block.evidence_id, block.text[:40]


def _schema_invalid(_: AnswerGenerationRequest) -> GeneratedAnswer:
    return GeneratedAnswer("", (), _usage())


def _answer_too_long(_: AnswerGenerationRequest) -> GeneratedAnswer:
    return GeneratedAnswer("x" * 4_001, (), _usage())


def _unsupported(request: AnswerGenerationRequest) -> GeneratedAnswer:
    identifier, quote = _first(request)
    return GeneratedAnswer(
        f"https://unit.invalid [{identifier}]",
        (GeneratedCitationReference(identifier, quote),),
        _usage(),
    )


def _duplicate(request: AnswerGenerationRequest) -> GeneratedAnswer:
    identifier, quote = _first(request)
    reference = GeneratedCitationReference(identifier, quote)
    return GeneratedAnswer(f"Resposta [{identifier}]", (reference, reference), _usage())


def _unknown(_: AnswerGenerationRequest) -> GeneratedAnswer:
    return GeneratedAnswer(
        "Resposta [999]", (GeneratedCitationReference(999, "ausente"),), _usage()
    )


def _quote_mismatch(request: AnswerGenerationRequest) -> GeneratedAnswer:
    identifier, _ = _first(request)
    return GeneratedAnswer(
        f"Resposta [{identifier}]", (GeneratedCitationReference(identifier, ""),), _usage()
    )


def _quote_not_in_evidence(request: AnswerGenerationRequest) -> GeneratedAnswer:
    identifier, _ = _first(request)
    return GeneratedAnswer(
        f"Resposta [{identifier}]",
        (GeneratedCitationReference(identifier, "raw quote not in evidence"),),
        _usage(),
    )


def _missing(_: AnswerGenerationRequest) -> GeneratedAnswer:
    return GeneratedAnswer("Raw generated answer without marker", (), _usage())


def _marker_mismatch(request: AnswerGenerationRequest) -> GeneratedAnswer:
    identifier, quote = _first(request)
    return GeneratedAnswer(
        "Resposta [999]", (GeneratedCitationReference(identifier, quote),), _usage()
    )


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (_schema_invalid, GroundingErrorCode.SCHEMA_INVALID.value),
        (_answer_too_long, GroundingErrorCode.ANSWER_TOO_LONG.value),
        (_unsupported, GroundingErrorCode.UNSUPPORTED_CLAIM.value),
        (_duplicate, GroundingErrorCode.DUPLICATE_CITATION.value),
        (_unknown, GroundingErrorCode.UNKNOWN_CITATION.value),
        (_quote_mismatch, GroundingErrorCode.QUOTE_MISMATCH.value),
        (_quote_not_in_evidence, GroundingErrorCode.QUOTE_NOT_IN_EVIDENCE.value),
        (_missing, GroundingErrorCode.MISSING_CITATION.value),
        (_marker_mismatch, GroundingErrorCode.MARKER_CITATION_MISMATCH.value),
    ],
)
def test_each_real_grounding_failure_is_sanitized_and_single_attempt(
    retriever: Retriever,
    tmp_path: Path,
    factory: AnswerFactory,
    expected: str,
) -> None:
    provider = _ScriptedProvider(factory)
    pipeline = RagPipeline(retriever, provider, _rag_config())
    request = RagRequest("Como funciona o cancelamento fictício?")
    run = pipeline.answer_with_usage(request)

    assert run.response.status == "grounding_failed"
    assert run.response.reason_code == expected
    assert run.response.answer is None
    assert run.response.citations == ()
    assert provider.calls == 1
    assert run.application_attempts == 1
    assert run.answer_usage == _usage()
    assert cli_exit_code(run.response) == 1
    assert cli_response_data(run.response) == {
        "safe_error_code": expected,
        "status": "grounding_failed",
    }
    assert privacy_safe_report_required(run.response) is True

    report = privacy_safe_report(
        answer_report(
            run,
            provider.provider_name,
            provider.model_identifier,
            "nexodocs_chunks_v1",
            len(request.query),
            1,
        )
    )
    path = write_report(tmp_path, f"data/run-reports/{expected}.json", report)
    persisted = json.loads(path.read_text(encoding="utf-8"))
    serialized = json.dumps(persisted, ensure_ascii=False)
    assert persisted["status"] == "grounding_failed"
    assert persisted["safe_error_code"] == expected
    assert persisted["logical_api_calls"] == run.retrieval_usage.logical_api_calls + 1
    assert persisted["answer_usage"]["total_tokens"] == 27
    assert persisted["total_tokens"] == (run.retrieval_usage.total_tokens or 0) + 27
    for prohibited in (
        request.query,
        "Raw generated answer",
        "raw quote",
        "Grounding validation failed",
        "traceback",
        "req_fake_grounding",
    ):
        assert prohibited not in serialized


def test_success_fallback_and_provider_failures_remain_unchanged(retriever: Retriever) -> None:
    config = _rag_config()
    answered_provider = DeterministicFakeAnswerProvider(config)
    answered = RagPipeline(retriever, answered_provider, config).answer_with_usage(
        RagRequest("Como funciona o cancelamento?")
    )
    assert answered.response.status == "answered"
    assert answered.response.reason_code is None
    assert privacy_safe_report_required(answered.response) is False

    no_evidence = RagPipeline(retriever, answered_provider, config).answer_with_usage(
        RagRequest("astronomia galáctica", score_threshold=1.0)
    )
    assert no_evidence.response.status == "no_evidence"
    assert no_evidence.answer_usage is None
    assert no_evidence.application_attempts == 0

    generation_provider = _FailingProvider(False)
    generation = RagPipeline(retriever, generation_provider, config).answer_with_usage(
        RagRequest("Como funciona o cancelamento?")
    )
    assert generation.response.status == "generation_failed"
    assert generation.response.reason_code == "provider_invalid_output"
    assert generation_provider.calls == 1

    refusal_provider = _FailingProvider(True)
    refusal = RagPipeline(retriever, refusal_provider, config).answer_with_usage(
        RagRequest("Como funciona o cancelamento?")
    )
    assert refusal.response.status == "generation_failed"
    assert refusal.response.reason_code == "provider_refusal"
    assert refusal_provider.calls == 1
