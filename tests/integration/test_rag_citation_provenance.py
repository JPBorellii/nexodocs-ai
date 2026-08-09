"""Offline integration coverage for internal-to-public citation provenance."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import cast

import pytest

from nexodocs_ai.rag import pipeline as pipeline_module
from nexodocs_ai.rag.config import load_rag_config
from nexodocs_ai.rag.models import (
    AnswerGenerationRequest,
    Citation,
    EvidenceBlock,
    GeneratedAnswer,
    GeneratedCitationReference,
    RagRequest,
    RagResponse,
)
from nexodocs_ai.retrieval.models import (
    EmbeddingRunUsage,
    RetrievalOperationResult,
    RetrievalResponse,
    RetrievalResult,
)
from nexodocs_ai.retrieval.retriever import Retriever


class _EmbeddingIdentity:
    provider_name = "offline-provenance"
    model_identifier = "offline-provenance-v1"
    dimensions = 8
    batch_size = 1


class _Retriever:
    provider = _EmbeddingIdentity()
    max_per_document = 3

    def __init__(self, result_count: int) -> None:
        self.results = tuple(
            _retrieval_result(identifier) for identifier in range(1, result_count + 1)
        )

    def retrieve_with_usage(
        self,
        query: str,
        top_k: int | None = None,
        filters: object | None = None,
        score_threshold: float | None = None,
    ) -> RetrievalOperationResult:
        del top_k, filters, score_threshold
        return RetrievalOperationResult(
            RetrievalResponse("found", query, self.results, {}),
            EmbeddingRunUsage(
                "offline-provenance",
                "offline-provenance-v1",
                8,
                1,
                1,
                1,
                0,
                0,
                None,
                None,
                True,
                (),
            ),
        )


class _ScriptedProvider:
    provider_name = "offline-provenance"
    model_identifier = "offline-provenance-v1"

    def __init__(self, markers: Sequence[int]) -> None:
        self.markers = tuple(markers)

    def generate(self, request: AnswerGenerationRequest) -> GeneratedAnswer:
        by_id = {block.evidence_id: block for block in request.evidence_blocks}
        unique_ids = tuple(dict.fromkeys(self.markers))
        return GeneratedAnswer(
            " ".join(
                f"Claim {index} [{identifier}]" for index, identifier in enumerate(self.markers)
            ),
            tuple(
                GeneratedCitationReference(identifier, by_id[identifier].text)
                for identifier in unique_ids
            ),
        )


def _retrieval_result(identifier: int) -> RetrievalResult:
    text = f"Fictitious text unique to internal evidence {identifier}."
    return RetrievalResult(
        f"chunk-{identifier}",
        0.9 - identifier / 100,
        text,
        f"source-{identifier}.pdf",
        f"page:{identifier}",
        f"Fictitious source {identifier} - page {identifier}",
        {
            "document_id": f"DOC-{identifier}",
            "title": f"Fictitious source {identifier}",
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        },
    )


def _answer(markers: Sequence[int], result_count: int) -> RagResponse:
    retriever = _Retriever(result_count)
    pipeline = pipeline_module.RagPipeline(
        cast(Retriever, retriever),
        _ScriptedProvider(markers),
        load_rag_config(
            {
                "APP_ENV": "test",
                "ANSWER_PROVIDER": "fake",
                "RAG_MIN_CONTEXT_CHARACTERS": "1",
            }
        ),
    )
    return pipeline.answer(RagRequest("Fictitious policy question"))


def _unproven_validator(
    answer: GeneratedAnswer,
    evidence: tuple[EvidenceBlock, ...],
    maximum: int,
) -> tuple[str, tuple[Citation, ...]]:
    del answer, evidence, maximum
    return (
        "Claim 0 [1]",
        (
            Citation(
                1,
                "missing-chunk",
                "MISSING-DOC",
                "Missing source",
                "missing.pdf",
                "page:999",
                "Missing source - page 999",
                "Missing text.",
                0.1,
            ),
        ),
    )


def _assert_sources(response: RagResponse, expected_internal_ids: Sequence[int]) -> None:
    assert response.status == "answered"
    assert len(response.citations) == len(response.evidence) == len(expected_internal_ids)
    for public_id, internal_id in enumerate(expected_internal_ids, start=1):
        citation = response.citations[public_id - 1]
        summary = response.evidence[public_id - 1]
        source = _retrieval_result(internal_id)
        assert citation.citation_id == summary.citation_id == public_id
        assert citation.chunk_id == summary.chunk_id == source.chunk_id
        assert citation.document_id == summary.document_id == source.metadata["document_id"]
        assert citation.citation_label == summary.citation_label == source.citation_label
        assert citation.score == summary.score == source.score
        assert summary.text_sha256 == source.metadata["text_sha256"]


def test_out_of_order_single_citation_preserves_source_provenance() -> None:
    response = _answer((2,), 2)

    assert response.answer == "Claim 0 [1]"
    _assert_sources(response, (2,))


def test_out_of_order_multiple_citations_preserve_source_provenance() -> None:
    response = _answer((3, 1), 3)

    assert response.answer == "Claim 0 [1] Claim 1 [2]"
    _assert_sources(response, (3, 1))


def test_repeated_out_of_order_markers_are_stable_without_duplicate_evidence() -> None:
    response = _answer((2, 1, 2), 2)

    assert response.answer == "Claim 0 [1] Claim 1 [2] Claim 2 [1]"
    _assert_sources(response, (2, 1))


def test_in_order_citations_remain_unchanged() -> None:
    response = _answer((1, 2), 2)

    assert response.answer == "Claim 0 [1] Claim 1 [2]"
    _assert_sources(response, (1, 2))


def test_evidence_summary_provenance_lookup_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline_module, "validate_generated", _unproven_validator)

    response = _answer((1,), 1)

    assert response.status == "grounding_failed"
    assert response.reason_code == "grounding_validation_failed"
    assert response.citations == ()
    assert response.evidence == ()
