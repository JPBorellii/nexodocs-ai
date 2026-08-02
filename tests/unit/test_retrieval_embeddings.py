"""Tests for offline and injected embedding providers."""

from __future__ import annotations

import math
from typing import Literal

import pytest
from openai.types import CreateEmbeddingResponse
from openai.types.create_embedding_response import Usage
from openai.types.embedding import Embedding

from nexodocs_ai.retrieval.embeddings import (
    DeterministicFakeEmbeddingProvider,
    OpenAIEmbeddingProvider,
    validate_vectors,
)
from nexodocs_ai.retrieval.models import RetrievalConfig, RetrievalError


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _response(
    indexed_vectors: list[tuple[int, list[float]]],
    total_tokens: int,
    request_id: str | None = None,
) -> CreateEmbeddingResponse:
    response = CreateEmbeddingResponse(
        data=[
            Embedding(index=index, embedding=vector, object="embedding")
            for index, vector in indexed_vectors
        ],
        model="fake-openai-model",
        object="list",
        usage=Usage(prompt_tokens=total_tokens, total_tokens=total_tokens),
    )
    if request_id is not None:
        object.__setattr__(response, "_request_id", request_id)
    return response


class FakeEmbeddingsEndpoint:
    def __init__(self, responses: list[CreateEmbeddingResponse]) -> None:
        self._responses = responses
        self.calls: list[tuple[str | list[str], str, int, Literal["float"]]] = []

    def create(
        self,
        *,
        input: str | list[str],
        model: str,
        dimensions: int,
        encoding_format: Literal["float"],
    ) -> CreateEmbeddingResponse:
        self.calls.append((input, model, dimensions, encoding_format))
        return self._responses.pop(0)


class FakeEmbeddingsClient:
    def __init__(self, responses: list[CreateEmbeddingResponse]) -> None:
        self._embeddings = FakeEmbeddingsEndpoint(responses)

    @property
    def embeddings(self) -> FakeEmbeddingsEndpoint:
        return self._embeddings


def test_fake_embedder_is_deterministic_normalized_and_lexical() -> None:
    provider = DeterministicFakeEmbeddingProvider(64)

    reference = provider.embed_query("cancelamento de consulta pelo paciente")
    repeated = provider.embed_query("cancelamento de consulta pelo paciente")
    related = provider.embed_query("regras de cancelamento da consulta")
    unrelated = provider.embed_query("astronomia gal\u00e1xias estrelas")

    assert reference == repeated
    assert len(reference) == 64
    assert math.sqrt(sum(value * value for value in reference)) == pytest.approx(1.0)
    assert _cosine(reference, related) > _cosine(reference, unrelated)
    assert all(math.isfinite(value) for value in reference)


def test_fake_embedder_preserves_input_order() -> None:
    provider = DeterministicFakeEmbeddingProvider(32)
    texts = ["cancelamento", "f\u00e9rias", "privacidade"]

    assert provider.embed_documents(texts) == [provider.embed_query(text) for text in texts]


@pytest.mark.parametrize("text", ["", " ", "---"])
def test_fake_embedder_rejects_text_without_tokens(text: str) -> None:
    with pytest.raises(RetrievalError, match="vazio"):
        DeterministicFakeEmbeddingProvider(32).embed_query(text)


@pytest.mark.parametrize(
    ("texts", "vectors", "dimensions"),
    [
        ([], [], 2),
        (["ok", ""], [[1.0, 0.0], [0.0, 1.0]], 2),
        (["one", "two"], [[1.0, 0.0]], 2),
        (["one"], [[1.0]], 2),
        (["one"], [[float("inf"), 0.0]], 2),
        (["one"], [[float("nan"), 0.0]], 2),
    ],
)
def test_validate_vectors_rejects_invalid_responses(
    texts: list[str], vectors: list[list[float]], dimensions: int
) -> None:
    with pytest.raises(RetrievalError):
        validate_vectors(texts, vectors, dimensions)


def test_validate_vectors_returns_detached_lists() -> None:
    vector = (1.0, 0.0)

    assert validate_vectors(("text",), (vector,), 2) == [[1.0, 0.0]]


def test_openai_provider_preserves_order_batches_and_usage_with_injected_client() -> None:
    client = FakeEmbeddingsClient(
        [
            _response([(1, [0.0, 1.0]), (0, [1.0, 0.0])], 3, "req_batch-1"),
            _response([(0, [0.5, 0.5])], 2, "unsafe request id with spaces"),
        ]
    )
    config = RetrievalConfig(
        app_env="test",
        embedding_provider="openai",
        openai_api_key="",
        embedding_model="fake-openai-model",
        embedding_dimensions=2,
        embedding_batch_size=2,
    )
    provider = OpenAIEmbeddingProvider(config, client)

    result = provider.embed_documents_with_usage(["first", "second", "third"])
    assert result.vector_lists() == [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5],
    ]
    assert client.embeddings.calls == [
        (["first", "second"], "fake-openai-model", 2, "float"),
        (["third"], "fake-openai-model", 2, "float"),
    ]
    assert provider.total_tokens == 5
    assert result.usage.input_count == 3
    assert result.usage.batch_count == result.usage.logical_api_calls == 2
    assert result.usage.physical_attempts == 2
    assert result.usage.prompt_tokens == result.usage.total_tokens == 5
    assert result.usage.batches[0].request_id == "req_batch-1"
    assert result.usage.batches[1].request_id is None


def test_openai_provider_marks_missing_usage_and_hidden_retries_unavailable() -> None:
    response = _response([(0, [1.0, 0.0])], 1)
    object.__setattr__(response, "usage", None)
    client = FakeEmbeddingsClient([response])
    config = RetrievalConfig(
        app_env="test",
        embedding_provider="openai",
        openai_api_key="",
        embedding_dimensions=2,
        embedding_batch_size=2,
        openai_max_retries=1,
    )

    usage = OpenAIEmbeddingProvider(config, client).embed_documents_with_usage(["first"]).usage

    assert usage.prompt_tokens is None and usage.total_tokens is None
    assert usage.physical_attempts is None
    assert usage.transport_attempts_observable is False
    assert usage.batches[0].attempt_count is None


def test_openai_provider_rejects_duplicate_response_indices() -> None:
    client = FakeEmbeddingsClient([_response([(0, [1.0, 0.0]), (0, [0.0, 1.0])], total_tokens=2)])
    config = RetrievalConfig(
        app_env="test",
        embedding_provider="openai",
        openai_api_key="",
        embedding_dimensions=2,
        embedding_batch_size=2,
    )

    with pytest.raises(RetrievalError, match="ndices"):
        OpenAIEmbeddingProvider(config, client).embed_documents(["first", "second"])
