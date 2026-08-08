"""Validated OpenAI and deterministic lexical embedding providers."""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections.abc import Sequence
from typing import Literal, Protocol

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)
from openai.types import CreateEmbeddingResponse

from nexodocs_ai.observability.safety import safe_opaque_identifier

from .constants import DEFAULT_FAKE_DIMENSIONS
from .models import (
    ConfigurationError,
    EmbeddingBatchUsage,
    EmbeddingProvider,
    EmbeddingResult,
    EmbeddingRunUsage,
    RetrievalConfig,
    RetrievalError,
)

_TOKEN = re.compile(r"[\wÀ-ÿ]+", re.UNICODE)
LOGGER = logging.getLogger(__name__)


class EmbeddingProviderError(RetrievalError):
    """Closed OpenAI embedding failure without provider exception details."""

    code = "embedding_provider_unavailable"

    def __init__(self) -> None:
        super().__init__(self.code)


class EmbeddingsEndpoint(Protocol):
    """Subset of the OpenAI embeddings resource used by this package."""

    def create(
        self,
        *,
        input: str | list[str],
        model: str,
        dimensions: int,
        encoding_format: Literal["float"],
    ) -> CreateEmbeddingResponse: ...


class EmbeddingsClient(Protocol):
    """Injectable OpenAI-compatible client boundary for offline tests."""

    @property
    def embeddings(self) -> EmbeddingsEndpoint: ...


def validate_vectors(
    texts: Sequence[str], vectors: Sequence[Sequence[float]], dimensions: int
) -> list[list[float]]:
    """Reject partial, malformed, non-finite, or wrongly sized responses."""
    if not texts or any(not text.strip() for text in texts):
        raise RetrievalError("Textos para embedding não podem ser vazios")
    if len(texts) != len(vectors):
        raise RetrievalError("Quantidade de vetores não corresponde às entradas")
    output = [list(vector) for vector in vectors]
    if any(
        len(vector) != dimensions or not vector or not all(math.isfinite(value) for value in vector)
        for vector in output
    ):
        raise RetrievalError("Vetores inválidos para a dimensão configurada")
    return output


class DeterministicFakeEmbeddingProvider:
    """Offline lexical hash embedder for tests, not semantic production quality."""

    provider_name = "deterministic-fake"
    model_identifier = "lexical-hash-test-v1"

    def __init__(self, dimensions: int = DEFAULT_FAKE_DIMENSIONS) -> None:
        if dimensions < 8:
            raise RetrievalError("Dimensão fake deve ser ao menos 8")
        self.dimensions = dimensions
        self.batch_size = 0

    def _embed(self, text: str) -> list[float]:
        tokens = _TOKEN.findall(text.casefold())
        if not tokens:
            raise RetrievalError("Texto para embedding não pode ser vazio")
        vector = [0.0] * self.dimensions
        features = [*tokens, *(f"{left}|{right}" for left, right in zip(tokens, tokens[1:]))]
        for feature in features:
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:8], "big") % self.dimensions
            vector[bucket] += 1.0 if digest[8] & 1 else -1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            fallback = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
            vector[fallback % self.dimensions] = 1.0
            norm = 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed_documents_with_usage(texts).vector_lists()

    def embed_documents_with_usage(self, texts: Sequence[str]) -> EmbeddingResult:
        """Embed locally while declaring that no API call or token usage occurred."""
        vectors = validate_vectors(texts, [self._embed(text) for text in texts], self.dimensions)
        usage = EmbeddingRunUsage(
            self.provider_name,
            self.model_identifier,
            self.dimensions,
            self.batch_size,
            len(texts),
            0,
            0,
            0,
            None,
            None,
            True,
            (),
        )
        return EmbeddingResult(tuple(tuple(vector) for vector in vectors), usage)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_query_with_usage(text).vector_lists()[0]

    def embed_query_with_usage(self, text: str) -> EmbeddingResult:
        """Embed one local query with explicit zero API usage."""
        return self.embed_documents_with_usage([text])


class OpenAIEmbeddingProvider:
    """OpenAI provider with an injectable client and no logging of sensitive data."""

    provider_name = "openai"

    def __init__(self, config: RetrievalConfig, client: EmbeddingsClient | None = None) -> None:
        if not config.openai_api_key and client is None:
            raise ConfigurationError("OPENAI_API_KEY é obrigatória para OpenAI")
        self.model_identifier, self.dimensions, self.batch_size = (
            config.embedding_model,
            config.embedding_dimensions,
            config.embedding_batch_size,
        )
        self._client: EmbeddingsClient = client or OpenAI(
            api_key=config.openai_api_key,
            timeout=config.openai_timeout_seconds,
            max_retries=config.openai_max_retries,
        )
        self._transport_retries = config.openai_max_retries
        self.total_tokens = 0

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed_documents_with_usage(texts).vector_lists()

    def embed_documents_with_usage(self, texts: Sequence[str]) -> EmbeddingResult:
        """Return validated vectors and closed usage metadata for all API batches."""
        if not texts or any(not text.strip() for text in texts):
            raise RetrievalError("Textos para embedding não podem ser vazios")
        ordered: list[list[float] | None] = [None] * len(texts)
        batches: list[EmbeddingBatchUsage] = []
        for start in range(0, len(texts), self.batch_size):
            LOGGER.info(
                "Embedding document batch provider=%s model=%s dimensions=%d size=%d",
                self.provider_name,
                self.model_identifier,
                self.dimensions,
                min(self.batch_size, len(texts) - start),
            )
            try:
                response = self._client.embeddings.create(
                    model=self.model_identifier,
                    input=list(texts[start : start + self.batch_size]),
                    dimensions=self.dimensions,
                    encoding_format="float",
                )
            except (
                BadRequestError,
                AuthenticationError,
                PermissionDeniedError,
                NotFoundError,
                RateLimitError,
                APITimeoutError,
                APIConnectionError,
                APIStatusError,
            ):
                raise EmbeddingProviderError() from None
            data = list(response.data)
            if sorted(item.index for item in data) != list(range(len(data))):
                raise RetrievalError("Índices de embedding OpenAI inválidos")
            for item in data:
                ordered[start + item.index] = list(item.embedding)
            prompt_tokens = getattr(response.usage, "prompt_tokens", None)
            total_tokens = getattr(response.usage, "total_tokens", None)
            self.total_tokens += total_tokens if isinstance(total_tokens, int) else 0
            batches.append(
                EmbeddingBatchUsage(
                    len(batches) + 1,
                    len(texts[start : start + self.batch_size]),
                    prompt_tokens if isinstance(prompt_tokens, int) else None,
                    total_tokens if isinstance(total_tokens, int) else None,
                    safe_opaque_identifier(getattr(response, "_request_id", None)),
                    1 if self._transport_retries == 0 else None,
                )
            )
        if any(vector is None for vector in ordered):
            raise RetrievalError("Resposta OpenAI parcial")
        vectors = validate_vectors(
            texts, [vector for vector in ordered if vector is not None], self.dimensions
        )
        prompt_values = [batch.prompt_tokens for batch in batches]
        total_values = [batch.total_tokens for batch in batches]
        attempts_observable = self._transport_retries == 0
        usage = EmbeddingRunUsage(
            self.provider_name,
            self.model_identifier,
            self.dimensions,
            self.batch_size,
            len(texts),
            len(batches),
            len(batches),
            len(batches) if attempts_observable else None,
            sum(value for value in prompt_values if value is not None)
            if all(value is not None for value in prompt_values)
            else None,
            sum(value for value in total_values if value is not None)
            if all(value is not None for value in total_values)
            else None,
            attempts_observable,
            tuple(batches),
        )
        return EmbeddingResult(tuple(tuple(vector) for vector in vectors), usage)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_query_with_usage(text).vector_lists()[0]

    def embed_query_with_usage(self, text: str) -> EmbeddingResult:
        """Embed one query and return its safe API usage."""
        return self.embed_documents_with_usage([text])


def empty_embedding_usage(provider: EmbeddingProvider) -> EmbeddingRunUsage:
    """Describe an operation that performed no embedding work."""
    return EmbeddingRunUsage(
        provider.provider_name,
        provider.model_identifier,
        provider.dimensions,
        int(getattr(provider, "batch_size", 0)),
        0,
        0,
        0,
        0,
        None,
        None,
        True,
        (),
    )


def create_embedding_provider(
    config: RetrievalConfig, client: EmbeddingsClient | None = None
) -> EmbeddingProvider:
    """Create only the provider permitted by the validated environment."""
    if config.embedding_provider == "fake":
        if config.app_env != "test":
            raise ConfigurationError("Provedor fake é permitido somente em teste")
        return DeterministicFakeEmbeddingProvider(config.embedding_dimensions)
    return OpenAIEmbeddingProvider(config, client)
