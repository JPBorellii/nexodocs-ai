"""Validated OpenAI and deterministic lexical embedding providers."""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections.abc import Sequence
from typing import Literal, Protocol

from openai import OpenAI
from openai.types import CreateEmbeddingResponse

from .constants import DEFAULT_FAKE_DIMENSIONS
from .models import ConfigurationError, EmbeddingProvider, RetrievalConfig, RetrievalError

_TOKEN = re.compile(r"[\wÀ-ÿ]+", re.UNICODE)
LOGGER = logging.getLogger(__name__)


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
        return validate_vectors(texts, [self._embed(text) for text in texts], self.dimensions)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


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
        self.total_tokens = 0

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts or any(not text.strip() for text in texts):
            raise RetrievalError("Textos para embedding não podem ser vazios")
        ordered: list[list[float] | None] = [None] * len(texts)
        for start in range(0, len(texts), self.batch_size):
            LOGGER.info(
                "Embedding document batch provider=%s model=%s dimensions=%d size=%d",
                self.provider_name,
                self.model_identifier,
                self.dimensions,
                min(self.batch_size, len(texts) - start),
            )
            response = self._client.embeddings.create(
                model=self.model_identifier,
                input=list(texts[start : start + self.batch_size]),
                dimensions=self.dimensions,
                encoding_format="float",
            )
            data = list(response.data)
            if sorted(item.index for item in data) != list(range(len(data))):
                raise RetrievalError("Índices de embedding OpenAI inválidos")
            for item in data:
                ordered[start + item.index] = list(item.embedding)
            self.total_tokens += response.usage.total_tokens
        if any(vector is None for vector in ordered):
            raise RetrievalError("Resposta OpenAI parcial")
        return validate_vectors(
            texts, [vector for vector in ordered if vector is not None], self.dimensions
        )

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def create_embedding_provider(
    config: RetrievalConfig, client: EmbeddingsClient | None = None
) -> EmbeddingProvider:
    """Create only the provider permitted by the validated environment."""
    if config.embedding_provider == "fake":
        if config.app_env != "test":
            raise ConfigurationError("Provedor fake é permitido somente em teste")
        return DeterministicFakeEmbeddingProvider(config.embedding_dimensions)
    return OpenAIEmbeddingProvider(config, client)
