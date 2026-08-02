"""Typed contracts for indexing and retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol


class RetrievalError(ValueError):
    """Raised when a retrieval invariant is violated."""


class ConfigurationError(RetrievalError):
    """Raised for unsafe or incomplete runtime configuration."""


class CollectionCompatibilityError(RetrievalError):
    """Raised when an existing collection is incompatible."""


class EmbeddingProvider(Protocol):
    """Structural contract implemented by embedding providers."""

    provider_name: str
    model_identifier: str
    dimensions: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_documents_with_usage(self, texts: Sequence[str]) -> EmbeddingResult: ...

    def embed_query(self, text: str) -> list[float]: ...

    def embed_query_with_usage(self, text: str) -> EmbeddingResult: ...


class EmbeddingIdentity(Protocol):
    """Non-operational embedding identity used by offline plans."""

    @property
    def provider_name(self) -> str: ...

    @property
    def model_identifier(self) -> str: ...

    @property
    def dimensions(self) -> int: ...


@dataclass(frozen=True)
class EmbeddingSpecification:
    """Embedding identity without a client or embedding operation."""

    provider_name: str
    model_identifier: str
    dimensions: int


@dataclass(frozen=True)
class EmbeddingBatchUsage:
    """Safe metadata for one logical embeddings API batch."""

    batch_index: int
    input_count: int
    prompt_tokens: int | None
    total_tokens: int | None
    request_id: str | None
    attempt_count: int | None


@dataclass(frozen=True)
class EmbeddingRunUsage:
    """Aggregate embedding usage without source text, vectors, or SDK objects."""

    provider: str
    model: str
    dimensions: int
    batch_size: int
    input_count: int
    batch_count: int
    logical_api_calls: int
    physical_attempts: int | None
    prompt_tokens: int | None
    total_tokens: int | None
    transport_attempts_observable: bool
    batches: tuple[EmbeddingBatchUsage, ...]


@dataclass(frozen=True)
class EmbeddingResult:
    """Immutable vectors plus independently safe usage metadata."""

    vectors: tuple[tuple[float, ...], ...] = field(repr=False)
    usage: EmbeddingRunUsage

    def vector_lists(self) -> list[list[float]]:
        """Return detached mutable vectors for vector-store clients."""
        return [list(vector) for vector in self.vectors]


@dataclass(frozen=True)
class RetrievalConfig:
    app_env: str
    embedding_provider: str
    openai_api_key: str = field(repr=False)
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    openai_timeout_seconds: float = 30.0
    openai_max_retries: int = 0
    embedding_batch_size: int = 32
    qdrant_mode: str = "local"
    qdrant_url: str = field(default="", repr=False)
    qdrant_api_key: str = field(default="", repr=False)
    qdrant_path: str = "data/qdrant"
    collection_name: str = "nexodocs_chunks_v1"
    qdrant_timeout_seconds: int = 10
    top_k: int = 5
    max_top_k: int = 20
    max_per_document: int = 2
    score_threshold: float | None = None

    def safe_summary(self) -> dict[str, object]:
        """Return useful non-sensitive configuration fields only."""
        return {
            "app_env": self.app_env,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
            "qdrant_mode": self.qdrant_mode,
            "collection_name": self.collection_name,
        }


@dataclass(frozen=True)
class PlannedPoint:
    chunk_id: str
    point_id: str
    document_id: str
    text_sha256: str
    payload_sha256: str


@dataclass(frozen=True)
class IndexPlan:
    data: dict[str, object]
    points: tuple[PlannedPoint, ...]


@dataclass(frozen=True)
class IndexManifest:
    """Deterministic metadata written only after a successful real index operation."""

    data: dict[str, object]


@dataclass(frozen=True)
class IndexingResult:
    inserted_or_updated: int
    reused: int
    obsolete: int
    total_points: int


@dataclass(frozen=True)
class IndexingOperationResult:
    result: IndexingResult
    embedding_usage: EmbeddingRunUsage


@dataclass(frozen=True)
class StoredPoint:
    point_id: str
    payload: dict[str, object]


@dataclass(frozen=True)
class StoredSearchResult:
    score: float
    payload: dict[str, object]


@dataclass(frozen=True)
class RetrievalFilters:
    document_id: str | None = None
    category: str | None = None
    source_format: str | None = None
    owner_area: str | None = None
    version: str | None = None
    classification: str | None = None

    def as_dict(self) -> dict[str, str]:
        """Return non-empty filters in stable field order."""
        values = {
            "document_id": self.document_id,
            "category": self.category,
            "source_format": self.source_format,
            "owner_area": self.owner_area,
            "version": self.version,
            "classification": self.classification,
        }
        if any(value is not None and not value.strip() for value in values.values()):
            raise RetrievalError("Filter values cannot be blank.")
        return {key: value for key, value in values.items() if value is not None}


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    score: float
    text: str
    source: str
    locator: str
    citation_label: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class RetrievalResponse:
    status: str
    query: str
    results: tuple[RetrievalResult, ...]
    applied_filters: dict[str, str]
    reason: str | None = None


@dataclass(frozen=True)
class RetrievalOperationResult:
    response: RetrievalResponse
    embedding_usage: EmbeddingRunUsage


@dataclass(frozen=True)
class EvaluationMetrics:
    total_cases: int
    passed_cases: int
    hit_rate_at_k: float
    recall_at_k: float
    mrr: float
    fallback_precision: float


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    query: str
    expected_status: str
    expected_document_ids: tuple[str, ...]
    optional_filters: RetrievalFilters
    top_k: int
    fake_score_threshold: float | None = None
