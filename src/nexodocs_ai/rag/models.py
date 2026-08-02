"""Typed contracts kept separate from retrieval and provider SDK types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from nexodocs_ai.retrieval.models import EmbeddingRunUsage, RetrievalFilters


@dataclass(frozen=True)
class GenerationUsage:
    """Safe generation metadata detached from all provider SDK objects."""

    provider: str
    model: str
    input_tokens: int | None
    cached_input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    request_id: str | None
    logical_api_calls: int
    physical_attempts: int | None
    refusal_detected: bool
    application_attempt: int
    transport_attempts_observable: bool


class RagError(ValueError):
    """Base error for a controlled RAG contract failure."""


class ProviderError(RagError):
    """Provider failed without producing a valid answer."""

    def __init__(self, message: str, usage: GenerationUsage | None = None) -> None:
        super().__init__(message)
        self.usage = usage


class ProviderRefusalError(ProviderError):
    """Provider explicitly refused the request."""


class ValidationError(RagError):
    """Generated output violates a closed contract."""


@dataclass(frozen=True)
class RagRequest:
    query: str
    top_k: int | None = None
    score_threshold: float | None = None
    filters: RetrievalFilters | None = None
    max_context_characters: int | None = None
    max_results_per_document: int | None = None
    answer_language: Literal["pt-BR"] = "pt-BR"
    include_debug_metadata: bool = False


@dataclass(frozen=True)
class EvidenceBlock:
    evidence_id: int
    chunk_id: str
    document_id: str
    title: str
    source_filename: str
    locator: str
    citation_label: str
    score: float
    text: str
    text_sha256: str


@dataclass(frozen=True)
class DiscardedEvidence:
    chunk_id: str
    reason: str


@dataclass(frozen=True)
class ContextBuildResult:
    evidence_blocks: tuple[EvidenceBlock, ...]
    included_chunk_ids: tuple[str, ...]
    discarded_chunks: tuple[DiscardedEvidence, ...]
    total_characters: int
    total_words: int
    context_sha256: str


@dataclass(frozen=True)
class EvidenceAssessment:
    status: Literal["sufficient", "insufficient", "ambiguous", "out_of_scope"]
    reason_code: str | None = None


@dataclass(frozen=True)
class AnswerGenerationRequest:
    system_prompt: str
    user_prompt: str
    evidence_ids: tuple[int, ...]
    max_answer_characters: int
    prompt_version: str
    prompt_sha256: str
    attempt_number: int
    evidence_blocks: tuple[EvidenceBlock, ...] = field(repr=False)


@dataclass(frozen=True)
class GeneratedCitationReference:
    citation_id: int
    quote: str


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    citations: tuple[GeneratedCitationReference, ...]
    usage: GenerationUsage | None = None


class AnswerProvider(Protocol):
    provider_name: str
    model_identifier: str

    def generate(self, request: AnswerGenerationRequest) -> GeneratedAnswer: ...


@dataclass(frozen=True)
class Citation:
    citation_id: int
    chunk_id: str
    document_id: str
    title: str
    source_filename: str
    locator: str
    citation_label: str
    supporting_excerpt: str
    score: float


@dataclass(frozen=True)
class EvidenceSummary:
    citation_id: int
    chunk_id: str
    document_id: str
    citation_label: str
    score: float
    text_sha256: str


@dataclass(frozen=True)
class RetrievalSummary:
    retrieval_status: str
    retrieved_count: int
    included_count: int
    discarded_count: int
    document_count: int
    highest_score: float | None
    applied_filters: dict[str, str]


@dataclass(frozen=True)
class RagResponse:
    schema_version: str
    status: Literal[
        "answered", "no_evidence", "invalid_request", "generation_failed", "grounding_failed"
    ]
    query: str
    warnings: tuple[str, ...]
    answer: str | None = None
    citations: tuple[Citation, ...] = ()
    evidence: tuple[EvidenceSummary, ...] = ()
    retrieval_summary: RetrievalSummary | None = None
    provider: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    reason_code: str | None = None
    message: str | None = None
    debug: dict[str, object] | None = None


@dataclass(frozen=True)
class RagRunResult:
    """Internal response wrapper used only by authorized operational reporting."""

    response: RagResponse
    retrieval_usage: EmbeddingRunUsage
    answer_usage: GenerationUsage | None
    application_attempts: int


@dataclass(frozen=True)
class RagEvaluationMetrics:
    total_cases: int
    passed_cases: int
    status_accuracy: float
    citation_validity_rate: float
    citation_coverage: float
    fallback_accuracy: float
    required_term_coverage: float
    forbidden_term_violation_rate: float
    prompt_injection_resistance: float
    determinism: float
