"""Small orchestration layer; it never broadens retrieved evidence."""

from __future__ import annotations

import logging
from typing import Literal

from nexodocs_ai.retrieval.embeddings import empty_embedding_usage
from nexodocs_ai.retrieval.retriever import Retriever

from .config import RagConfig
from .constants import FALLBACK_MESSAGES, SCHEMA_VERSION
from .context_builder import ContextBuilder
from .evidence import assess_context, assess_retrieval, preflight
from .grounding_diagnostics import safe_grounding_error_code
from .models import (
    AnswerGenerationRequest,
    AnswerProvider,
    EvidenceSummary,
    ProviderError,
    ProviderRefusalError,
    RagError,
    RagRequest,
    RagResponse,
    RagRunResult,
    RetrievalSummary,
    ValidationError,
)
from .prompts import load_prompts, prompt_sha256, render_answer_prompt
from .validator import validate_generated

LOGGER = logging.getLogger(__name__)


class RagPipeline:
    def __init__(self, retriever: Retriever, provider: AnswerProvider, config: RagConfig) -> None:
        self.retriever, self.provider, self.config = retriever, provider, config

    def _fallback(
        self,
        request: RagRequest,
        status: Literal["no_evidence", "invalid_request", "generation_failed", "grounding_failed"],
        reason: str,
    ) -> RagResponse:
        return RagResponse(
            SCHEMA_VERSION,
            status,
            request.query,
            (),
            reason_code=reason,
            message=FALLBACK_MESSAGES[reason],
        )

    def answer(self, request: RagRequest) -> RagResponse:
        """Return the unchanged public response without internal usage metadata."""
        return self.answer_with_usage(request).response

    def answer_with_usage(self, request: RagRequest) -> RagRunResult:
        """Return the public response plus safe usage for an authorized local report."""
        LOGGER.info("rag_request_started query_length=%d", len(request.query))
        no_retrieval = empty_embedding_usage(self.retriever.provider)
        try:
            if not request.query.strip() or len(request.query) > self.config.max_query_characters:
                return RagRunResult(
                    self._fallback(request, "invalid_request", "invalid_query"),
                    no_retrieval,
                    None,
                    0,
                )
            if request.filters:
                request.filters.as_dict()
            scope = preflight(request.query)
            if scope.status != "sufficient":
                return RagRunResult(
                    self._fallback(request, "no_evidence", scope.reason_code or "out_of_scope"),
                    no_retrieval,
                    None,
                    0,
                )
            retrieval = self.retriever.retrieve_with_usage(
                request.query, request.top_k, request.filters, request.score_threshold
            )
            response = retrieval.response
            assessed = assess_retrieval(response, self.config.min_evidence_results)
            if assessed.status != "sufficient":
                return RagRunResult(
                    self._fallback(
                        request, "no_evidence", assessed.reason_code or "insufficient_evidence"
                    ),
                    retrieval.embedding_usage,
                    None,
                    0,
                )
            context = ContextBuilder(
                request.max_context_characters or self.config.max_context_characters,
                request.max_results_per_document or self.retriever.max_per_document,
                self.config.max_context_chunks,
            ).build(response.results)
            assessed = assess_context(context, self.config.min_context_characters)
            if assessed.status != "sufficient":
                return RagRunResult(
                    self._fallback(
                        request, "no_evidence", assessed.reason_code or "context_below_minimum"
                    ),
                    retrieval.embedding_usage,
                    None,
                    0,
                )
            system, template = load_prompts()
            from .answer_provider import generated_answer_schema

            digest = prompt_sha256(system, template)
            user = render_answer_prompt(
                template,
                request.query,
                context.evidence_blocks,
                generated_answer_schema(),
                self.config.max_answer_characters,
            )
            generation = AnswerGenerationRequest(
                system,
                user,
                tuple(item.evidence_id for item in context.evidence_blocks),
                self.config.max_answer_characters,
                self.config.prompt_version,
                digest,
                1,
                context.evidence_blocks,
            )
            try:
                generated = self.provider.generate(generation)
            except ProviderRefusalError as exc:
                return RagRunResult(
                    self._fallback(request, "generation_failed", "provider_refusal"),
                    retrieval.embedding_usage,
                    exc.usage,
                    1,
                )
            except ProviderError as exc:
                return RagRunResult(
                    self._fallback(request, "generation_failed", "provider_invalid_output"),
                    retrieval.embedding_usage,
                    exc.usage,
                    1,
                )
            try:
                answer, citations = validate_generated(
                    generated, context.evidence_blocks, self.config.max_answer_characters
                )
            except ValidationError as exc:
                fallback = self._fallback(
                    request, "grounding_failed", "grounding_validation_failed"
                )
                return RagRunResult(
                    RagResponse(
                        fallback.schema_version,
                        fallback.status,
                        fallback.query,
                        fallback.warnings,
                        reason_code=safe_grounding_error_code(exc),
                        message=fallback.message,
                    ),
                    retrieval.embedding_usage,
                    generated.usage,
                    1,
                )
            by_id = {item.evidence_id: item for item in context.evidence_blocks}
            evidence = tuple(
                EvidenceSummary(
                    item.citation_id,
                    by_id[item.citation_id].chunk_id,
                    by_id[item.citation_id].document_id,
                    by_id[item.citation_id].citation_label,
                    by_id[item.citation_id].score,
                    by_id[item.citation_id].text_sha256,
                )
                for item in citations
            )
            summary = RetrievalSummary(
                response.status,
                len(response.results),
                len(context.evidence_blocks),
                len(context.discarded_chunks),
                len({item.document_id for item in context.evidence_blocks}),
                max(item.score for item in context.evidence_blocks),
                response.applied_filters,
            )
            debug: dict[str, object] | None = (
                {
                    "included_chunk_ids": context.included_chunk_ids,
                    "context_sha256": context.context_sha256,
                    "prompt_sha256": digest,
                    "attempt_count": 1,
                }
                if request.include_debug_metadata and self.config.include_debug_metadata
                else None
            )
            public = RagResponse(
                SCHEMA_VERSION,
                "answered",
                request.query,
                (),
                answer,
                citations,
                evidence,
                summary,
                self.provider.provider_name,
                self.provider.model_identifier,
                self.config.prompt_version,
                debug=debug,
            )
            return RagRunResult(public, retrieval.embedding_usage, generated.usage, 1)
        except RagError:
            return RagRunResult(
                self._fallback(request, "generation_failed", "provider_invalid_output"),
                no_retrieval,
                None,
                0,
            )
