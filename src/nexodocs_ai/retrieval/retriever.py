"""Structured retrieval with citations and explicit no-evidence fallback."""

from __future__ import annotations

import logging
import math

from .constants import MAX_QUERY_CHARACTERS
from .models import (
    EmbeddingProvider,
    RetrievalError,
    RetrievalFilters,
    RetrievalOperationResult,
    RetrievalResponse,
    RetrievalResult,
)
from .qdrant_store import QdrantStore

LOGGER = logging.getLogger(__name__)


def citation_label(payload: dict[str, object]) -> str:
    """Format human-readable source citations from preserved locations."""
    required = {"source_format", "title"}
    if not required <= payload.keys():
        raise RetrievalError("Retrieved payload is missing citation metadata")
    if payload["source_format"] == "pdf":
        label = f"{payload['title']} — página {payload['page_number']}"
        return f"{label} — {payload['section_title']}" if "section_title" in payload else label
    return f"{payload['title']} — linha {payload['row_number']}"


class Retriever:
    """Retrieve chunks only; it intentionally does not generate answers."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        store: QdrantStore,
        default_top_k: int = 5,
        max_top_k: int = 20,
        max_per_document: int = 2,
    ) -> None:
        self.provider, self.store, self.default_top_k, self.max_top_k, self.max_per_document = (
            provider,
            store,
            default_top_k,
            max_top_k,
            max_per_document,
        )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
        score_threshold: float | None = None,
    ) -> RetrievalResponse:
        """Return ranked, deduplicated evidence or a typed no-evidence result."""
        return self.retrieve_with_usage(query, top_k, filters, score_threshold).response

    def retrieve_with_usage(
        self,
        query: str,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
        score_threshold: float | None = None,
    ) -> RetrievalOperationResult:
        """Return retrieval results with safe query-embedding usage metadata."""
        if not query.strip() or len(query) > MAX_QUERY_CHARACTERS:
            raise RetrievalError("Query vazia ou excessivamente longa")
        limit = self.default_top_k if top_k is None else top_k
        if limit < 1 or limit > self.max_top_k:
            raise RetrievalError("top_k fora do limite permitido")
        if score_threshold is not None and (
            not math.isfinite(score_threshold) or not -1.0 <= score_threshold <= 1.0
        ):
            raise RetrievalError("score_threshold must be finite and between -1 and 1")
        applied = {} if filters is None else filters.as_dict()
        embedding = self.provider.embed_query_with_usage(query)
        points = self.store.search(
            embedding.vector_lists()[0], limit * self.max_per_document, applied
        )
        results: list[RetrievalResult] = []
        seen: set[str] = set()
        per_document: dict[str, int] = {}
        for point in sorted(points, key=lambda item: item.score, reverse=True):
            payload = point.payload
            required = {
                "chunk_id",
                "document_id",
                "text",
                "source_filename",
                "locator",
                "source_format",
                "title",
            }
            if not required <= payload.keys():
                raise RetrievalError("Retrieved payload is incomplete")
            score = float(point.score)
            chunk_id, document_id = str(payload["chunk_id"]), str(payload["document_id"])
            if score_threshold is not None and score < score_threshold:
                continue
            if chunk_id in seen or per_document.get(document_id, 0) >= self.max_per_document:
                continue
            seen.add(chunk_id)
            per_document[document_id] = per_document.get(document_id, 0) + 1
            results.append(
                RetrievalResult(
                    chunk_id,
                    score,
                    str(payload["text"]),
                    str(payload["source_filename"]),
                    str(payload["locator"]),
                    citation_label(payload),
                    {key: value for key, value in payload.items() if key != "text"},
                )
            )
            if len(results) == limit:
                break
        if not results:
            LOGGER.info(
                "Retrieval completed top_k=%d filter_names=%s result_count=0",
                limit,
                sorted(applied),
            )
            return RetrievalOperationResult(
                RetrievalResponse(
                    "no_evidence", query, (), dict(applied), "no_match_or_below_threshold"
                ),
                embedding.usage,
            )
        scores = [result.score for result in results]
        LOGGER.info(
            "Retrieval completed top_k=%d filter_names=%s result_count=%d score_min=%.6f score_max=%.6f score_mean=%.6f",
            limit,
            sorted(applied),
            len(results),
            min(scores),
            max(scores),
            sum(scores) / len(scores),
        )
        return RetrievalOperationResult(
            RetrievalResponse("found", query, tuple(results), dict(applied)), embedding.usage
        )
