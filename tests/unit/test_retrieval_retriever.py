"""Tests for citations, thresholds, and explicit retrieval fallback."""

from __future__ import annotations

import pytest

from nexodocs_ai.retrieval.embeddings import DeterministicFakeEmbeddingProvider
from nexodocs_ai.retrieval.models import RetrievalError, RetrievalFilters, StoredSearchResult
from nexodocs_ai.retrieval.qdrant_store import QdrantStore
from nexodocs_ai.retrieval.retriever import Retriever, citation_label


class StubStore(QdrantStore):
    def __init__(self, points: list[StoredSearchResult]) -> None:
        self._points = list(points)
        self.calls: list[tuple[list[float], int, dict[str, str]]] = []

    def search(
        self, vector: list[float], limit: int, filters: dict[str, str]
    ) -> list[StoredSearchResult]:
        self.calls.append((vector, limit, dict(filters)))
        return self._points[:limit]


def _point(
    score: float,
    chunk_id: str,
    document_id: str,
    *,
    section_title: str | None = None,
) -> StoredSearchResult:
    payload: dict[str, object] = {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "text": "Evid\u00eancia fict\u00edcia sobre cancelamento.",
        "source_filename": "politica_cancelamentos.pdf",
        "source_format": "pdf",
        "title": "Pol\u00edtica de cancelamentos",
        "locator": "p\u00e1gina 3",
        "page_number": 3,
    }
    if section_title is not None:
        payload["section_title"] = section_title
    return StoredSearchResult(score=score, payload=payload)


def test_citation_label_formats_pdf_and_csv_locations() -> None:
    separator = " \N{EM DASH} "
    pdf: dict[str, object] = {
        "source_format": "pdf",
        "title": "Pol\u00edtica",
        "page_number": 4,
        "section_title": "Reagendamento",
    }
    csv: dict[str, object] = {
        "source_format": "csv",
        "title": "Diret\u00f3rio",
        "row_number": 7,
    }

    assert citation_label(pdf) == f"Pol\u00edtica{separator}p\u00e1gina 4{separator}Reagendamento"
    assert citation_label(csv) == f"Diret\u00f3rio{separator}linha 7"


def test_retriever_applies_threshold_and_preserves_structured_evidence() -> None:
    store = StubStore(
        [
            _point(0.91, "chunk-high", "DOC-001", section_title="Cancelamento"),
            _point(0.20, "chunk-low", "DOC-002"),
        ]
    )
    retriever = Retriever(DeterministicFakeEmbeddingProvider(32), store)

    response = retriever.retrieve("como cancelar", top_k=2, score_threshold=0.5)

    assert response.status == "found"
    assert response.reason is None
    assert len(response.results) == 1
    assert response.results[0].chunk_id == "chunk-high"
    assert response.results[0].score == pytest.approx(0.91)
    assert response.results[0].text == "Evid\u00eancia fict\u00edcia sobre cancelamento."
    assert response.results[0].source == "politica_cancelamentos.pdf"
    assert response.results[0].citation_label.endswith("Cancelamento")


def test_retriever_returns_explicit_fallback_below_threshold_and_applies_filters() -> None:
    store = StubStore([_point(0.49, "chunk-low", "DOC-001")])
    retriever = Retriever(DeterministicFakeEmbeddingProvider(32), store)
    filters = RetrievalFilters(category="operations", classification="internal")

    response = retriever.retrieve(
        "evid\u00eancia inexistente",
        top_k=3,
        filters=filters,
        score_threshold=0.5,
    )

    assert response.status == "no_evidence"
    assert response.results == ()
    assert response.reason == "no_match_or_below_threshold"
    assert response.applied_filters == {
        "category": "operations",
        "classification": "internal",
    }
    assert store.calls[0][1] == 6
    assert store.calls[0][2] == response.applied_filters


@pytest.mark.parametrize("query", ["", "   "])
def test_retriever_rejects_empty_queries(query: str) -> None:
    retriever = Retriever(DeterministicFakeEmbeddingProvider(32), StubStore([]))

    with pytest.raises(RetrievalError, match="Query"):
        retriever.retrieve(query)
