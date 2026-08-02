"""Offline end-to-end observability with fake providers and in-memory Qdrant."""

from __future__ import annotations

import json
from pathlib import Path

from nexodocs_ai.knowledge_base.generator import repository_root
from nexodocs_ai.observability.reports import (
    answer_report,
    index_report,
    search_report,
    write_report,
)
from nexodocs_ai.rag.answer_provider import DeterministicFakeAnswerProvider
from nexodocs_ai.rag.config import load_rag_config
from nexodocs_ai.rag.models import RagRequest
from nexodocs_ai.rag.pipeline import RagPipeline
from nexodocs_ai.retrieval.config import load_config
from nexodocs_ai.retrieval.embeddings import DeterministicFakeEmbeddingProvider
from nexodocs_ai.retrieval.indexer import incremental_index_with_usage
from nexodocs_ai.retrieval.qdrant_store import QdrantStore, create_client
from nexodocs_ai.retrieval.retriever import Retriever


def _pipeline() -> tuple[DeterministicFakeEmbeddingProvider, QdrantStore, Retriever, RagPipeline]:
    retrieval = load_config(
        {
            "APP_ENV": "test",
            "EMBEDDING_PROVIDER": "fake",
            "OPENAI_EMBEDDING_DIMENSIONS": "192",
            "QDRANT_MODE": "memory",
        }
    )
    provider = DeterministicFakeEmbeddingProvider(retrieval.embedding_dimensions)
    store = QdrantStore(create_client(retrieval), retrieval.collection_name, provider.dimensions)
    first = incremental_index_with_usage(repository_root(), retrieval, provider, store)
    assert first.result.inserted_or_updated == 44
    assert first.embedding_usage.input_count == 44
    assert first.embedding_usage.logical_api_calls == 0
    second = incremental_index_with_usage(repository_root(), retrieval, provider, store)
    assert second.result.reused == 44
    assert second.embedding_usage.input_count == 0
    assert second.embedding_usage.logical_api_calls == 0
    retriever = Retriever(provider, store)
    rag_config = load_rag_config(
        {"APP_ENV": "test", "ANSWER_PROVIDER": "fake", "RAG_MIN_CONTEXT_CHARACTERS": "1"}
    )
    rag = RagPipeline(retriever, DeterministicFakeAnswerProvider(rag_config), rag_config)
    return provider, store, retriever, rag


def test_index_search_and_rag_reports_are_safe_and_offline(tmp_path: Path) -> None:
    provider, store, retriever, rag = _pipeline()
    retrieval = load_config(
        {
            "APP_ENV": "test",
            "EMBEDDING_PROVIDER": "fake",
            "OPENAI_EMBEDDING_DIMENSIONS": "192",
            "QDRANT_MODE": "memory",
        }
    )
    reused = incremental_index_with_usage(repository_root(), retrieval, provider, store)
    index_path = write_report(
        tmp_path,
        "data/run-reports/index.json",
        index_report(reused, retrieval.collection_name, 1),
    )
    assert json.loads(index_path.read_text(encoding="utf-8"))["logical_api_calls"] == 0

    searched = retriever.retrieve_with_usage("cancelamento 24 horas")
    search_path = write_report(
        tmp_path,
        "data/run-reports/search.json",
        search_report(
            searched.embedding_usage,
            retrieval.collection_name,
            len("cancelamento 24 horas"),
            searched.response.status,
            1,
        ),
    )
    search_data = json.loads(search_path.read_text(encoding="utf-8"))
    assert search_data["query_character_count"] == len("cancelamento 24 horas")
    assert "cancelamento" not in search_path.read_text(encoding="utf-8")

    answered = rag.answer_with_usage(RagRequest("Como funciona o cancelamento?"))
    answered_path = write_report(
        tmp_path,
        "data/run-reports/answer.json",
        answer_report(
            answered,
            rag.provider.provider_name,
            rag.provider.model_identifier,
            retrieval.collection_name,
            len("Como funciona o cancelamento?"),
            1,
        ),
    )
    answered_data = json.loads(answered_path.read_text(encoding="utf-8"))
    assert answered.response.status == "answered"
    assert answered_data["application_attempts"] == 1
    assert answered_data["answer_usage"]["logical_api_calls"] == 0
    assert answered.response.answer is not None
    assert answered.response.answer not in answered_path.read_text(encoding="utf-8")

    no_evidence = rag.answer_with_usage(
        RagRequest("astronomia gal\u00e1ctica", score_threshold=1.0)
    )
    assert no_evidence.response.status == "no_evidence"
    assert no_evidence.answer_usage is None
    assert no_evidence.application_attempts == 0

    blocked = rag.answer_with_usage(RagRequest("Fa\u00e7a um diagn\u00f3stico para mim"))
    blocked_report = answer_report(
        blocked,
        rag.provider.provider_name,
        rag.provider.model_identifier,
        retrieval.collection_name,
        len("Fa\u00e7a um diagn\u00f3stico para mim"),
        1,
    )
    blocked_data = blocked_report.as_dict()
    blocked_retrieval = blocked_data["retrieval_usage"]
    blocked_answer = blocked_data["answer_usage"]
    assert isinstance(blocked_retrieval, dict) and isinstance(blocked_answer, dict)
    assert blocked.response.reason_code == "clinical_guidance_not_supported"
    assert blocked.retrieval_usage.logical_api_calls == 0
    assert blocked_retrieval["logical_api_calls"] == 0
    assert blocked_answer["logical_api_calls"] == 0
