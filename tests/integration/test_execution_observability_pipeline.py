"""Offline end-to-end observability with fake providers and in-memory Qdrant."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from nexodocs_ai.knowledge_base.generator import repository_root
from nexodocs_ai.observability.reports import (
    answer_report,
    index_report,
    privacy_safe_report,
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


def test_privacy_safe_reporting_preserves_fake_rag_behavior(tmp_path: Path) -> None:
    """The report projection must not affect a deterministic RAG run."""
    _, _, _, rag = _pipeline()
    retrieval = load_config(
        {
            "APP_ENV": "test",
            "EMBEDDING_PROVIDER": "fake",
            "OPENAI_EMBEDDING_DIMENSIONS": "192",
            "QDRANT_MODE": "memory",
        }
    )
    request = RagRequest("Como funciona o cancelamento?")
    standard_run = rag.answer_with_usage(request)
    private_run = rag.answer_with_usage(request)
    assert private_run.response == standard_run.response
    assert private_run.retrieval_usage == standard_run.retrieval_usage
    assert private_run.answer_usage == standard_run.answer_usage
    assert private_run.application_attempts == standard_run.application_attempts
    assert (0 if private_run.response.status in {"answered", "no_evidence"} else 1) == (
        0 if standard_run.response.status in {"answered", "no_evidence"} else 1
    )

    standard = answer_report(
        standard_run,
        rag.provider.provider_name,
        rag.provider.model_identifier,
        retrieval.collection_name,
        len(request.query),
        1,
    )
    private = privacy_safe_report(standard)
    standard_data = standard.as_dict()
    for field in (
        "status",
        "safe_error_code",
        "logical_api_calls",
        "physical_attempts",
        "prompt_tokens",
        "cached_input_tokens",
        "output_tokens",
        "total_tokens",
        "application_attempts",
    ):
        assert private[field] == standard_data[field]
    private_path = write_report(tmp_path, "data/run-reports/private.json", private)
    persisted = json.loads(private_path.read_text(encoding="utf-8"))

    def collect_keys(value: object) -> set[str]:
        if isinstance(value, dict):
            mapping = cast(dict[str, object], value)
            return set(mapping) | set().union(*(collect_keys(item) for item in mapping.values()))
        return set()

    assert not {
        "request_id",
        "request_ids",
        "run_id",
        "timestamp",
        "timestamp_utc",
        "prompt",
        "context",
    } & collect_keys(persisted)
    assert request.query not in private_path.read_text(encoding="utf-8")
