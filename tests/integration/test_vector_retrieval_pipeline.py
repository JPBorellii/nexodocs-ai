"""Offline integration tests for vector indexing and retrieval."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest
from qdrant_client.models import Distance, VectorParams

from nexodocs_ai.knowledge_base.generator import repository_root
from nexodocs_ai.retrieval.config import load_config
from nexodocs_ai.retrieval.embeddings import DeterministicFakeEmbeddingProvider
from nexodocs_ai.retrieval.evaluation import evaluate
from nexodocs_ai.retrieval.indexer import (
    build_index_manifest,
    build_plan,
    check_index,
    incremental_index,
    point_id,
)
from nexodocs_ai.retrieval.models import (
    CollectionCompatibilityError,
    ConfigurationError,
    RetrievalConfig,
    RetrievalError,
    RetrievalFilters,
)
from nexodocs_ai.retrieval.qdrant_store import QdrantStore, create_client
from nexodocs_ai.retrieval.retriever import Retriever

OPENAI_SECRET = "offline-openai-secret-must-not-appear"
QDRANT_SECRET = "offline-qdrant-secret-must-not-appear"
URL_SECRET = "url-password-must-not-appear"


@dataclass(frozen=True)
class MemoryPipeline:
    root: Path
    config: RetrievalConfig
    provider: DeterministicFakeEmbeddingProvider
    store: QdrantStore


class AlternateModelFakeEmbeddingProvider(DeterministicFakeEmbeddingProvider):
    """Same vector shape with a deliberately incompatible model identity."""

    model_identifier = "lexical-hash-test-v2"


def _memory_config() -> RetrievalConfig:
    return load_config(
        {
            "APP_ENV": "test",
            "EMBEDDING_PROVIDER": "fake",
            "OPENAI_API_KEY": OPENAI_SECRET,
            "OPENAI_EMBEDDING_DIMENSIONS": "192",
            "QDRANT_MODE": "memory",
            "QDRANT_URL": f"https://user:{URL_SECRET}@qdrant.invalid",
            "QDRANT_API_KEY": QDRANT_SECRET,
            "QDRANT_COLLECTION_NAME": "nexodocs_chunks_v1",
        }
    )


def _new_pipeline() -> MemoryPipeline:
    config = _memory_config()
    provider = DeterministicFakeEmbeddingProvider(config.embedding_dimensions)
    store = QdrantStore(create_client(config), config.collection_name, provider.dimensions)
    return MemoryPipeline(repository_root(), config, provider, store)


def _index(pipeline: MemoryPipeline) -> None:
    result = incremental_index(
        pipeline.root,
        pipeline.config,
        pipeline.provider,
        pipeline.store,
    )
    assert result.inserted_or_updated == 44
    assert result.reused == 0
    assert result.obsolete == 0
    assert result.total_points == 44


def test_memory_pipeline_indexes_reuses_searches_evaluates_and_checks_integrity(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Exercise the complete offline pipeline without network or persistent storage."""
    caplog.set_level(logging.INFO, logger="nexodocs_ai.retrieval.indexer")
    caplog.set_level(logging.INFO, logger="nexodocs_ai.retrieval.qdrant_store")
    pipeline = _new_pipeline()

    _index(pipeline)

    assert pipeline.store.client.collection_exists(pipeline.config.collection_name)
    assert pipeline.store.count() == 44
    second = incremental_index(
        pipeline.root,
        pipeline.config,
        pipeline.provider,
        pipeline.store,
    )
    assert second.inserted_or_updated == 0
    assert second.reused == 44
    assert second.obsolete == 0
    checked = check_index(
        pipeline.root,
        pipeline.config,
        pipeline.provider,
        pipeline.store,
    )
    assert checked.reused == checked.total_points == 44
    assert checked.obsolete == 0

    plan, payloads = build_plan(pipeline.root, pipeline.config, pipeline.provider)
    manifest = build_index_manifest(plan)
    assert manifest == build_index_manifest(plan)
    assert manifest.data["total_points"] == 44
    assert manifest.data["indexed_documents"] == plan.data["documents"]
    assert "timestamp" not in manifest.data
    assert "duration" not in manifest.data

    retriever = Retriever(
        pipeline.provider,
        pipeline.store,
        pipeline.config.top_k,
        pipeline.config.max_top_k,
        pipeline.config.max_per_document,
    )
    cancellation = retriever.retrieve("anteced\u00eancia cancelamento 24 horas", top_k=5)
    assert cancellation.status == "found"
    assert any(
        result.metadata["document_id"] == "NSI-POL-OPS-001" for result in cancellation.results
    )

    filtered = retriever.retrieve(
        "prazo de resposta da \u00e1rea",
        top_k=3,
        filters=RetrievalFilters(document_id="NSI-TAB-DIR-001"),
    )
    assert filtered.status == "found"
    assert filtered.applied_filters == {"document_id": "NSI-TAB-DIR-001"}
    assert all(result.metadata["document_id"] == "NSI-TAB-DIR-001" for result in filtered.results)

    diverse = retriever.retrieve(
        "pol\u00edtica administrativa \u00e1rea contato dados cobertura",
        top_k=5,
    )
    document_counts = Counter(str(result.metadata["document_id"]) for result in diverse.results)
    assert diverse.status == "found"
    assert document_counts
    assert max(document_counts.values()) <= pipeline.config.max_per_document

    no_evidence = retriever.retrieve(
        "astronomia gal\u00e1ctica telesc\u00f3pio orbital",
        top_k=5,
        score_threshold=1.0,
    )
    assert no_evidence.status == "no_evidence"
    assert no_evidence.results == ()
    assert no_evidence.reason == "no_match_or_below_threshold"
    with pytest.raises(RetrievalError, match="threshold"):
        retriever.retrieve("consulta", score_threshold=1.01)

    metrics = evaluate(retriever, pipeline.root / "evals" / "retrieval_cases.json")
    assert metrics.total_cases == 13
    assert metrics.hit_rate_at_k >= 0.6
    assert metrics.recall_at_k >= 0.6
    assert metrics.mrr >= 0.5
    assert metrics.fallback_precision >= 0.5
    assert metrics.passed_cases / metrics.total_cases >= 0.7

    log_output = caplog.text
    assert OPENAI_SECRET not in log_output
    assert QDRANT_SECRET not in log_output
    assert URL_SECRET not in log_output
    assert str(payloads[0]["text"]) not in log_output

    incompatible_provider = AlternateModelFakeEmbeddingProvider(pipeline.provider.dimensions)
    with pytest.raises(CollectionCompatibilityError, match="embedding identity"):
        incremental_index(
            pipeline.root,
            pipeline.config,
            incompatible_provider,
            pipeline.store,
        )
    assert (
        check_index(
            pipeline.root,
            pipeline.config,
            pipeline.provider,
            pipeline.store,
        ).reused
        == 44
    )

    corrupted_payload = dict(payloads[0])
    corrupted_payload["text_sha256"] = "0" * 64
    pipeline.store.upsert(
        [
            (
                plan.points[0].point_id,
                pipeline.provider.embed_query(str(corrupted_payload["text"])),
                corrupted_payload,
            )
        ]
    )
    with pytest.raises(RetrievalError, match="payload mismatch"):
        check_index(
            pipeline.root,
            pipeline.config,
            pipeline.provider,
            pipeline.store,
        )


def test_obsolete_point_is_reported_and_pruned_only_after_exact_confirmation() -> None:
    pipeline = _new_pipeline()
    _index(pipeline)
    plan, _ = build_plan(pipeline.root, pipeline.config, pipeline.provider)
    expected_ids = {planned.point_id for planned in plan.points}
    obsolete_id = point_id("nsi-chk-obsolete-integration-test")
    pipeline.store.upsert(
        [
            (
                obsolete_id,
                pipeline.provider.embed_query("obsolete integration point"),
                {"chunk_id": "nsi-chk-obsolete-integration-test"},
            )
        ]
    )

    assert pipeline.store.count() == 45
    assert pipeline.store.obsolete_ids(expected_ids) == {obsolete_id}
    assert (
        check_index(
            pipeline.root,
            pipeline.config,
            pipeline.provider,
            pipeline.store,
        ).obsolete
        == 1
    )

    with pytest.raises(ConfigurationError, match="nome exato"):
        pipeline.store.prune(expected_ids, "wrong-collection")
    assert pipeline.store.count() == 45

    assert pipeline.store.prune(expected_ids, pipeline.config.collection_name) == 1
    assert pipeline.store.count() == 44
    assert pipeline.store.obsolete_ids(expected_ids) == set()
    assert (
        check_index(
            pipeline.root,
            pipeline.config,
            pipeline.provider,
            pipeline.store,
        ).obsolete
        == 0
    )


def test_existing_collection_with_wrong_dimensions_is_rejected() -> None:
    pipeline = _new_pipeline()
    pipeline.store.client.create_collection(
        collection_name=pipeline.config.collection_name,
        vectors_config=VectorParams(
            size=pipeline.provider.dimensions + 1,
            distance=Distance.COSINE,
        ),
    )

    with pytest.raises(CollectionCompatibilityError, match="incompat"):
        pipeline.store.ensure_collection()

    assert pipeline.store.client.collection_exists(pipeline.config.collection_name)
    assert pipeline.store.count() == 0
