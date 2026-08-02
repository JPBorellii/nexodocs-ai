"""Tests for deterministic vector index planning."""

from __future__ import annotations

import hashlib
from uuid import UUID, uuid5

from nexodocs_ai.knowledge_base.generator import repository_root
from nexodocs_ai.retrieval.config import load_config
from nexodocs_ai.retrieval.constants import NEXODOCS_CHUNK_NAMESPACE
from nexodocs_ai.retrieval.embeddings import DeterministicFakeEmbeddingProvider
from nexodocs_ai.retrieval.indexer import (
    build_index_manifest,
    build_payload,
    build_plan,
    deterministic_json,
    point_id,
)
from nexodocs_ai.retrieval.models import EmbeddingSpecification


def _test_config_values() -> dict[str, str]:
    return {
        "APP_ENV": "test",
        "EMBEDDING_PROVIDER": "fake",
        "OPENAI_EMBEDDING_DIMENSIONS": "64",
        "QDRANT_MODE": "memory",
    }


def test_point_id_is_stable_uuid5_in_the_project_namespace() -> None:
    chunk_id = "nsi-chk-0123456789abcdef01234567"

    generated = point_id(chunk_id)

    assert generated == str(uuid5(NEXODOCS_CHUNK_NAMESPACE, chunk_id))
    assert UUID(generated).version == 5
    assert point_id(chunk_id) == generated
    assert point_id(f"{chunk_id}-other") != generated


def test_payload_is_closed_and_omits_non_applicable_fields() -> None:
    provider = DeterministicFakeEmbeddingProvider(64)
    chunk: dict[str, object] = {
        "schema_version": "1.0",
        "chunk_id": "nsi-chk-0123456789abcdef01234567",
        "document_id": "NSI-POL-OPS-001",
        "source_filename": "politica_ficticia.pdf",
        "source_format": "pdf",
        "source_sha256": "a" * 64,
        "title": "Pol\u00edtica fict\u00edcia",
        "category": "operations",
        "version": "1.0",
        "effective_date": "2026-01-01",
        "owner_area": "Opera\u00e7\u00f5es",
        "owner_contact": "operacoes@nexosaude.example",
        "language": "pt-BR",
        "classification": "internal",
        "fictitious_notice": "Conte\u00fado totalmente fict\u00edcio.",
        "locator": "p\u00e1gina 2",
        "chunk_index": 0,
        "text": "Cancelamento fict\u00edcio.",
        "text_sha256": "b" * 64,
        "char_count": 24,
        "word_count": 2,
        "page_number": 2,
        "section_title": "Cancelamento",
        "row_number": None,
        "unexpected": "must-not-be-indexed",
    }

    payload = build_payload(chunk, "c" * 64, provider)

    assert payload["chunk_id"] == chunk["chunk_id"]
    assert payload["processing_manifest_sha256"] == "c" * 64
    assert payload["embedding_provider"] == "deterministic-fake"
    assert payload["embedding_model"] == "lexical-hash-test-v1"
    assert payload["embedding_dimensions"] == 64
    assert payload["page_number"] == 2
    assert "unexpected" not in payload
    assert "row_number" not in payload


def test_deterministic_json_is_independent_of_mapping_insertion_order() -> None:
    left = deterministic_json({"b": 2, "a": 1})
    right = deterministic_json({"a": 1, "b": 2})

    assert left == right == b'{"a":1,"b":2}\n'


def test_build_plan_is_deterministic_for_all_processed_chunks() -> None:
    config = load_config(_test_config_values())
    provider = DeterministicFakeEmbeddingProvider(config.embedding_dimensions)
    root = repository_root()

    first, first_payloads = build_plan(root, config, provider)
    second, second_payloads = build_plan(root, config, provider)

    assert first.data == second.data
    assert first.points == second.points
    assert first_payloads == second_payloads
    assert first.data["total_points"] == 44
    assert len(first.points) == len(first_payloads) == 44
    assert len({planned.chunk_id for planned in first.points}) == 44
    assert len({planned.point_id for planned in first.points}) == 44
    assert all(UUID(planned.point_id).version == 5 for planned in first.points)
    expected_payload_hash = hashlib.sha256(deterministic_json(first_payloads[0])).hexdigest()
    assert first.points[0].payload_sha256 == expected_payload_hash


def test_future_openai_manifest_is_official_and_has_no_operational_usage() -> None:
    config = load_config(
        {
            "EMBEDDING_PROVIDER": "openai",
            "OPENAI_EMBEDDING_MODEL": "text-embedding-3-small",
            "OPENAI_EMBEDDING_DIMENSIONS": "1536",
            "QDRANT_MODE": "local",
        }
    )
    provider = EmbeddingSpecification("openai", config.embedding_model, config.embedding_dimensions)
    plan, _ = build_plan(repository_root(), config, provider)
    manifest = build_index_manifest(plan)

    assert manifest.data["embedding_model"] == "text-embedding-3-small"
    assert manifest.data["embedding_dimensions"] == 1536
    assert manifest == build_index_manifest(plan)
    assert not {"timestamp", "tokens", "request_id", "telemetry"} & manifest.data.keys()
