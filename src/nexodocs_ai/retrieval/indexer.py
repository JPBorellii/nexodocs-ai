"""Deterministic index planning and safe incremental indexing."""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast
from uuid import uuid5

from jsonschema import Draft202012Validator

from nexodocs_ai.knowledge_base.generator import repository_root
from nexodocs_ai.processing.validator import load_chunks, validate_processed

from .constants import (
    DISTANCE_NAME,
    INDEX_MANIFEST_FILENAME,
    INDEX_PLAN_FILENAME,
    INDEX_VERSION,
    NEXODOCS_CHUNK_NAMESPACE,
    PLAN_VERSION,
    POINT_ID_STRATEGY,
    SCHEMA_VERSION,
)
from .models import (
    CollectionCompatibilityError,
    EmbeddingIdentity,
    EmbeddingProvider,
    IndexingResult,
    IndexManifest,
    IndexPlan,
    PlannedPoint,
    RetrievalConfig,
    RetrievalError,
)
from .qdrant_store import QdrantStore

LOGGER = logging.getLogger(__name__)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def point_id(chunk_id: str) -> str:
    """Return stable UUIDv5 identity for a stable chunk identity."""
    return str(uuid5(NEXODOCS_CHUNK_NAMESPACE, chunk_id))


def deterministic_json(value: object, indent: int | None = None) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
            indent=indent,
        )
        + "\n"
    ).encode("utf-8")


def build_payload(
    chunk: dict[str, object], source_manifest_sha256: str, provider: EmbeddingIdentity
) -> dict[str, object]:
    """Build the closed payload contract without environment-specific fields."""
    keys = {
        "schema_version",
        "chunk_id",
        "document_id",
        "source_filename",
        "source_format",
        "source_sha256",
        "title",
        "category",
        "version",
        "effective_date",
        "owner_area",
        "owner_contact",
        "language",
        "classification",
        "fictitious_notice",
        "locator",
        "chunk_index",
        "text",
        "text_sha256",
        "char_count",
        "word_count",
        "page_number",
        "section_title",
        "row_number",
        "row_key",
    }
    payload = {key: chunk[key] for key in keys if key in chunk and chunk[key] is not None}
    payload.update(
        {
            "processing_manifest_sha256": source_manifest_sha256,
            "embedding_provider": provider.provider_name,
            "embedding_model": provider.model_identifier,
            "embedding_dimensions": provider.dimensions,
            "indexed_schema_version": SCHEMA_VERSION,
        }
    )
    return payload


def build_plan(
    root: Path | None, config: RetrievalConfig, provider: EmbeddingIdentity
) -> tuple[IndexPlan, list[dict[str, object]]]:
    """Build a deterministic, offline plan solely from validated processed chunks."""
    project = root or repository_root()
    validate_processed(project)
    processed = project / "knowledge_base" / "processed"
    chunks, manifest_path = load_chunks(processed / "chunks.jsonl"), processed / "manifest.json"
    source_manifest_sha256 = _sha256(manifest_path.read_bytes())
    points: list[PlannedPoint] = []
    payloads: list[dict[str, object]] = []
    for chunk in chunks:
        payload = build_payload(chunk, source_manifest_sha256, provider)
        planned = PlannedPoint(
            str(chunk["chunk_id"]),
            point_id(str(chunk["chunk_id"])),
            str(chunk["document_id"]),
            str(chunk["text_sha256"]),
            _sha256(deterministic_json(payload)),
        )
        points.append(planned)
        payloads.append(payload)
    if len({item.chunk_id for item in points}) != len(points) or len(
        {item.point_id for item in points}
    ) != len(points):
        raise RetrievalError("Colisão de chunk_id ou point_id")
    documents = [
        {
            "document_id": document_id,
            "chunk_count": sum(point.document_id == document_id for point in points),
            "source_sha256": next(
                str(chunk["source_sha256"])
                for chunk in chunks
                if chunk["document_id"] == document_id
            ),
        }
        for document_id in sorted({point.document_id for point in points})
    ]
    data: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "plan_version": PLAN_VERSION,
        "collection_name": config.collection_name,
        "distance": DISTANCE_NAME,
        "vector_size": provider.dimensions,
        "embedding_provider": provider.provider_name,
        "embedding_model": provider.model_identifier,
        "embedding_dimensions": provider.dimensions,
        "source_chunks_sha256": _sha256((processed / "chunks.jsonl").read_bytes()),
        "source_manifest_sha256": source_manifest_sha256,
        "total_points": len(points),
        "chunk_ids_sha256": _sha256("\n".join(point.chunk_id for point in points).encode()),
        "payload_schema_version": SCHEMA_VERSION,
        "point_id_strategy": POINT_ID_STRATEGY,
        "documents": documents,
        "points": [asdict(item) for item in points],
    }
    return IndexPlan(data, tuple(points)), payloads


def write_plan(
    root: Path | None, config: RetrievalConfig, provider: EmbeddingIdentity
) -> IndexPlan:
    """Write the checked-in plan only when deterministic bytes differ."""
    project = root or repository_root()
    plan, _ = build_plan(project, config, provider)
    path = project / "knowledge_base" / "index" / INDEX_PLAN_FILENAME
    content = deterministic_json(plan.data, indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_bytes() != content:
        path.write_bytes(content)
    return plan


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrievalError(f"Invalid JSON artifact: {path.name}") from exc
    if not isinstance(value, dict):
        raise RetrievalError(f"JSON artifact must be an object: {path.name}")
    return cast(dict[str, object], value)


def _validate_schema(project: Path, filename: str, value: dict[str, object]) -> None:
    schema = _load_json_object(project / "knowledge_base" / "index" / filename)
    validator = Draft202012Validator(cast(Any, schema))
    errors = sorted(
        validator.iter_errors(cast(Any, value)),  # pyright: ignore[reportUnknownMemberType] - jsonschema is a dynamic boundary.
        key=lambda item: list(item.path),
    )
    if errors:
        raise RetrievalError(f"Invalid {filename}: {errors[0].message}")


def validate_index_plan(
    root: Path | None, config: RetrievalConfig, provider: EmbeddingIdentity
) -> IndexPlan:
    """Validate schema, hashes, UUIDs, payload identities, and all processed chunks."""
    project = root or repository_root()
    path = project / "knowledge_base" / "index" / INDEX_PLAN_FILENAME
    if not path.is_file():
        raise RetrievalError("index-plan.json is missing")
    actual = _load_json_object(path)
    _validate_schema(project, "index-plan.schema.json", actual)
    expected, _ = build_plan(project, config, provider)
    if actual != expected.data:
        raise RetrievalError("index-plan.json hashes or identities are inconsistent")
    if len(expected.points) != 44:
        raise RetrievalError("Index plan must contain exactly 44 points")
    if len({point.point_id for point in expected.points}) != len(expected.points):
        raise RetrievalError("Index plan contains duplicate point IDs")
    return expected


def check_plan(root: Path | None, config: RetrievalConfig, provider: EmbeddingIdentity) -> None:
    """Regenerate to temporary storage and compare the plan byte-for-byte."""
    project = root or repository_root()
    plan = validate_index_plan(project, config, provider)
    with tempfile.TemporaryDirectory() as temporary:
        staged = Path(temporary) / INDEX_PLAN_FILENAME
        staged.write_bytes(deterministic_json(plan.data, indent=2))
        path = project / "knowledge_base" / "index" / INDEX_PLAN_FILENAME
        if staged.read_bytes() != path.read_bytes():
            raise RetrievalError("index-plan.json fora de sincronização")


def build_index_manifest(plan: IndexPlan) -> IndexManifest:
    """Build deterministic metadata for a successfully materialized index."""
    source = plan.data
    data: dict[str, object] = {
        "schema_version": source["schema_version"],
        "index_version": INDEX_VERSION,
        "collection_name": source["collection_name"],
        "distance": source["distance"],
        "vector_size": source["vector_size"],
        "embedding_provider": source["embedding_provider"],
        "embedding_model": source["embedding_model"],
        "embedding_dimensions": source["embedding_dimensions"],
        "source_chunks_sha256": source["source_chunks_sha256"],
        "source_manifest_sha256": source["source_manifest_sha256"],
        "total_points": source["total_points"],
        "payload_schema_version": source["payload_schema_version"],
        "point_id_strategy": source["point_id_strategy"],
        "chunk_ids_sha256": source["chunk_ids_sha256"],
        "indexed_documents": source["documents"],
    }
    return IndexManifest(data)


def write_index_manifest(project: Path, manifest: IndexManifest) -> Path:
    """Write the real index manifest only after all online validation succeeds."""
    _validate_schema(project, "index-manifest.schema.json", manifest.data)
    path = project / "knowledge_base" / "index" / INDEX_MANIFEST_FILENAME
    content = deterministic_json(manifest.data, indent=2)
    if not path.exists() or path.read_bytes() != content:
        path.write_bytes(content)
    return path


def incremental_index(
    root: Path | None, config: RetrievalConfig, provider: EmbeddingProvider, store: QdrantStore
) -> IndexingResult:
    """Upsert only absent or incompatible points; never prune implicitly."""
    plan, payloads = build_plan(root, config, provider)
    store.ensure_collection()
    identity_keys = ("embedding_provider", "embedding_model", "embedding_dimensions")
    expected_identity = {
        "embedding_provider": provider.provider_name,
        "embedding_model": provider.model_identifier,
        "embedding_dimensions": provider.dimensions,
    }
    if any(
        payload.get(key) is not None and payload.get(key) != expected_identity[key]
        for record in store.all_records()
        for payload in [record.payload]
        for key in identity_keys
    ):
        raise CollectionCompatibilityError(
            "Collection contains an incompatible embedding identity; use a new collection version"
        )
    existing: dict[str, dict[str, object]] = {
        point.point_id: point.payload
        for point in store.retrieve_batched(point.point_id for point in plan.points)
    }
    required = (
        "chunk_id",
        "text_sha256",
        "embedding_provider",
        "embedding_model",
        "embedding_dimensions",
        "indexed_schema_version",
        "processing_manifest_sha256",
    )
    incompatible = [
        point.point_id
        for point, payload in zip(plan.points, payloads)
        if point.point_id in existing
        and any(existing[point.point_id].get(key) != payload[key] for key in identity_keys)
    ]
    if incompatible:
        raise CollectionCompatibilityError(
            "Existing points use an incompatible embedding identity; use a new collection version"
        )
    pending = [
        (point, payload)
        for point, payload in zip(plan.points, payloads)
        if any(existing.get(point.point_id, {}).get(key) != payload[key] for key in required)
    ]
    vectors = (
        provider.embed_documents([str(payload["text"]) for _, payload in pending])
        if pending
        else []
    )
    if pending:
        store.upsert(
            (point.point_id, vector, payload) for (point, payload), vector in zip(pending, vectors)
        )
    expected_ids = {point.point_id for point in plan.points}
    obsolete = len(store.obsolete_ids(expected_ids))
    if store.count() != len(plan.points) + obsolete:
        raise RetrievalError("Qdrant point count is inconsistent after upsert")
    check_index(root, config, provider, store)
    result = IndexingResult(
        len(pending), len(plan.points) - len(pending), obsolete, len(plan.points)
    )
    LOGGER.info(
        "Indexed vector points collection=%s provider=%s model=%s dimensions=%d updated=%d reused=%d obsolete=%d",
        config.collection_name,
        provider.provider_name,
        provider.model_identifier,
        provider.dimensions,
        result.inserted_or_updated,
        result.reused,
        result.obsolete,
    )
    return result


def check_index(
    root: Path | None,
    config: RetrievalConfig,
    provider: EmbeddingIdentity,
    store: QdrantStore,
) -> IndexingResult:
    """Validate collection configuration, expected payloads, IDs, and counts without writes."""
    plan, payloads = build_plan(root, config, provider)
    store.validate_collection()
    records = {
        record.point_id: record.payload
        for record in store.retrieve_batched(point.point_id for point in plan.points)
    }
    for point, payload in zip(plan.points, payloads):
        if records.get(point.point_id) != payload:
            raise RetrievalError(f"Qdrant payload mismatch for chunk {point.chunk_id}")
    obsolete = len(store.obsolete_ids({point.point_id for point in plan.points}))
    if store.count() != len(plan.points) + obsolete:
        raise RetrievalError("Qdrant collection count is inconsistent")
    return IndexingResult(0, len(plan.points), obsolete, len(plan.points))
