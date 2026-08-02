"""Direct, guarded Qdrant access."""

from __future__ import annotations

import logging
from collections.abc import Iterable

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Condition,
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

from .constants import DISTANCE_NAME
from .models import (
    CollectionCompatibilityError,
    ConfigurationError,
    RetrievalConfig,
    StoredPoint,
    StoredSearchResult,
)

LOGGER = logging.getLogger(__name__)


def create_client(config: RetrievalConfig) -> QdrantClient:
    """Create a client only for the explicitly configured mode."""
    if config.qdrant_mode == "memory":
        if config.app_env != "test":
            raise ConfigurationError("Qdrant memory é permitido somente em teste")
        return QdrantClient(":memory:")
    if config.qdrant_mode == "local":
        return QdrantClient(path=config.qdrant_path)
    if config.qdrant_mode == "remote":
        if not config.qdrant_url:
            raise ConfigurationError("QDRANT_URL é obrigatório")
        return QdrantClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key or None,
            timeout=config.qdrant_timeout_seconds,
        )
    raise ConfigurationError("Modo Qdrant inválido")


class QdrantStore:
    """Collection lifecycle, points, and searches without destructive defaults."""

    def __init__(self, client: QdrantClient, collection_name: str, dimensions: int) -> None:
        self.client, self.collection_name, self.dimensions = client, collection_name, dimensions

    def ensure_collection(self) -> None:
        """Create once, otherwise reject incompatible existing configuration."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                self.collection_name,
                vectors_config=VectorParams(size=self.dimensions, distance=Distance.COSINE),
            )
            LOGGER.info(
                "Created vector collection name=%s dimensions=%d distance=%s",
                self.collection_name,
                self.dimensions,
                DISTANCE_NAME,
            )
            return
        self.validate_collection()

    def validate_collection(self) -> None:
        """Validate an existing collection without creating or mutating it."""
        if not self.client.collection_exists(self.collection_name):
            raise CollectionCompatibilityError(f"Collection does not exist: {self.collection_name}")
        info = self.client.get_collection(self.collection_name)
        params = info.config.params.vectors
        if (
            not isinstance(params, VectorParams)
            or params.size != self.dimensions
            or params.distance != Distance.COSINE
        ):
            raise CollectionCompatibilityError(
                f"Coleção {self.collection_name} incompatível; esperado {self.dimensions}/{DISTANCE_NAME}"
            )

    def retrieve(self, point_ids: Iterable[str]) -> list[StoredPoint]:
        records = self.client.retrieve(
            self.collection_name, ids=list(point_ids), with_payload=True, with_vectors=False
        )
        return [StoredPoint(str(record.id), dict(record.payload or {})) for record in records]

    def retrieve_batched(
        self, point_ids: Iterable[str], batch_size: int = 128
    ) -> list[StoredPoint]:
        """Retrieve known IDs in bounded deterministic batches."""
        identifiers = list(point_ids)
        records: list[StoredPoint] = []
        for start in range(0, len(identifiers), batch_size):
            records.extend(self.retrieve(identifiers[start : start + batch_size]))
        return records

    def upsert(self, points: Iterable[tuple[str, list[float], dict[str, object]]]) -> None:
        self.client.upsert(
            self.collection_name,
            points=[
                PointStruct(id=point_id, vector=vector, payload=payload)
                for point_id, vector, payload in points
            ],
            wait=True,
        )

    def search(
        self, vector: list[float], limit: int, filters: dict[str, str]
    ) -> list[StoredSearchResult]:
        conditions: list[Condition] = []
        for key, value in filters.items():
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
        points: list[ScoredPoint] = self.client.query_points(
            self.collection_name,
            query=vector,
            query_filter=Filter(must=conditions) if conditions else None,
            limit=limit,
            with_payload=True,
        ).points
        return [
            StoredSearchResult(float(point.score), dict(point.payload or {})) for point in points
        ]

    def obsolete_ids(self, expected_ids: set[str]) -> set[str]:
        return {record.point_id for record in self.all_records(with_payload=False)} - expected_ids

    def all_records(self, *, with_payload: bool = True) -> list[StoredPoint]:
        """Read every point with Qdrant pagination."""
        records: list[StoredPoint] = []
        offset = None
        while True:
            page, offset = self.client.scroll(
                self.collection_name,
                limit=256,
                offset=offset,
                with_payload=with_payload,
                with_vectors=False,
            )
            records.extend(
                StoredPoint(str(record.id), dict(record.payload or {})) for record in page
            )
            if offset is None:
                return records

    def count(self) -> int:
        """Return the exact collection point count."""
        return self.client.count(self.collection_name, exact=True).count

    def prune(self, expected_ids: set[str], confirmation_collection: str) -> int:
        """Delete only obsolete points after exact-name confirmation."""
        if confirmation_collection != self.collection_name:
            raise ConfigurationError("Confirmação deve conter o nome exato da coleção")
        obsolete = self.obsolete_ids(expected_ids)
        if obsolete:
            self.client.delete(self.collection_name, points_selector=list(obsolete), wait=True)
            LOGGER.info(
                "Pruned obsolete vector points collection=%s count=%d",
                self.collection_name,
                len(obsolete),
            )
        return len(obsolete)
