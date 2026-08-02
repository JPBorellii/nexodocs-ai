"""Stable constants for offline-safe vector retrieval."""

from uuid import UUID

SCHEMA_VERSION = "1.0"
PLAN_VERSION = "1.0.0"
INDEX_VERSION = "1.0.0"
DEFAULT_COLLECTION_NAME = "nexodocs_chunks_v1"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSIONS = 1536
DEFAULT_OPENAI_MAX_RETRIES = 0
DEFAULT_FAKE_DIMENSIONS = 192
DEFAULT_TOP_K = 5
DEFAULT_MAX_TOP_K = 20
DEFAULT_MAX_PER_DOCUMENT = 2
MAX_QUERY_CHARACTERS = 4_000
DISTANCE_NAME = "Cosine"
POINT_ID_STRATEGY = "uuid5-nexodocs-chunk-id"
NEXODOCS_CHUNK_NAMESPACE = UUID("04c8cd10-5735-5c9e-9e28-5f5f9d9d0355")
INDEX_PLAN_FILENAME = "index-plan.json"
INDEX_MANIFEST_FILENAME = "index-manifest.json"
ALLOWED_FILTERS = frozenset(
    {"document_id", "category", "source_format", "owner_area", "version", "classification"}
)
