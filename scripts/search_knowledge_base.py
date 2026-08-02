"""Search an explicitly configured vector index and print structured JSON."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from _project_bootstrap import bootstrap_project


def main() -> int:
    """Run a manual structured search without accepting secrets on the CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--score-threshold", type=float)
    for name in (
        "document-id",
        "category",
        "source-format",
        "owner-area",
        "version",
        "classification",
    ):
        parser.add_argument(f"--{name}")
    args = parser.parse_args()
    bootstrap_project()
    from nexodocs_ai.retrieval.config import load_config
    from nexodocs_ai.retrieval.embeddings import create_embedding_provider
    from nexodocs_ai.retrieval.models import RetrievalFilters
    from nexodocs_ai.retrieval.qdrant_store import QdrantStore, create_client
    from nexodocs_ai.retrieval.retriever import Retriever

    config = load_config()
    provider = create_embedding_provider(config)
    store = QdrantStore(create_client(config), config.collection_name, provider.dimensions)
    filters = RetrievalFilters(
        document_id=args.document_id,
        category=args.category,
        source_format=args.source_format,
        owner_area=args.owner_area,
        version=args.version,
        classification=args.classification,
    )
    response = Retriever(
        provider,
        store,
        config.top_k,
        config.max_top_k,
        config.max_per_document,
    ).retrieve(
        args.query,
        args.top_k,
        filters,
        args.score_threshold if args.score_threshold is not None else config.score_threshold,
    )
    print(json.dumps(asdict(response), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
