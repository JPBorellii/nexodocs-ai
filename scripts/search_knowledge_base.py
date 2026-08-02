"""Search an explicitly configured vector index and print structured JSON."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

from _project_bootstrap import bootstrap_project


def build_parser() -> argparse.ArgumentParser:
    """Build the search CLI with optional safe local reporting."""
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--score-threshold", type=float)
    parser.add_argument("--usage-report")
    parser.add_argument("--overwrite-usage-report", action="store_true")
    for name in (
        "document-id",
        "category",
        "source-format",
        "owner-area",
        "version",
        "classification",
    ):
        parser.add_argument(f"--{name}")
    return parser


def main() -> int:
    """Run a manual structured search without accepting secrets on the CLI."""
    parser = build_parser()
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
    retriever = Retriever(
        provider,
        store,
        config.top_k,
        config.max_top_k,
        config.max_per_document,
    )
    started = time.perf_counter()
    threshold = args.score_threshold if args.score_threshold is not None else config.score_threshold
    if args.usage_report:
        operation = retriever.retrieve_with_usage(args.query, args.top_k, filters, threshold)
        response = operation.response
        from nexodocs_ai.observability.reports import search_report, write_report

        write_report(
            bootstrap_project(),
            args.usage_report,
            search_report(
                operation.embedding_usage,
                config.collection_name,
                len(args.query),
                response.status,
                int((time.perf_counter() - started) * 1000),
            ),
            overwrite=args.overwrite_usage_report,
        )
    else:
        response = retriever.retrieve(args.query, args.top_k, filters, threshold)
    print(json.dumps(asdict(response), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
