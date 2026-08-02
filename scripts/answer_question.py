"""Manual answer CLI; it never accepts credentials or endpoint settings."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

from _project_bootstrap import bootstrap_project


def build_parser() -> argparse.ArgumentParser:
    """Build the answer CLI with opt-in safe usage reporting."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--usage-report")
    parser.add_argument("--overwrite-usage-report", action="store_true")
    parser.add_argument("--privacy-safe-usage-report", action="store_true")
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


def validate_usage_report_options(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Reject privacy-safe reporting unless a report destination was explicitly requested."""
    if args.privacy_safe_usage_report and not args.usage_report:
        parser.error("--privacy-safe-usage-report requires --usage-report")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_usage_report_options(parser, args)
    bootstrap_project()
    from nexodocs_ai.rag.answer_provider import create_answer_provider
    from nexodocs_ai.rag.config import load_rag_config
    from nexodocs_ai.rag.models import RagRequest
    from nexodocs_ai.rag.pipeline import RagPipeline
    from nexodocs_ai.retrieval.config import load_config
    from nexodocs_ai.retrieval.embeddings import create_embedding_provider
    from nexodocs_ai.retrieval.models import RetrievalFilters
    from nexodocs_ai.retrieval.qdrant_store import QdrantStore, create_client
    from nexodocs_ai.retrieval.retriever import Retriever

    retrieval, rag = load_config(), load_rag_config()
    provider = create_embedding_provider(retrieval)
    store = QdrantStore(create_client(retrieval), retrieval.collection_name, provider.dimensions)
    store.validate_collection()
    filters = RetrievalFilters(
        args.document_id,
        args.category,
        args.source_format,
        args.owner_area,
        args.version,
        args.classification,
    )
    pipeline = RagPipeline(
        Retriever(
            provider, store, retrieval.top_k, retrieval.max_top_k, retrieval.max_per_document
        ),
        create_answer_provider(rag),
        rag,
    )
    request = RagRequest(
        args.query, args.top_k, args.threshold, filters, include_debug_metadata=args.debug
    )
    started = time.perf_counter()
    if args.usage_report:
        run = pipeline.answer_with_usage(request)
        response = run.response
        from nexodocs_ai.observability.reports import (
            answer_report,
            privacy_safe_report,
            write_report,
        )

        report = answer_report(
            run,
            pipeline.provider.provider_name,
            pipeline.provider.model_identifier,
            retrieval.collection_name,
            len(args.query),
            int((time.perf_counter() - started) * 1000),
        )
        write_report(
            bootstrap_project(),
            args.usage_report,
            privacy_safe_report(report) if args.privacy_safe_usage_report else report,
            overwrite=args.overwrite_usage_report,
        )
    else:
        response = pipeline.answer(request)
    data = asdict(response)
    print(
        json.dumps(data, ensure_ascii=False, sort_keys=True)
        if args.json
        else (response.answer or response.message)
    )
    return 0 if response.status in {"answered", "no_evidence"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
