"""Manual answer CLI; it never accepts credentials or endpoint settings."""

from __future__ import annotations

import argparse
import json
import time

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
    parser.add_argument("--sanitized-grounding-diagnostic-report")
    parser.add_argument(
        "--sanitized-grounding-diagnostic-case-id", choices=("HOLD-P02", "HOLD-P04")
    )
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
    diagnostic = args.sanitized_grounding_diagnostic_report
    case_id = args.sanitized_grounding_diagnostic_case_id
    if diagnostic and not case_id:
        parser.error(
            "--sanitized-grounding-diagnostic-report requires "
            "--sanitized-grounding-diagnostic-case-id"
        )
    if case_id and not diagnostic:
        parser.error(
            "--sanitized-grounding-diagnostic-case-id requires "
            "--sanitized-grounding-diagnostic-report"
        )
    if diagnostic and (not args.usage_report or not args.privacy_safe_usage_report):
        parser.error(
            "--sanitized-grounding-diagnostic-report requires --usage-report and "
            "--privacy-safe-usage-report"
        )
    if diagnostic and args.overwrite_usage_report:
        parser.error("D04 does not permit --overwrite-usage-report")


def sanitized_diagnostic_failure() -> int:
    """Emit the closed operational failure without candidate paths or content."""
    print(
        json.dumps(
            {
                "safe_error_code": "grounding_diagnostic_report_unavailable",
                "status": "diagnostic_failed",
            },
            sort_keys=True,
        )
    )
    return 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_usage_report_options(parser, args)
    root = bootstrap_project()
    from nexodocs_ai.rag.answer_provider import create_answer_provider
    from nexodocs_ai.rag.config import load_rag_config
    from nexodocs_ai.rag.models import RagRequest
    from nexodocs_ai.rag.pipeline import RagPipeline
    from nexodocs_ai.rag.serialization import (
        cli_exit_code,
        cli_response_data,
        privacy_safe_report_required,
    )
    from nexodocs_ai.retrieval.config import load_config
    from nexodocs_ai.retrieval.embeddings import create_embedding_provider
    from nexodocs_ai.retrieval.models import RetrievalFilters
    from nexodocs_ai.retrieval.qdrant_store import QdrantStore, create_client
    from nexodocs_ai.retrieval.retriever import Retriever

    diagnostic_requested = args.sanitized_grounding_diagnostic_report is not None
    if diagnostic_requested:
        from nexodocs_ai.observability.grounding_diagnostic_d04 import (
            D04ArtifactError,
            ensure_d04_destination_available,
        )

        try:
            ensure_d04_destination_available(root, args.sanitized_grounding_diagnostic_report)
        except D04ArtifactError:
            return sanitized_diagnostic_failure()

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
        run = pipeline.answer_with_usage(
            request, sanitized_grounding_diagnostic=diagnostic_requested
        )
        response = run.response
        from nexodocs_ai.observability.reports import (
            ReportError,
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
        try:
            usage_path = write_report(
                root,
                args.usage_report,
                privacy_safe_report(report)
                if args.privacy_safe_usage_report or privacy_safe_report_required(response)
                else report,
                overwrite=args.overwrite_usage_report,
            )
        except OSError, ReportError:
            if diagnostic_requested:
                return sanitized_diagnostic_failure()
            raise
        diagnostic_applicable = (
            response.status == "grounding_failed"
            and response.reason_code == "grounding_quote_not_in_evidence"
        )
        if diagnostic_requested and not diagnostic_applicable:
            return sanitized_diagnostic_failure()
        if diagnostic_requested and diagnostic_applicable:
            from nexodocs_ai.observability.grounding_diagnostic_d04 import (
                D04ArtifactError,
                build_d04_artifact,
                write_d04_artifact,
            )

            try:
                artifact = build_d04_artifact(
                    root,
                    args.sanitized_grounding_diagnostic_case_id,
                    run,
                    usage_path,
                )
                write_d04_artifact(root, args.sanitized_grounding_diagnostic_report, artifact)
            except D04ArtifactError:
                return sanitized_diagnostic_failure()
    else:
        response = pipeline.answer(request)
    data = cli_response_data(response)
    print(
        json.dumps(data, ensure_ascii=False, sort_keys=True)
        if args.json or response.status == "grounding_failed"
        else (response.answer or response.message)
    )
    return cli_exit_code(response)


if __name__ == "__main__":
    raise SystemExit(main())
