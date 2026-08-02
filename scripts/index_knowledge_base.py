"""Create/check deterministic plans or perform explicitly manual indexing."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from _project_bootstrap import bootstrap_project


def main() -> int:
    """Run an offline plan operation or an explicitly configured index operation."""
    bootstrap_project()
    from nexodocs_ai.retrieval.config import load_config, load_plan_config
    from nexodocs_ai.retrieval.embeddings import create_embedding_provider
    from nexodocs_ai.retrieval.indexer import (
        build_index_manifest,
        build_plan,
        check_index,
        check_plan,
        incremental_index,
        validate_index_plan,
        write_index_manifest,
        write_plan,
    )
    from nexodocs_ai.retrieval.models import EmbeddingSpecification, RetrievalError
    from nexodocs_ai.retrieval.qdrant_store import QdrantStore, create_client

    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="command", required=True)
    plan = subs.add_parser("plan")
    mode = plan.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    subs.add_parser("write")
    subs.add_parser("check-index")
    prune = subs.add_parser("prune")
    prune.add_argument("--confirm-collection", required=True)
    args = parser.parse_args()
    try:
        if args.command == "plan":
            config = load_plan_config()
            specification = EmbeddingSpecification(
                "openai", config.embedding_model, config.embedding_dimensions
            )
            if args.write:
                write_plan(None, config, specification)
            else:
                check_plan(None, config, specification)
            return 0
        config = load_config()
        provider = create_embedding_provider(config)
        validate_index_plan(None, config, provider)
        store = QdrantStore(create_client(config), config.collection_name, provider.dimensions)
        if args.command == "write":
            result = incremental_index(None, config, provider, store)
            plan_result, _ = build_plan(None, config, provider)
            root = bootstrap_project()
            write_index_manifest(root, build_index_manifest(plan_result))
            print(json.dumps(asdict(result), sort_keys=True))
        elif args.command == "check-index":
            print(json.dumps(asdict(check_index(None, config, provider, store)), sort_keys=True))
        else:
            plan_result, _ = build_plan(None, config, provider)
            removed = store.prune(
                {point.point_id for point in plan_result.points}, args.confirm_collection
            )
            print(json.dumps({"removed": removed}, sort_keys=True))
        return 0
    except RetrievalError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
