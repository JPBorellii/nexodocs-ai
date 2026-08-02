"""Run the offline lexical retrieval evaluation in memory."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from _project_bootstrap import bootstrap_project


def main() -> int:
    """Run deterministic fake retrieval and enforce its documented minimums."""
    parser = argparse.ArgumentParser()
    parser.parse_args()
    root = bootstrap_project()
    from nexodocs_ai.retrieval.config import load_config
    from nexodocs_ai.retrieval.embeddings import create_embedding_provider
    from nexodocs_ai.retrieval.evaluation import evaluate
    from nexodocs_ai.retrieval.indexer import incremental_index
    from nexodocs_ai.retrieval.qdrant_store import QdrantStore, create_client
    from nexodocs_ai.retrieval.retriever import Retriever

    config = load_config(
        {
            "APP_ENV": "test",
            "EMBEDDING_PROVIDER": "fake",
            "OPENAI_EMBEDDING_DIMENSIONS": "192",
            "QDRANT_MODE": "memory",
        }
    )
    provider = create_embedding_provider(config)
    store = QdrantStore(create_client(config), config.collection_name, provider.dimensions)
    incremental_index(None, config, provider, store)
    metrics = evaluate(Retriever(provider, store), root / "evals" / "retrieval_cases.json")
    print(json.dumps(asdict(metrics), sort_keys=True))
    minimums_met = (
        metrics.hit_rate_at_k >= 0.6
        and metrics.recall_at_k >= 0.6
        and metrics.mrr >= 0.5
        and metrics.fallback_precision >= 0.5
        and metrics.passed_cases / metrics.total_cases >= 0.7
    )
    return 0 if minimums_met else 1


if __name__ == "__main__":
    raise SystemExit(main())
