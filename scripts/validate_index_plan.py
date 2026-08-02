"""Offline validation of the checked-in deterministic index plan."""

from __future__ import annotations

import argparse

from _project_bootstrap import bootstrap_project


def main() -> int:
    """Validate the deterministic plan without network access or writes."""
    parser = argparse.ArgumentParser()
    parser.parse_args()
    bootstrap_project()
    from nexodocs_ai.retrieval.config import load_plan_config
    from nexodocs_ai.retrieval.indexer import validate_index_plan
    from nexodocs_ai.retrieval.models import EmbeddingSpecification

    config = load_plan_config()
    specification = EmbeddingSpecification(
        "openai", config.embedding_model, config.embedding_dimensions
    )
    plan = validate_index_plan(None, config, specification)
    print(f"Index plan validation passed: {len(plan.points)} points.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
