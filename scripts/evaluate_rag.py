"""Run the grounded-answer offline evaluation twice in memory."""

from __future__ import annotations

import json
import os

from _project_bootstrap import bootstrap_project


def _run() -> dict[str, object]:
    root = bootstrap_project()
    from nexodocs_ai.rag.answer_provider import DeterministicFakeAnswerProvider
    from nexodocs_ai.rag.config import load_rag_config
    from nexodocs_ai.rag.evaluation import evaluate, evaluation_document, validate_minimums
    from nexodocs_ai.rag.pipeline import RagPipeline
    from nexodocs_ai.retrieval.config import load_config
    from nexodocs_ai.retrieval.embeddings import create_embedding_provider
    from nexodocs_ai.retrieval.indexer import incremental_index
    from nexodocs_ai.retrieval.qdrant_store import QdrantStore, create_client
    from nexodocs_ai.retrieval.retriever import Retriever

    retrieval = load_config(
        {
            "APP_ENV": "test",
            "EMBEDDING_PROVIDER": "fake",
            "OPENAI_EMBEDDING_DIMENSIONS": "192",
            "QDRANT_MODE": "memory",
        }
    )
    embedding = create_embedding_provider(retrieval)
    store = QdrantStore(create_client(retrieval), retrieval.collection_name, embedding.dimensions)
    incremental_index(root, retrieval, embedding, store)
    rag = load_rag_config(
        {"APP_ENV": "test", "ANSWER_PROVIDER": "fake", "RAG_MIN_CONTEXT_CHARACTERS": "1"}
    )
    metrics, cases = evaluate(
        RagPipeline(Retriever(embedding, store), DeterministicFakeAnswerProvider(rag), rag),
        root / "evals" / "rag_cases.json",
    )
    if not validate_minimums(metrics):
        raise RuntimeError("Métricas RAG abaixo do limite")
    return evaluation_document(metrics, cases)


def main() -> int:
    if os.environ.get("APP_ENV") != "test":
        raise RuntimeError("evaluate_rag.py exige APP_ENV=test")
    first, second = _run(), _run()
    if first != second:
        raise RuntimeError("Avaliação RAG não determinística")
    print(json.dumps(first, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
