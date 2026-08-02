"""Validate real-execution observability contracts without network or runtime services."""

from __future__ import annotations

import subprocess
from pathlib import Path

from _project_bootstrap import bootstrap_project
from jsonschema import Draft202012Validator


def _example_environment(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        if not separator:
            raise RuntimeError(f"Linha inv\u00e1lida em .env.example: {name}")
        values[name] = value
    return values


def main() -> int:
    """Check safe defaults, schemas, ignored output, and absent runtime artifacts."""
    root = bootstrap_project()
    from nexodocs_ai.observability.reports import (
        REPORT_SCHEMA,
        index_report,
        validate_report,
    )
    from nexodocs_ai.rag.config import load_rag_config
    from nexodocs_ai.retrieval.config import load_config
    from nexodocs_ai.retrieval.models import (
        EmbeddingRunUsage,
        IndexingOperationResult,
        IndexingResult,
    )

    values = _example_environment(root / ".env.example")
    if values.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY deve permanecer vazia em .env.example")
    retrieval = load_config(values)
    rag = load_rag_config(values)
    expected = {
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-small",
        "embedding_dimensions": 1536,
        "qdrant_mode": "local",
        "answer_provider": "openai",
        "answer_model": "gpt-5.6-luna",
        "answer_max_output_tokens": 1200,
        "embedding_retries": 0,
        "answer_retries": 0,
    }
    actual = {
        "embedding_provider": retrieval.embedding_provider,
        "embedding_model": retrieval.embedding_model,
        "embedding_dimensions": retrieval.embedding_dimensions,
        "qdrant_mode": retrieval.qdrant_mode,
        "answer_provider": rag.answer_provider,
        "answer_model": rag.openai_answer_model,
        "answer_max_output_tokens": rag.openai_answer_max_output_tokens,
        "embedding_retries": retrieval.openai_max_retries,
        "answer_retries": rag.openai_answer_max_retries,
    }
    if actual != expected:
        raise RuntimeError(
            "Configura\u00e7\u00e3o oficial de execu\u00e7\u00e3o diverge do contrato"
        )
    if rag.openai_answer_max_output_tokens == rag.max_answer_characters:
        raise RuntimeError("Limites de tokens e caracteres devem permanecer independentes")
    if values.get("QDRANT_URL") or values.get("QDRANT_API_KEY"):
        raise RuntimeError("Qdrant local n\u00e3o deve possuir URL ou chave")

    Draft202012Validator.check_schema(REPORT_SCHEMA)
    empty_usage = EmbeddingRunUsage(
        "openai",
        retrieval.embedding_model,
        retrieval.embedding_dimensions,
        retrieval.embedding_batch_size,
        0,
        0,
        0,
        0,
        None,
        None,
        True,
        (),
    )
    validate_report(
        index_report(
            IndexingOperationResult(IndexingResult(0, 44, 0, 44), empty_usage),
            retrieval.collection_name,
            0,
        )
    )

    ignored = subprocess.run(
        ["git", "check-ignore", "--no-index", "data/run-reports/probe.json"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if ignored.returncode != 0:
        raise RuntimeError("data/run-reports n\u00e3o est\u00e1 ignorado pelo Git")
    if (root / "knowledge_base" / "index" / "index-manifest.json").exists():
        raise RuntimeError("index-manifest.json n\u00e3o deve existir nesta fase")
    if (root / "data" / "qdrant").exists():
        raise RuntimeError("Qdrant persistente n\u00e3o deve existir nesta fase")
    print("Real execution observability validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
