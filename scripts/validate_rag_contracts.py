"""Validate RAG schemas, versioned prompts and offline evaluation cases."""

from __future__ import annotations

import json

from _project_bootstrap import bootstrap_project
from jsonschema import Draft202012Validator


def main() -> int:
    root = bootstrap_project()
    from nexodocs_ai.rag.prompts import load_prompts, prompt_sha256

    metadata = root / "knowledge_base" / "metadata"
    schemas = [
        "rag-generated-answer.schema.json",
        "rag-answer.schema.json",
        "rag-cases.schema.json",
        "rag-evaluation.schema.json",
    ]
    for name in schemas:
        schema = json.loads((metadata / name).read_text(encoding="utf-8"))
        Draft202012Validator(schema).check_schema(schema)
    cases = json.loads((root / "evals" / "rag_cases.json").read_text(encoding="utf-8"))
    case_schema = json.loads((metadata / "rag-cases.schema.json").read_text(encoding="utf-8"))
    errors = list(Draft202012Validator(case_schema).iter_errors(cases))  # pyright: ignore[reportUnknownMemberType] - jsonschema is a dynamic boundary.
    if errors:
        raise RuntimeError(f"Casos RAG inválidos: {errors[0].message}")
    system, answer = load_prompts(root)
    if not prompt_sha256(system, answer):
        raise RuntimeError("Hash de prompt vazio")
    print("RAG contracts validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
