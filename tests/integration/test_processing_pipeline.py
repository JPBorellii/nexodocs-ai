"""End-to-end tests for the deterministic processing pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from nexodocs_ai.knowledge_base.generator import repository_root
from nexodocs_ai.processing.models import ProcessingError
from nexodocs_ai.processing.pipeline import check, generate
from nexodocs_ai.processing.validator import load_chunks, validate_processed


def test_pipeline_generates_identical_valid_output(tmp_path: Path) -> None:
    """Two writes produce byte-identical chunks and a consistent manifest."""
    root = repository_root()
    generate(root, tmp_path)
    first = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    generate(root, tmp_path)
    assert first == {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    check(root, tmp_path)
    validate_processed(root, tmp_path)
    chunks = load_chunks(tmp_path / "chunks.jsonl")
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert len(chunks) == 44
    assert manifest["chunks_sha256"] == hashlib.sha256(first["chunks.jsonl"]).hexdigest()
    assert any(chunk["source_format"] == "pdf" and "page_number" in chunk for chunk in chunks)
    assert all("row_key" in chunk for chunk in chunks if chunk["source_format"] == "csv")


def test_validator_rejects_duplicate_chunk(tmp_path: Path) -> None:
    """Duplicate IDs are detected in a temporary corruption case."""
    root = repository_root()
    generate(root, tmp_path)
    lines = (tmp_path / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    lines[1] = lines[0]
    (tmp_path / "chunks.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ProcessingError):
        validate_processed(root, tmp_path)
