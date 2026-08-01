"""Offline deterministic processing pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import cast

from nexodocs_ai.knowledge_base.generator import repository_root
from nexodocs_ai.knowledge_base.validator import validate_repository

from .chunker import csv_chunks, pdf_chunks
from .constants import (
    CHUNKS_FILENAME,
    MANIFEST_FILENAME,
    MAX_CHARACTERS,
    OVERLAP_CHARACTERS,
    PIPELINE_VERSION,
    SCHEMA_VERSION,
    TARGET_CHARACTERS,
)
from .extractors import extract_csv_rows, extract_pdf_pages
from .models import ProcessedOutput, ProcessingError
from .validator import validate_processed


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _root(root: Path | None) -> Path:
    return root or repository_root()


def build(root: Path | None = None) -> ProcessedOutput:
    """Build in-memory deterministic processed artifacts from validated sources."""
    project_root = _root(root)
    validate_repository(project_root)
    catalog_path = project_root / "knowledge_base" / "metadata" / "catalog.json"
    catalog = cast(dict[str, object], json.loads(catalog_path.read_text(encoding="utf-8")))
    entries = sorted(
        cast(list[dict[str, object]], catalog["documents"]),
        key=lambda item: str(item["document_id"]),
    )
    chunks: list[dict[str, object]] = []
    documents: list[dict[str, object]] = []
    for entry in entries:
        source = project_root / "knowledge_base" / "source" / str(entry["filename"])
        if entry["format"] == "pdf":
            document_chunks = pdf_chunks(entry, extract_pdf_pages(source))
            counts = {"page_count": entry["page_count"]}
        else:
            document_chunks = csv_chunks(
                entry,
                extract_csv_rows(
                    source, str(entry["document_id"]), int(cast(int, entry["row_count"]))
                ),
            )
            counts = {"row_count": entry["row_count"]}
        chunks.extend(document_chunks)
        documents.append(
            {
                "document_id": entry["document_id"],
                "source_filename": entry["filename"],
                "source_sha256": entry["sha256"],
                "chunk_count": len(document_chunks),
                "character_count": sum(cast(int, c["char_count"]) for c in document_chunks),
                "word_count": sum(cast(int, c["word_count"]) for c in document_chunks),
                **counts,
            }
        )
    # Entries, pages and rows are all traversed in their stable source order.  Keeping
    # that numeric order avoids lexical anomalies such as ``row:10`` before ``row:2``.
    jsonl = "".join(
        json.dumps(chunk, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for chunk in chunks
    ).encode("utf-8")
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "source_catalog_sha256": _sha256(catalog_path.read_bytes()),
        "chunks_sha256": _sha256(jsonl),
        "chunking_strategy": {
            "name": "section-paragraph-hybrid",
            "target_characters": TARGET_CHARACTERS,
            "max_characters": MAX_CHARACTERS,
            "overlap_characters": OVERLAP_CHARACTERS,
            "cross_page_overlap": False,
            "cross_section_overlap": False,
            "csv_overlap": False,
        },
        "total_documents": len(entries),
        "total_chunks": len(chunks),
        "total_characters": sum(cast(int, chunk["char_count"]) for chunk in chunks),
        "total_words": sum(cast(int, chunk["word_count"]) for chunk in chunks),
        "documents": documents,
    }
    return ProcessedOutput(chunks, manifest)


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == content:
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
        file.write(content)
        temporary = Path(file.name)
    os.replace(temporary, path)


def _materialize(root: Path, destination: Path) -> None:
    output = build(root)
    jsonl = "".join(
        json.dumps(chunk, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for chunk in output.chunks
    ).encode("utf-8")
    manifest = (
        json.dumps(output.manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    _write(destination / CHUNKS_FILENAME, jsonl)
    _write(destination / MANIFEST_FILENAME, manifest)
    validate_processed(root, destination)


def generate(root: Path | None = None, output_dir: Path | None = None) -> None:
    """Write validated artifacts atomically, preserving unexpected files as errors."""
    project_root = _root(root)
    destination = output_dir or project_root / "knowledge_base" / "processed"
    if destination.exists() and {path.name for path in destination.iterdir() if path.is_file()} - {
        CHUNKS_FILENAME,
        MANIFEST_FILENAME,
    }:
        raise ProcessingError("Diretório processado contém arquivo inesperado")
    with tempfile.TemporaryDirectory() as temporary:
        staged = Path(temporary)
        _materialize(project_root, staged)
        _write(destination / CHUNKS_FILENAME, (staged / CHUNKS_FILENAME).read_bytes())
        _write(destination / MANIFEST_FILENAME, (staged / MANIFEST_FILENAME).read_bytes())


def check(root: Path | None = None, output_dir: Path | None = None) -> None:
    """Regenerate to temporary storage and compare checked-in artifacts byte-for-byte."""
    project_root = _root(root)
    destination = output_dir or project_root / "knowledge_base" / "processed"
    with tempfile.TemporaryDirectory() as temporary:
        staged = Path(temporary)
        _materialize(project_root, staged)
        validate_processed(project_root, destination)
        for name in (CHUNKS_FILENAME, MANIFEST_FILENAME):
            if (staged / name).read_bytes() != (destination / name).read_bytes():
                raise ProcessingError(f"Artefato processado fora de sincronização: {name}")
