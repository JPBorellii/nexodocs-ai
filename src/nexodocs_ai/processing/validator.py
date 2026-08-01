"""Validation of processed JSONL artifacts and manifests."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

from jsonschema import Draft202012Validator

from nexodocs_ai.knowledge_base.models import KnowledgeBaseError
from nexodocs_ai.knowledge_base.validator import assert_safe_text, validate_repository

from .constants import CHUNKS_FILENAME, MANIFEST_FILENAME, SCHEMA_VERSION
from .models import ProcessingError


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _schema(root: Path, name: str) -> dict[str, object]:
    return cast(
        dict[str, object],
        json.loads((root / "knowledge_base" / "metadata" / name).read_text(encoding="utf-8")),
    )


def _validate_schema(value: object, schema: dict[str, object]) -> None:
    validator = Draft202012Validator(cast(Any, schema))
    errors = sorted(
        validator.iter_errors(cast(Any, value)),  # pyright: ignore[reportUnknownMemberType] - jsonschema has a dynamic boundary.
        key=lambda error: list(error.path),
    )
    if errors:
        raise ProcessingError(f"Schema processado inválido: {errors[0].message}")


def load_chunks(path: Path) -> list[dict[str, object]]:
    """Load strictly formatted UTF-8 JSONL chunks."""
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf") or not raw.endswith(b"\n"):
        raise ProcessingError("chunks.jsonl deve ser UTF-8 sem BOM e terminar em LF")
    lines = raw.decode("utf-8").splitlines()
    if not lines or any(not line for line in lines):
        raise ProcessingError("chunks.jsonl contém linha vazia")
    try:
        items = [json.loads(line) for line in lines]
    except json.JSONDecodeError as exc:
        raise ProcessingError(f"JSONL inválido: {exc}") from exc
    if any(not isinstance(item, dict) for item in items):
        raise ProcessingError("Cada linha JSONL deve ser objeto")
    return cast(list[dict[str, object]], items)


def validate_processed(root: Path, output_dir: Path | None = None) -> None:
    """Validate source and processed artifacts without writing files."""
    try:
        validate_repository(root)
    except KnowledgeBaseError as exc:
        raise ProcessingError(str(exc)) from exc
    destination = output_dir or root / "knowledge_base" / "processed"
    expected = {CHUNKS_FILENAME, MANIFEST_FILENAME}
    actual: set[str] = (
        {path.name for path in destination.iterdir() if path.is_file()}
        if destination.exists()
        else set()
    )
    if actual != expected:
        raise ProcessingError(
            f"Artefatos processados divergentes: esperado {sorted(expected)}, encontrado {sorted(actual)}"
        )
    chunks_path, manifest_path = destination / CHUNKS_FILENAME, destination / MANIFEST_FILENAME
    chunks = load_chunks(chunks_path)
    manifest = cast(dict[str, object], json.loads(manifest_path.read_text(encoding="utf-8")))
    _validate_schema(manifest, _schema(root, "processed-manifest.schema.json"))
    chunk_schema = _schema(root, "processed-chunk.schema.json")
    catalog = cast(
        dict[str, object],
        json.loads(
            (root / "knowledge_base" / "metadata" / "catalog.json").read_text(encoding="utf-8")
        ),
    )
    entries = {
        str(entry["document_id"]): entry
        for entry in cast(list[dict[str, object]], catalog["documents"])
    }
    ids: set[str] = set()
    seen_rows: set[tuple[str, str]] = set()
    indices: dict[str, int] = {}
    for chunk in chunks:
        _validate_schema(chunk, chunk_schema)
        if any(value is None for value in chunk.values()):
            raise ProcessingError("Chunk contém null")
        document_id = str(chunk["document_id"])
        entry = entries.get(document_id)
        if entry is None or chunk["source_sha256"] != entry["sha256"]:
            raise ProcessingError("Chunk órfão ou hash de fonte inválido")
        text = str(chunk["text"])
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != chunk["text_sha256"]:
            raise ProcessingError("text_sha256 inválido")
        if len(text) != chunk["char_count"] or len(text.split()) != chunk["word_count"]:
            raise ProcessingError("Contadores de texto inválidos")
        try:
            assert_safe_text(text)
        except KnowledgeBaseError as exc:
            raise ProcessingError(str(exc)) from exc
        chunk_id = str(chunk["chunk_id"])
        if chunk_id in ids:
            raise ProcessingError("chunk_id duplicado")
        ids.add(chunk_id)
        expected_index = indices.get(document_id, 0)
        if chunk["chunk_index"] != expected_index:
            raise ProcessingError("chunk_index não sequencial")
        indices[document_id] = expected_index + 1
        if entry["format"] == "pdf":
            if "page_number" not in chunk or "row_number" in chunk or "row_key" in chunk:
                raise ProcessingError("Campos de formato PDF inválidos")
        else:
            if "row_number" not in chunk or "row_key" not in chunk or "page_number" in chunk:
                raise ProcessingError("Campos de formato CSV inválidos")
            row = (document_id, str(chunk["row_key"]))
            if row in seen_rows:
                raise ProcessingError("row_key duplicada")
            seen_rows.add(row)
    if set(indices) != set(entries):
        raise ProcessingError("Documento do catálogo sem chunks")
    if manifest["source_catalog_sha256"] != _sha256_bytes(
        (root / "knowledge_base" / "metadata" / "catalog.json").read_bytes()
    ):
        raise ProcessingError("Hash do catálogo inválido")
    if manifest["chunks_sha256"] != _sha256_bytes(chunks_path.read_bytes()):
        raise ProcessingError("Hash de chunks inválido")
    if manifest["schema_version"] != SCHEMA_VERSION or manifest["total_chunks"] != len(chunks):
        raise ProcessingError("Manifesto inconsistente")
    if manifest["total_characters"] != sum(cast(int, chunk["char_count"]) for chunk in chunks):
        raise ProcessingError("Total de caracteres inválido")
    if manifest["total_words"] != sum(cast(int, chunk["word_count"]) for chunk in chunks):
        raise ProcessingError("Total de palavras inválido")
    counts = Counter(str(chunk["document_id"]) for chunk in chunks)
    for document in cast(list[dict[str, object]], manifest["documents"]):
        if counts[str(document["document_id"])] != document["chunk_count"]:
            raise ProcessingError("Contagem documental inválida")
