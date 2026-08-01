"""Offline validation for canonical content and generated knowledge-base artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import cast

from jsonschema import Draft202012Validator
from pypdf import PdfReader

from .constants import (
    ALLOWED_CATEGORIES,
    ALLOWED_FORMATS,
    CSV_HEADERS,
    DOCUMENT_IDS,
    FICTITIOUS_NOTICE,
    FORBIDDEN_CELL_PREFIXES,
)
from .models import CanonicalDocument, KnowledgeBaseError, required_string

_FORBIDDEN = re.compile(
    r"(?i)(\b\d{3}\.\d{3}\.\d{3}-\d{2}\b|\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b|\bRG\b|\b\(?\d{2}\)?\s?\d{4,5}-\d{4}\b|\b\d{5}-\d{3}\b|https?://|-----BEGIN .*PRIVATE KEY-----|\b(api[_-]?key|token|secret|password)\b)"
)


def _schema(path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def _assert_schema(data: object, schema_path: Path) -> None:
    errors = sorted(
        Draft202012Validator(_schema(schema_path)).iter_errors(data),  # pyright: ignore[reportUnknownMemberType, reportArgumentType] - jsonschema exposes a dynamic JSON boundary.
        key=lambda error: list(error.path),
    )
    if errors:
        raise KnowledgeBaseError(f"Schema inválido: {errors[0].message}")


def _text_of(document: CanonicalDocument) -> str:
    return json.dumps(document.data, ensure_ascii=False)


def assert_safe_text(text: str) -> None:
    if _FORBIDDEN.search(text):
        raise KnowledgeBaseError("Conteúdo contém padrão proibido")
    for match in re.findall(r"[\w.+-]+@[\w.-]+", text):
        email = match.rstrip(".")
        if not email.endswith("@nexosaude.example"):
            raise KnowledgeBaseError(f"Email externo não permitido: {email}")


def validate_canonical_documents(documents: list[CanonicalDocument], schema_path: Path) -> None:
    """Validate schema and shared invariants of the five canonical documents."""
    if len(documents) != 5:
        raise KnowledgeBaseError("Devem existir exatamente cinco documentos canônicos")
    if {document.document_id for document in documents} != set(DOCUMENT_IDS):
        raise KnowledgeBaseError("IDs canônicos não correspondem à lista oficial")
    filenames = [document.filename for document in documents]
    if len(set(filenames)) != len(filenames):
        raise KnowledgeBaseError("Nomes de saída duplicados")
    for document in documents:
        _assert_schema(document.data, schema_path)
        data = document.data
        if data["category"] not in ALLOWED_CATEGORIES or data["format"] not in ALLOWED_FORMATS:
            raise KnowledgeBaseError("Categoria ou formato não permitido")
        if data["fictitious"] is not True or data["fictitious_notice"] != FICTITIOUS_NOTICE:
            raise KnowledgeBaseError("Aviso fictício ou flag fictitious inválidos")
        assert_safe_text(_text_of(document))
        if document.format == "csv":
            content = cast(dict[str, object], data["content"])
            columns = cast(list[str], content["columns"])
            if columns != CSV_HEADERS[document.document_id]:
                raise KnowledgeBaseError("Cabeçalho CSV canônico inválido")
            for row in cast(list[dict[str, object]], content["rows"]):
                if set(row) != set(columns):
                    raise KnowledgeBaseError("Coluna CSV inesperada")
                for value in row.values():
                    if isinstance(value, str) and value.lstrip().startswith(
                        FORBIDDEN_CELL_PREFIXES
                    ):
                        raise KnowledgeBaseError("Fórmula CSV não permitida")


def validate_generated_output(
    root: Path, output_root: Path, documents: list[CanonicalDocument]
) -> None:
    """Validate generated files, catalog hashes and cross-document contacts."""
    catalog_path = output_root / "metadata" / "catalog.json"
    catalog = cast(dict[str, object], json.loads(catalog_path.read_text(encoding="utf-8")))
    _assert_schema(catalog, root / "knowledge_base" / "metadata" / "catalog.schema.json")
    entries = cast(list[dict[str, object]], catalog["documents"])
    if [cast(str, entry["document_id"]) for entry in entries] != sorted(
        document.document_id for document in documents
    ):
        raise KnowledgeBaseError("Catálogo sem ordenação determinística")
    directory = next(
        document for document in documents if document.document_id == "NSI-TAB-DIR-001"
    )
    contacts = {
        cast(str, row["contact_email"])
        for row in cast(
            list[dict[str, object]], cast(dict[str, object], directory.data["content"])["rows"]
        )
    }
    for document, entry in zip(documents, entries, strict=True):
        path = output_root / "source" / document.filename
        if not path.exists():
            raise KnowledgeBaseError(f"Artefato ausente: {document.filename}")
        if (
            hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]
            or path.stat().st_size != entry["byte_size"]
        ):
            raise KnowledgeBaseError(f"Hash ou tamanho inválido: {document.filename}")
        if required_string(document.data, "owner_contact") not in contacts:
            raise KnowledgeBaseError("Contato proprietário ausente do diretório")
        if document.format == "pdf":
            if not path.read_bytes().startswith(b"%PDF-"):
                raise KnowledgeBaseError("Assinatura PDF inválida")
            reader = PdfReader(path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            if len(reader.pages) < 2:
                raise KnowledgeBaseError("PDF sem o mínimo de duas páginas")
            if "Documento fictício criado exclusivamente para fins educacionais." not in text:
                raise KnowledgeBaseError("PDF sem aviso fictício extraível")
            if document.document_id not in text:
                raise KnowledgeBaseError("PDF sem document_id extraível")
            assert_safe_text(text)
        else:
            raw = path.read_bytes()
            if not raw.startswith(b"\xef\xbb\xbf"):
                raise KnowledgeBaseError("CSV deve usar UTF-8 com BOM")
            lines = raw.decode("utf-8-sig").splitlines()
            if not lines or any(not line for line in lines):
                raise KnowledgeBaseError("CSV contém linha vazia")
            reader = csv.DictReader(lines, delimiter=";")
            rows = list(reader)
            if (
                reader.fieldnames != CSV_HEADERS[document.document_id]
                or len(rows) != entry["row_count"]
            ):
                raise KnowledgeBaseError("Estrutura CSV inválida")
            assert_safe_text("\n".join(";".join(row.values()) for row in rows))


def validate_repository(root: Path) -> None:
    """Validate canonical inputs and checked-in generated artifacts without writing files."""
    from .generator import load_canonical

    documents = load_canonical(root)
    validate_generated_output(root, root / "knowledge_base", documents)
