"""Page- and row-preserving source extractors."""

from __future__ import annotations

import csv
from pathlib import Path

from pypdf import PdfReader

from nexodocs_ai.knowledge_base.constants import CSV_HEADERS

from .models import CsvRow, PdfPage, ProcessingError


def extract_pdf_pages(path: Path) -> list[PdfPage]:
    """Extract non-empty PDF pages in source order."""
    try:
        reader = PdfReader(path)
    except Exception as exc:
        raise ProcessingError(f"Não foi possível ler PDF {path.name}: {exc}") from exc
    pages: list[PdfPage] = []
    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if not text or not text.strip():
            raise ProcessingError(f"PDF sem texto extraível: {path.name}, página {number}")
        pages.append(PdfPage(number, text))
    return pages


def extract_csv_rows(path: Path, document_id: str, expected_count: int) -> list[CsvRow]:
    """Extract validated UTF-8-BOM CSV data rows in source order."""
    raw = path.read_bytes()
    if not raw.startswith(b"\xef\xbb\xbf"):
        raise ProcessingError(f"CSV sem UTF-8 BOM: {path.name}")
    lines = raw.decode("utf-8-sig").splitlines()
    if not lines or any(not line for line in lines):
        raise ProcessingError(f"CSV contém linha vazia: {path.name}")
    reader = csv.DictReader(lines, delimiter=";")
    if reader.fieldnames != CSV_HEADERS[document_id]:
        raise ProcessingError(f"Cabeçalho CSV inválido: {path.name}")
    rows = [CsvRow(index, dict(row)) for index, row in enumerate(reader, start=1)]
    if len(rows) != expected_count or any(None in row.values.values() for row in rows):
        raise ProcessingError(f"Quantidade ou valores CSV inválidos: {path.name}")
    return rows
