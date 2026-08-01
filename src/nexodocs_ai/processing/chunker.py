"""Deterministic chunk construction."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from .constants import CSV_LABELS, MAX_CHARACTERS, OVERLAP_CHARACTERS, SCHEMA_VERSION
from .models import CsvRow, PdfPage
from .normalizer import normalize_text

_SECTION = re.compile(r"^(\d+)\.\s+(.+)$")


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _chunk_id(document_id: str, locator: str, index: int, text_sha256: str) -> str:
    payload = f"{SCHEMA_VERSION}\n{document_id}\n{locator}\n{index}\n{text_sha256}"
    return f"nsi-chk-{_sha256(payload)[:24]}"


def _split_at_words(text: str, maximum: int) -> list[str]:
    words = text.split()
    pieces: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if current and len(candidate) > maximum:
            pieces.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces


def split_text(text: str) -> list[str]:
    """Split text at paragraphs, then sentences, then words without breaking words."""
    paragraphs = [item.strip() for item in text.split("\n\n") if item.strip()]
    result: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidates = [paragraph]
        if len(paragraph) > MAX_CHARACTERS:
            candidates = [
                item.strip() for item in re.split(r"(?<=[.!?])\s+", paragraph) if item.strip()
            ]
        for candidate in candidates:
            units = (
                [candidate]
                if len(candidate) <= MAX_CHARACTERS
                else _split_at_words(candidate, MAX_CHARACTERS)
            )
            for unit in units:
                combined = f"{current}\n\n{unit}" if current else unit
                if current and len(combined) > MAX_CHARACTERS:
                    result.append(current)
                    current = unit
                else:
                    current = combined
    if current:
        result.append(current)
    if len(result) < 2:
        return result
    overlapped = [result[0]]
    for previous, following in zip(result, result[1:]):
        overlap_words: list[str] = []
        for word in reversed(previous.split()):
            candidate = " ".join(reversed([word, *reversed(overlap_words)]))
            if len(candidate) > OVERLAP_CHARACTERS:
                break
            overlap_words.insert(0, word)
        prefix = " ".join(overlap_words)
        while prefix and len(f"{prefix} {following}") > MAX_CHARACTERS:
            prefix = " ".join(prefix.split()[1:])
        overlapped.append(f"{prefix} {following}".strip())
    return overlapped


def _base(metadata: dict[str, object], locator: str, index: int, text: str) -> dict[str, object]:
    text_sha256 = _sha256(text)
    return {
        "schema_version": SCHEMA_VERSION,
        "chunk_id": _chunk_id(str(metadata["document_id"]), locator, index, text_sha256),
        "document_id": metadata["document_id"],
        "source_filename": metadata["filename"],
        "source_format": metadata["format"],
        "source_sha256": metadata["sha256"],
        "title": metadata["title"],
        "category": metadata["category"],
        "version": metadata["version"],
        "effective_date": metadata["effective_date"],
        "owner_area": metadata["owner_area"],
        "owner_contact": metadata["owner_contact"],
        "language": metadata["language"],
        "classification": metadata["classification"],
        "fictitious_notice": "Documento fictício criado exclusivamente para fins educacionais. Não representa políticas, dados ou operações de uma organização real.",
        "locator": locator,
        "chunk_index": index,
        "text": text,
        "text_sha256": text_sha256,
        "char_count": len(text),
        "word_count": len(text.split()),
    }


def pdf_chunks(metadata: dict[str, object], pages: Iterable[PdfPage]) -> list[dict[str, object]]:
    """Create page-bounded, section-aware chunks from extracted PDF pages."""
    chunks: list[dict[str, object]] = []
    for page in pages:
        lines = [
            line
            for line in normalize_text(page.text).splitlines()
            if not line.startswith("Página ")
        ]
        lines = [
            line
            for line in lines
            if line != f"{metadata['document_id']} | versão {metadata['version']}"
        ]
        sections: list[tuple[str | None, list[str]]] = []
        heading: str | None = None
        content: list[str] = []
        for line in lines:
            match = _SECTION.match(line)
            if match:
                if content:
                    sections.append((heading, content))
                heading, content = match.group(2), [line]
            else:
                content.append(line)
        if content:
            sections.append((heading, content))
        for section_title, section_lines in sections:
            for text in split_text(normalize_text("\n".join(section_lines))):
                locator = f"page:{page.page_number}" + (
                    f";section:{section_title.split('.', 1)[0]}" if section_title else ""
                )
                # Number is derived from the source heading, not from a synthetic title.
                if section_title:
                    number_match = re.match(r"^(\d+)\.", section_lines[0])
                    locator = f"page:{page.page_number};section:{number_match.group(1) if number_match else section_title}"
                chunk = _base(metadata, locator, len(chunks), text)
                chunk["page_number"] = page.page_number
                if section_title:
                    chunk["section_title"] = section_title
                chunks.append(chunk)
    return chunks


def csv_chunks(metadata: dict[str, object], rows: Iterable[CsvRow]) -> list[dict[str, object]]:
    """Create one self-contained chunk per CSV data row."""
    chunks: list[dict[str, object]] = []
    for row in rows:
        text = normalize_text(
            ". ".join(f"{CSV_LABELS[key]}: {value}" for key, value in row.values.items()) + "."
        )
        row_key = (
            f"{row.values['unit_id']}|{row.values['plan_id']}"
            if "unit_id" in row.values
            else row.values["area_id"]
        )
        locator = f"row:{row.row_number};key:{row_key}"
        chunk = _base(metadata, locator, len(chunks), text)
        chunk["row_number"] = row.row_number
        chunk["row_key"] = row_key
        chunks.append(chunk)
    return chunks
